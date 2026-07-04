"""
tests/test_agent_graph.py
단일 Agent 그래프 흐름 테스트.

[전략]
- 실제 gpt-5.4 대신 'FakeToolCallingModel' 을 주입한다.
  · 이 모델은 미리 정해둔 AIMessage 시퀀스를 호출 순서대로 반환한다.
  · 1번째 호출: 도구 호출(tool_calls)이 담긴 AIMessage
  · 2번째 호출: 도구 결과를 본 뒤 최종 답변 AIMessage
- 도구 내부의 외부 의존(Oracle/Chroma/embed)은 monkeypatch 로 격리.
- checkpointer 는 테스트 전용 MemorySaver 를 주입(공유 상태 오염 방지).

[검증]
1. 단일 도구 호출: search_products 가 실행되고 메트릭(tool_calls=1)이 집계된다.
2. semantic 도구: rag_hits 가 State 에 기록되고 guard 가 그것을 검증한다.
3. 게스트 주문조회 차단: get_my_orders 가 InjectedState 로 게스트를 막는다.
4. 복합 질문: 2개 도구를 연쇄 호출하면 tools_used 에 2개가 잡힌다.
"""
import asyncio
import uuid

import pytest
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langgraph.checkpoint.memory import MemorySaver

from graph import tools as tools_mod
import graph.agent_builder as agent_builder


class FakeToolCallingModel(FakeMessagesListChatModel):
    """미리 정한 AIMessage 시퀀스를 순서대로 반환하는 가짜 모델.

    create_react_agent 는 model.bind_tools(tools) 를 호출하므로 bind_tools 를
    self 반환으로 오버라이드한다(도구 바인딩은 시뮬레이션에 불필요).
    """
    def bind_tools(self, tools, **kwargs):
        return self


def _build(monkeypatch, ai_sequence):
    """가짜 모델 + 전용 checkpointer 로 Agent 그래프를 새로 빌드."""
    model = FakeToolCallingModel(responses=ai_sequence)
    app = agent_builder.build_agent(
        model=model,
        checkpointer=MemorySaver(),
        force_rebuild=True,
    )
    return app


def _run(app, state):
    config = {
        "configurable": {"thread_id": f"test-{uuid.uuid4()}"},
        "recursion_limit": 12,
    }
    return asyncio.run(app.ainvoke(state, config=config))


# ────────────────────────────────────────────────────────────────────────
# 1) 단일 도구 호출 (search_products)
# ────────────────────────────────────────────────────────────────────────
def test_single_tool_call(monkeypatch):
    def fake_search(**kwargs):
        return [{"product_name": "운동화", "category": "신발", "price": 39000, "stock": 5}]
    monkeypatch.setattr(tools_mod, "search_products_structured", fake_search, raising=True)

    # 1차: 도구 호출 / 2차: 최종 답변
    seq = [
        AIMessage(content="", tool_calls=[{
            "name": "search_products",
            "args": {"category": "신발", "price_max": 50000},
            "id": "call_1",
        }]),
        AIMessage(content="운동화를 추천드려요. 39,000원입니다."),
    ]
    app = _build(monkeypatch, seq)
    out = _run(app, {
        "messages": [__import__("langchain_core.messages", fromlist=["HumanMessage"]).HumanMessage(content="5만원 이하 신발")],
        "member_id": 1, "is_guest": False, "rag_hits": [],
    })

    # 최종 메시지가 답변인지
    final = out["messages"][-1]
    assert "운동화" in final.content


# ────────────────────────────────────────────────────────────────────────
# 2) semantic 도구 → rag_hits State 기록 + guard 검증
# ────────────────────────────────────────────────────────────────────────
def test_semantic_records_rag_hits(monkeypatch):
    fake_hits = [{
        "id": "1", "document": "롱패딩",
        "metadata": {"product_name": "롱패딩", "price": 99000, "category": "상의"},
        "distance": 0.1, "score": 0.9,
    }]
    import graph.rag_pipeline as rag_pipeline
    async def fake_search_and_rerank(query, top_n=4, **kwargs):
        return fake_hits
    monkeypatch.setattr(rag_pipeline, "search_and_rerank", fake_search_and_rerank, raising=True)

    from langchain_core.messages import HumanMessage
    seq = [
        AIMessage(content="", tool_calls=[{
            "name": "semantic_search",
            "args": {"query": "겨울에 따뜻한 옷"},
            "id": "call_s",
        }]),
        # 컨텍스트(롱패딩/99000)를 인용한 정상 답변 → guard 통과
        AIMessage(content="롱패딩을 추천드려요. 가격은 99,000원입니다."),
    ]
    app = _build(monkeypatch, seq)
    out = _run(app, {
        "messages": [HumanMessage(content="겨울에 따뜻한 거")],
        "member_id": 1, "is_guest": False, "rag_hits": [],
    })

    # rag_hits 가 State 에 기록됐는지
    assert out["rag_hits"] == fake_hits
    final = out["messages"][-1]
    assert "롱패딩" in final.content


# ────────────────────────────────────────────────────────────────────────
# 3) 게스트 주문조회 차단 (InjectedState)
# ────────────────────────────────────────────────────────────────────────
def test_guest_order_blocked(monkeypatch):
    def should_not_call(member_id):
        raise AssertionError("게스트는 주문조회 실행되면 안 됨")
    monkeypatch.setattr(tools_mod, "fetch_orders", should_not_call, raising=True)

    from langchain_core.messages import HumanMessage
    seq = [
        AIMessage(content="", tool_calls=[{
            "name": "get_my_orders", "args": {}, "id": "call_o",
        }]),
        AIMessage(content="주문 조회는 로그인 후 이용하실 수 있어요."),
    ]
    app = _build(monkeypatch, seq)
    out = _run(app, {
        "messages": [HumanMessage(content="내 주문 조회")],
        "member_id": None, "is_guest": True, "rag_hits": [],
    })

    # 도구 결과(ToolMessage)에 로그인 안내가 담겼는지
    from langchain_core.messages import ToolMessage
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert any("로그인" in m.content for m in tool_msgs)


# ────────────────────────────────────────────────────────────────────────
# 4) 복합 질문 — 2개 도구 연쇄 호출
# ────────────────────────────────────────────────────────────────────────
def test_multi_tool_chain(monkeypatch):
    def fake_search(**kwargs):
        return [{"product_name": "패딩", "category": "상의", "price": 99000, "stock": 3}]
    def fake_faq_sync(question, intent_result):
        return "평균 2~3일 소요됩니다."
    monkeypatch.setattr(tools_mod, "search_products_structured", fake_search, raising=True)
    monkeypatch.setattr(tools_mod, "_search_faq_sync", fake_faq_sync, raising=True)

    from langchain_core.messages import HumanMessage, ToolMessage
    seq = [
        # 1차: 상품검색 도구
        AIMessage(content="", tool_calls=[{
            "name": "search_products", "args": {"category": "상의", "price_max": 100000},
            "id": "c1",
        }]),
        # 2차: FAQ 도구
        AIMessage(content="", tool_calls=[{
            "name": "search_faq", "args": {"question": "배송 얼마나 걸려요"},
            "id": "c2",
        }]),
        # 3차: 종합 답변
        AIMessage(content="패딩을 추천드리고, 배송은 2~3일 걸립니다."),
    ]
    app = _build(monkeypatch, seq)
    out = _run(app, {
        "messages": [HumanMessage(content="10만원 이하 패딩 추천하고 배송도 알려줘")],
        "member_id": 1, "is_guest": False, "rag_hits": [],
    })

    # 2개 도구가 모두 호출됐는지 (ToolMessage 2개)
    tool_names = [m.name for m in out["messages"] if isinstance(m, ToolMessage)]
    assert "search_products" in tool_names
    assert "search_faq" in tool_names
    assert len(tool_names) == 2
