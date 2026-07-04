"""
tests/test_multi_agent_graph.py
멀티 Agent(Supervisor) 그래프 흐름 테스트.

[전략]
- supervisor 노드는 get_intent_llm 을 직접 호출하므로, 테스트에서는
  supervisor 노드를 'monkeypatch' 로 치환해 라우팅 결정을 제어한다.
  · 라우팅 결정 시퀀스를 미리 정해두고, 호출될 때마다 하나씩 반환한다.
- sub-agent(product/support)는 가짜 모델(FakeToolCallingModel)을 주입해
  도구 호출을 시뮬레이션한다.
- 도구 내부 외부 의존(Oracle/Chroma/embed)은 monkeypatch 로 격리.

[검증]
1. supervisor → product_agent 라우팅 → 도구 실행 → FINISH 종합답변
2. rag_hits 보존: product_agent 의 semantic_search 가 기록한 hits 가
   supervisor 왕복을 거쳐 guard 까지 전달된다.
3. support_agent 게스트 주문차단
4. 핑퐁 방지: 라우팅 상한(MAX_ROUTING) 도달 시 강제 FINISH
"""
import asyncio
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langgraph.checkpoint.memory import MemorySaver

from graph import tools as tools_mod
import graph.multi_agent_builder as mab
from graph.multi_agent_state import RouteDecision


class FakeToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _patch_supervisor(monkeypatch, decisions):
    """supervisor 노드를 미리 정한 RouteDecision 시퀀스로 치환.

    decisions: [(next, final_answer), ...] 순서대로 소비.
    routing_count 증가도 동일하게 모사한다.
    """
    seq = list(decisions)
    call_idx = {"i": 0}

    async def fake_supervisor(state):
        i = call_idx["i"]
        call_idx["i"] += 1
        routing_count = state.get("routing_count", 0)
        nxt, final = seq[i]
        update = {"next_agent": nxt, "routing_count": routing_count + 1}
        if nxt == "FINISH":
            update["final_answer"] = final
        return update

    monkeypatch.setattr(mab, "_supervisor_node", fake_supervisor, raising=True)


def _build(monkeypatch, sub_ai_sequence):
    model = FakeToolCallingModel(responses=sub_ai_sequence)
    return mab.build_multi_agent(
        model=model, checkpointer=MemorySaver(), force_rebuild=True,
    )


def _run(app, state):
    config = {
        "configurable": {"thread_id": f"test-{uuid.uuid4()}"},
        "recursion_limit": 20,
    }
    return asyncio.run(app.ainvoke(state, config=config))


# ────────────────────────────────────────────────────────────────────────
# 1) supervisor → product_agent → FINISH
# ────────────────────────────────────────────────────────────────────────
def test_route_to_product_then_finish(monkeypatch):
    def fake_search(**kwargs):
        return [{"product_name": "패딩", "category": "상의", "price": 99000, "stock": 3}]
    monkeypatch.setattr(tools_mod, "search_products_structured", fake_search, raising=True)

    # supervisor: 1) product_agent, 2) FINISH(종합답변)
    _patch_supervisor(monkeypatch, [
        ("product_agent", ""),
        ("FINISH", "패딩을 추천드려요. 99,000원입니다."),
    ])
    # product_agent sub-agent: 1) 도구호출, 2) (내부 종료용) 답변
    sub_seq = [
        AIMessage(content="", tool_calls=[{
            "name": "search_products", "args": {"category": "상의"}, "id": "c1"}]),
        AIMessage(content="패딩 찾았습니다."),
    ]
    app = _build(monkeypatch, sub_seq)
    out = _run(app, {
        "messages": [HumanMessage(content="10만원 이하 패딩 추천")],
        "member_id": 1, "is_guest": False, "rag_hits": [], "routing_count": 0,
    })

    assert "패딩" in out["final_answer"]


# ────────────────────────────────────────────────────────────────────────
# 2) rag_hits 보존 → guard 전달
# ────────────────────────────────────────────────────────────────────────
def test_rag_hits_preserved_to_guard(monkeypatch):
    fake_hits = [{
        "id": "1", "document": "롱패딩",
        "metadata": {"product_name": "롱패딩", "price": 99000, "category": "상의"},
        "distance": 0.1, "score": 0.9,
    }]
    import graph.rag_pipeline as rag_pipeline
    async def fake_search_and_rerank(query, top_n=4, **kwargs):
        return fake_hits
    monkeypatch.setattr(rag_pipeline, "search_and_rerank", fake_search_and_rerank, raising=True)

    _patch_supervisor(monkeypatch, [
        ("product_agent", ""),
        # 컨텍스트(롱패딩/99000) 인용 → guard 통과하는 답변
        ("FINISH", "롱패딩을 추천드려요. 가격은 99,000원입니다."),
    ])
    sub_seq = [
        AIMessage(content="", tool_calls=[{
            "name": "semantic_search", "args": {"query": "겨울옷"}, "id": "cs"}]),
        AIMessage(content="검색 완료."),
    ]
    app = _build(monkeypatch, sub_seq)
    out = _run(app, {
        "messages": [HumanMessage(content="겨울에 따뜻한 거")],
        "member_id": 1, "is_guest": False, "rag_hits": [], "routing_count": 0,
    })

    # rag_hits 가 끝까지 보존됐는지
    assert out["rag_hits"] == fake_hits
    assert "롱패딩" in out["final_answer"]


# ────────────────────────────────────────────────────────────────────────
# 3) support_agent 게스트 주문차단
# ────────────────────────────────────────────────────────────────────────
def test_support_guest_order_blocked(monkeypatch):
    def should_not_call(member_id):
        raise AssertionError("게스트는 주문조회 실행되면 안 됨")
    monkeypatch.setattr(tools_mod, "fetch_orders", should_not_call, raising=True)

    _patch_supervisor(monkeypatch, [
        ("support_agent", ""),
        ("FINISH", "주문 조회는 로그인 후 이용하실 수 있어요."),
    ])
    sub_seq = [
        AIMessage(content="", tool_calls=[{
            "name": "get_my_orders", "args": {}, "id": "co"}]),
        AIMessage(content="안내 완료."),
    ]
    app = _build(monkeypatch, sub_seq)
    out = _run(app, {
        "messages": [HumanMessage(content="내 주문 조회")],
        "member_id": None, "is_guest": True, "rag_hits": [], "routing_count": 0,
    })

    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert any("로그인" in m.content for m in tool_msgs)


# ────────────────────────────────────────────────────────────────────────
# 4) 핑퐁 방지 — 라우팅 상한 도달 시 강제 FINISH
# ────────────────────────────────────────────────────────────────────────
def test_routing_limit_forces_finish(monkeypatch):
    """실제 _supervisor_node 를 쓰되, 라우팅 LLM 과 강제답변 LLM 을 가짜로 치환.

    supervisor 가 매번 product_agent 로만 보내려 해도 MAX_ROUTING 도달 시
    강제 FINISH 로 빠져 무한 루프가 끝나는지 검증.
    """
    def fake_search(**kwargs):
        return [{"product_name": "X", "category": "Y", "price": 1000, "stock": 1}]
    monkeypatch.setattr(tools_mod, "search_products_structured", fake_search, raising=True)

    # 라우팅 LLM: 항상 product_agent (FINISH 를 안 함 → 상한으로만 종료돼야 함)
    class AlwaysProductLLM:
        def with_structured_output(self, schema):
            return self
        async def ainvoke(self, messages):
            return RouteDecision(next="product_agent", final_answer="")
    monkeypatch.setattr(mab, "get_intent_llm", lambda temperature=0.0: AlwaysProductLLM(),
                        raising=True)

    # 강제 FINISH 답변 생성 LLM
    class ForceAnswerLLM:
        async def ainvoke(self, messages):
            return AIMessage(content="상한 도달로 마무리합니다.")
    monkeypatch.setattr(mab, "get_main_llm", lambda temperature=0.3: ForceAnswerLLM(),
                        raising=True)

    # sub-agent 는 매번 도구 1회 호출 후 답변 (충분히 길게 반복 응답 공급)
    sub_seq = [
        AIMessage(content="", tool_calls=[{
            "name": "search_products", "args": {}, "id": f"c{i}"}])
        if i % 2 == 0 else AIMessage(content="ok")
        for i in range(40)
    ]
    app = _build(monkeypatch, sub_seq)
    out = _run(app, {
        "messages": [HumanMessage(content="무한 루프 유발")],
        "member_id": 1, "is_guest": False, "rag_hits": [], "routing_count": 0,
    })

    # 강제 FINISH 답변이 나오고 종료됐는지
    assert "마무리" in out["final_answer"]
    # 라우팅 횟수가 상한 + 1(강제 finish 호출 포함) 수준에서 멈췄는지
    assert out["routing_count"] <= mab.MAX_ROUTING + 1
