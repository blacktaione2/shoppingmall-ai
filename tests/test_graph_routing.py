"""
tests/test_graph_routing.py
LangGraph 라우팅/흐름 오프라인 단위테스트.

[전략]
- 외부 의존(GPT/Oracle/Chroma)을 전부 monkeypatch 로 격리한다.
  · classify_node 의 분류 LLM → 고정 IntentResult 반환
  · 각 핸들러가 호출하는 서비스 함수 → 가짜 값 반환
- 검증 포인트:
  1) 인텐트별로 올바른 핸들러 노드를 거치는가 (raw_answer 가 해당 노드 산출인가)
  2) 게스트 ORDER_INQUIRY → guest_block 으로 분기되는가
  3) semantic → rag_hits 가 State 에 실려 guard 까지 전달되는가
  4) 프리픽스가 제거됐는가 ([추천]/[검색결과] 등이 final_answer 에 없음)
"""
import asyncio

import pytest

from schemas.intent_schema import IntentResult, IntentType, Entities
from graph.builder import build_graph
from graph import nodes


def _make_intent(intent: IntentType, **entity_kwargs) -> IntentResult:
    emotion = entity_kwargs.pop("emotion", None)
    return IntentResult(
        intent=intent,
        entities=Entities(**entity_kwargs),
        emotion=emotion,
        confidence=0.95,
    )


def _patch_classify(monkeypatch, intent_result: IntentResult):
    """classify_node 를 고정 결과 반환으로 치환."""
    async def fake_classify(state):
        return {"intent_result": intent_result}
    monkeypatch.setattr(nodes, "classify_node", fake_classify, raising=True)
    # build_graph 싱글톤 캐시 초기화 (치환된 노드로 재컴파일되도록)
    import graph.builder as b
    b._compiled_app = None


def _run(state):
    # checkpointer 가 달린 그래프는 thread_id 가 필요하다.
    # 라우팅 단위테스트는 멀티턴 누적이 목적이 아니므로 1회성 thread_id 를 부여한다.
    import uuid
    app = build_graph()
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}
    return asyncio.run(app.ainvoke(state, config=config))


# ────────────────────────────────────────────────────────────────────────
# STRUCTURED_QUERY
# ────────────────────────────────────────────────────────────────────────
def test_structured_routing(monkeypatch):
    _patch_classify(monkeypatch, _make_intent(
        IntentType.STRUCTURED_QUERY, category="신발", price_max=50000,
    ))

    fake_products = [
        {"product_name": "운동화", "category": "신발", "price": 39000, "stock": 5},
    ]

    def fake_search(**kwargs):
        return fake_products
    monkeypatch.setattr(nodes, "search_products_structured", fake_search, raising=True)

    out = _run({"question": "5만원 이하 신발", "is_guest": False, "history": []})

    assert "운동화" in out["final_answer"]
    assert "39,000원" in out["final_answer"]
    # 프리픽스 제거 확인
    assert not out["final_answer"].startswith("[검색결과]")


# ────────────────────────────────────────────────────────────────────────
# SEMANTIC_SEARCH (rag_hits 전달 + guard pass)
# ────────────────────────────────────────────────────────────────────────
def test_semantic_routing_and_rag_hits(monkeypatch):
    _patch_classify(monkeypatch, _make_intent(IntentType.SEMANTIC_SEARCH))

    fake_hits = [
        {
            "id": "1",
            "document": "겨울 패딩",
            "metadata": {"product_name": "롱패딩", "price": 99000, "category": "상의"},
            "distance": 0.1,
            "score": 0.9,
        },
    ]

    # [RAG 고도화] semantic_node 는 이제 rag_pipeline.search_and_rerank 를 호출
    import graph.rag_pipeline as rag_pipeline
    async def fake_search_and_rerank(query, top_n=4, **kwargs):
        return fake_hits
    async def fake_rag(question, hits, history=None):
        return "롱패딩을 추천드려요. 가격은 99,000원입니다."
    async def fake_resolve_attr(question, history):
        return None, None

    monkeypatch.setattr(rag_pipeline, "search_and_rerank", fake_search_and_rerank, raising=True)
    monkeypatch.setattr(nodes.rag_service, "generate_rag_response", fake_rag, raising=True)
    monkeypatch.setattr(nodes, "_resolve_product_attribute_query", fake_resolve_attr, raising=True)

    out = _run({"question": "겨울에 따뜻한 거", "is_guest": False, "history": []})

    assert "롱패딩" in out["final_answer"]
    # rag_hits 가 State 에 실렸는지
    assert out["rag_hits"] == fake_hits
    assert not out["final_answer"].startswith("[추천]")


# ────────────────────────────────────────────────────────────────────────
# SEMANTIC_SEARCH 0건 → 고정 안내 (GPT 미호출)
# ────────────────────────────────────────────────────────────────────────
def test_semantic_no_hits(monkeypatch):
    _patch_classify(monkeypatch, _make_intent(IntentType.SEMANTIC_SEARCH))

    import graph.rag_pipeline as rag_pipeline
    async def fake_search_and_rerank(query, top_n=4, **kwargs):
        return []
    async def fake_rag(question, hits, history=None):
        raise AssertionError("0건이면 RAG 호출하면 안 됨")
    async def fake_resolve_attr(question, history):
        return None, None

    monkeypatch.setattr(rag_pipeline, "search_and_rerank", fake_search_and_rerank, raising=True)
    monkeypatch.setattr(nodes.rag_service, "generate_rag_response", fake_rag, raising=True)
    monkeypatch.setattr(nodes, "_resolve_product_attribute_query", fake_resolve_attr, raising=True)

    out = _run({"question": "존재하지않는상품xyz", "is_guest": False, "history": []})
    assert "찾지 못했" in out["final_answer"]
    assert out["rag_hits"] == []


# ────────────────────────────────────────────────────────────────────────
# FAQ
# ────────────────────────────────────────────────────────────────────────
def test_faq_routing(monkeypatch):
    _patch_classify(monkeypatch, _make_intent(IntentType.FAQ, keywords=["배송"]))

    def fake_faq_sync(question, intent_result):
        return "[FAQ] 배송은 며칠 걸려요?\n\n평균 2~3일 소요됩니다."
    monkeypatch.setattr(nodes, "_search_faq_sync", fake_faq_sync, raising=True)

    out = _run({"question": "배송 얼마나 걸려요?", "is_guest": False, "history": []})
    assert "2~3일" in out["final_answer"]


# ────────────────────────────────────────────────────────────────────────
# ORDER_INQUIRY (로그인) — 전체 목록
# ────────────────────────────────────────────────────────────────────────
def test_order_routing_member(monkeypatch):
    _patch_classify(monkeypatch, _make_intent(IntentType.ORDER_INQUIRY))

    def fake_orders(member_id):
        assert member_id == 42
        return [{
            "order_id": "ORD-20260601-0001",
            "order_date": "2026-06-01",
            "items": [{"product_name": "셔츠", "quantity": 1, "price": 20000}],
            "total_price": 20000,
            "status": "배송중",
        }]
    monkeypatch.setattr(nodes, "fetch_orders", fake_orders, raising=True)

    out = _run({
        "question": "내 주문 어디까지 왔어",
        "is_guest": False, "member_id": 42, "history": [],
    })
    assert "ORD-20260601-0001" in out["final_answer"]
    assert "배송중" in out["final_answer"]


# ────────────────────────────────────────────────────────────────────────
# ORDER_INQUIRY (게스트) → guest_block 차단
# ────────────────────────────────────────────────────────────────────────
def test_order_guest_blocked(monkeypatch):
    _patch_classify(monkeypatch, _make_intent(IntentType.ORDER_INQUIRY))

    def fake_orders(member_id):
        raise AssertionError("게스트는 order_node 에 도달하면 안 됨")
    monkeypatch.setattr(nodes, "fetch_orders", fake_orders, raising=True)

    out = _run({"question": "내 주문 조회", "is_guest": True, "history": []})
    assert "로그인" in out["final_answer"]


# ────────────────────────────────────────────────────────────────────────
# COMPLAINT — 단정 사실 표현 시 guard 안내문 append
# ────────────────────────────────────────────────────────────────────────
def test_complaint_guard_appends_guidance(monkeypatch):
    _patch_classify(monkeypatch, _make_intent(IntentType.COMPLAINT, emotion="분노"))

    # complaint_node 가 내부에서 LCEL 체인을 호출하므로, 노드 자체를 치환해
    # "환불 완료" 단정 표현을 강제 → guard 가 GUIDANCE_SUFFIX 를 붙이는지 검증
    async def fake_complaint(state):
        return {"raw_answer": "정말 죄송합니다. 환불 처리 완료되었습니다."}
    monkeypatch.setattr(nodes, "complaint_node", fake_complaint, raising=True)
    import graph.builder as b
    b._compiled_app = None

    out = _run({"question": "환불해줘 화나", "is_guest": False, "history": []})
    assert "주문조회" in out["final_answer"]  # GUIDANCE_SUFFIX


# ────────────────────────────────────────────────────────────────────────
# SMALL_TALK — 정상 응답 pass-through
# ────────────────────────────────────────────────────────────────────────
def test_small_talk_passthrough(monkeypatch):
    _patch_classify(monkeypatch, _make_intent(IntentType.SMALL_TALK))

    async def fake_small_talk(state):
        return {"raw_answer": "안녕하세요! 무엇을 도와드릴까요?"}
    monkeypatch.setattr(nodes, "small_talk_node", fake_small_talk, raising=True)
    import graph.builder as b
    b._compiled_app = None

    out = _run({"question": "안녕", "is_guest": False, "history": []})
    assert "안녕하세요" in out["final_answer"]
