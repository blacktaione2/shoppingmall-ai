"""
tests/test_semantic_price_superlative.py
semantic_node 의 결정적 처리 경로 오프라인 단위테스트.

[구성]
1. 가격 비교("그 중 제일 싼 건?") — _detect_price_superlative/_match_history_products
   로 히스토리 속 실제 언급 상품을 파이썬으로 결정적으로 정렬한다. LLM 자유생성이
   가격을 잘못 비교하는 오류가 실제로 재현됐었다(수동 테스트).
2. 특정 상품 가격/재고 질문("오버핏 양털 후리스 재고 있어?", "그거 얼마야?") —
   과거엔 키워드/토큰 텍스트 매칭으로 처리했는데, 카탈로그에 같은 단어를 공유하는
   상품이 있거나(예: "오버핏"이 두 상품에 공통) 히스토리가 비어있을 때 계속 깨졌다.
   지금은 LLM이 실제 카탈로그를 보고 직접 판단하는 _resolve_product_attribute_query
   로 교체했다 — 이 테스트들은 그 함수를 monkeypatch 로 mock 해서, semantic_node 가
   그 결과를 올바르게 소비(가격/재고 답변 생성, 모호하면 되묻기, 아니면 검색 폴백)
   하는지만 검증한다(LLM 판단 자체는 여기서 검증하지 않음 — 외부 의존이라 오프라인
   불가능하고, 프롬프트 품질은 실제 대화로 검증해야 함).
Oracle/GPT 외부 의존은 전부 monkeypatch 로 격리한다(오프라인).
"""
import asyncio

import pytest

from graph import nodes
from pipeline.hallucination_guard import _validate_semantic_answer


_PRODUCTS = [
    {"product_id": 15, "product_name": "경량 패딩 점퍼", "price": 135000, "stock": 8},
    {"product_id": 3, "product_name": "슬림핏 터틀넥 니트", "price": 68000, "stock": 30},
    {"product_id": 13, "product_name": "오버핏 양털 후리스", "price": 98000, "stock": 22},
]

_BOT_TURN_TEXT = (
    "겨울에 따뜻하게 입으시려면 경량 패딩 점퍼 135,000원, "
    "슬림핏 터틀넥 니트 68,000원, 오버핏 양털 후리스 98,000원을 추천드려요."
)


def _history_with_bot_turn():
    return [
        {"role": "user", "text": "겨울에 따뜻한 옷 추천해줘"},
        {"role": "bot", "text": _BOT_TURN_TEXT},
    ]


def test_cheap_superlative_picks_actual_lowest_price(monkeypatch):
    """'제일 싼 건?' → 히스토리 3개 중 실제 최저가(터틀넥 니트 68,000)를 골라야 한다."""
    monkeypatch.setattr(nodes, "fetch_all_products", lambda: _PRODUCTS, raising=True)

    state = {"question": "그 중에 제일 싼 건?", "history": _history_with_bot_turn()}
    result = asyncio.run(nodes.semantic_node(state))

    assert "슬림핏 터틀넥 니트" in result["raw_answer"]
    assert "68,000" in result["raw_answer"]
    assert "오버핏 양털 후리스" not in result["raw_answer"]
    assert _validate_semantic_answer(result["raw_answer"], result["rag_hits"]) is True


def test_expensive_superlative_picks_actual_highest_price(monkeypatch):
    """'제일 비싼 건?' → 실제 최고가(경량 패딩 점퍼 135,000)를 골라야 한다."""
    monkeypatch.setattr(nodes, "fetch_all_products", lambda: _PRODUCTS, raising=True)

    state = {"question": "그 중 가장 비싼 건?", "history": _history_with_bot_turn()}
    result = asyncio.run(nodes.semantic_node(state))

    assert "경량 패딩 점퍼" in result["raw_answer"]
    assert "135,000" in result["raw_answer"]
    assert _validate_semantic_answer(result["raw_answer"], result["rag_hits"]) is True


def test_soldout_product_excluded_even_if_mentioned(monkeypatch):
    """STOCK=0 상품은 히스토리에 언급됐어도 비교 후보에서 제외돼야 한다."""
    products = _PRODUCTS + [
        {"product_id": 99, "product_name": "품절임시상품", "price": 1000, "stock": 0},
    ]
    monkeypatch.setattr(nodes, "fetch_all_products", lambda: products, raising=True)
    history = _history_with_bot_turn()
    history[-1]["text"] += " 품절임시상품 1,000원도 있었지만 품절이에요."

    state = {"question": "그 중에 제일 싼 건?", "history": history}
    result = asyncio.run(nodes.semantic_node(state))

    assert "품절임시상품" not in result["raw_answer"]
    assert "슬림핏 터틀넥 니트" in result["raw_answer"]


def test_superlative_without_matching_history_falls_back_to_rag(monkeypatch):
    """히스토리에 언급된 상품이 없으면 기존 임베딩 검색+RAG 경로로 정상 폴백해야 한다."""
    monkeypatch.setattr(nodes, "fetch_all_products", lambda: _PRODUCTS, raising=True)

    called = {"search": False, "rag": False}

    async def fake_search_and_rerank(question, top_n=4, member_id=None):
        called["search"] = True
        return [{"product_id": 1, "product_name": "화이트 베이직 크루넥 티셔츠", "price": 35000}]

    async def fake_generate_rag_response(question, hits, history=None):
        called["rag"] = True
        return "티셔츠를 추천드려요."

    import graph.rag_pipeline as rag_pipeline
    monkeypatch.setattr(rag_pipeline, "search_and_rerank", fake_search_and_rerank, raising=True)
    monkeypatch.setattr(nodes.rag_service, "generate_rag_response", fake_generate_rag_response, raising=True)

    state = {"question": "제일 싼 거 뭐 있어?", "history": []}
    result = asyncio.run(nodes.semantic_node(state))

    assert called["search"] is True
    assert called["rag"] is True
    assert result["raw_answer"] == "티셔츠를 추천드려요."


_BOT_TURN_WITH_COMBO_MENTION = (
    "겨울에 따뜻하게 입으시려면 이 3가지를 특히 추천드려요. "
    "경량 패딩 점퍼 135,000원이 좋아요. "
    "슬림핏 터틀넥 니트 68,000원도 추천드려요. "
    "오버핏 양털 후리스 98,000원도 잘 맞아요. "
    "편안한 캐주얼 느낌을 원하시면 화이트 베이직 크루넥 티셔츠 + "
    "오버핏 양털 후리스 조합도 좋아요."
)


def test_combo_mention_without_price_is_not_counted_as_candidate(monkeypatch):
    """조합 추천 문장에 가격 없이 이름만 언급된 상품(티셔츠)은 후보에서 제외돼야 한다."""
    products = _PRODUCTS + [
        {"product_id": 1, "product_name": "화이트 베이직 크루넥 티셔츠", "price": 35000, "stock": 45},
    ]
    monkeypatch.setattr(nodes, "fetch_all_products", lambda: products, raising=True)

    history = [
        {"role": "user", "text": "겨울에 따뜻한 옷 추천해줘"},
        {"role": "bot", "text": _BOT_TURN_WITH_COMBO_MENTION},
    ]
    state = {"question": "그 중 제일 싼 건?", "history": history}
    result = asyncio.run(nodes.semantic_node(state))

    assert "화이트 베이직 크루넥 티셔츠" not in result["raw_answer"]
    assert "슬림핏 터틀넥 니트" in result["raw_answer"]
    assert "68,000" in result["raw_answer"]
    assert _validate_semantic_answer(result["raw_answer"], result["rag_hits"]) is True


def _mock_resolve(monkeypatch, target, attribute):
    async def fake_resolve(question, history):
        return target, attribute
    monkeypatch.setattr(nodes, "_resolve_product_attribute_query", fake_resolve, raising=True)


def test_price_attribute_resolved_answers_deterministically(monkeypatch):
    """_resolve_product_attribute_query 가 상품을 찾으면(price), 검색 없이 바로 답해야 한다."""
    target = _PRODUCTS[1]
    _mock_resolve(monkeypatch, target, "price")

    state = {"question": "그거 얼마야?", "history": _history_with_bot_turn()}
    result = asyncio.run(nodes.semantic_node(state))

    assert "슬림핏 터틀넥 니트" in result["raw_answer"]
    assert "68,000" in result["raw_answer"]
    assert _validate_semantic_answer(result["raw_answer"], result["rag_hits"]) is True


def test_stock_attribute_resolved_in_stock(monkeypatch):
    """_resolve_product_attribute_query 가 상품을 찾으면(stock, 재고 있음), 재고 수를 답해야 한다."""
    target = _PRODUCTS[1]
    _mock_resolve(monkeypatch, target, "stock")

    state = {"question": "오버핏 양털 후리스 재고 있어?", "history": []}
    result = asyncio.run(nodes.semantic_node(state))

    assert "슬림핏 터틀넥 니트" in result["raw_answer"]
    assert "재고 30개" in result["raw_answer"]


def test_stock_attribute_resolved_soldout(monkeypatch):
    """실제 재현 버그: 방금 언급한 상품이 품절(stock=0)이어도 정확히 "품절"이라고 답해야 한다."""
    target = dict(_PRODUCTS[1])
    target["stock"] = 0
    _mock_resolve(monkeypatch, target, "stock")

    state = {"question": "그거 재고 있어?", "history": _history_with_bot_turn()}
    result = asyncio.run(nodes.semantic_node(state))

    assert "슬림핏 터틀넥 니트" in result["raw_answer"]
    assert "품절" in result["raw_answer"]


def test_attribute_detected_but_target_unresolved_asks_clarifying(monkeypatch):
    """가격/재고 질문인 건 맞지만 상품이 특정 안 되면(모호), 검색으로 폴백하지 않고 바로 되물어야 한다."""
    _mock_resolve(monkeypatch, None, "stock")

    called = {"search": False}

    async def fake_search_and_rerank(question, top_n=4, member_id=None):
        called["search"] = True
        return []

    import graph.rag_pipeline as rag_pipeline
    monkeypatch.setattr(rag_pipeline, "search_and_rerank", fake_search_and_rerank, raising=True)

    state = {"question": "재고 있어?", "history": _history_with_bot_turn()}
    result = asyncio.run(nodes.semantic_node(state))

    assert called["search"] is False
    assert result["raw_answer"] == "어떤 상품을 말씀하시는지 다시 알려주시겠어요?"
    assert result["rag_hits"] == []


def test_not_attribute_question_falls_through_to_search(monkeypatch):
    """가격/재고 질문이 아니면(_resolve_product_attribute_query 가 (None, None)) 기존 검색+RAG 경로로 진행해야 한다."""
    _mock_resolve(monkeypatch, None, None)

    async def fake_search_and_rerank(question, top_n=4, member_id=None):
        return [{"product_id": 3, "product_name": "슬림핏 터틀넥 니트", "price": 68000}]

    async def fake_generate_rag_response(question, hits, history=None):
        return "터틀넥 니트를 추천드려요."

    import graph.rag_pipeline as rag_pipeline
    monkeypatch.setattr(rag_pipeline, "search_and_rerank", fake_search_and_rerank, raising=True)
    monkeypatch.setattr(nodes.rag_service, "generate_rag_response", fake_generate_rag_response, raising=True)

    state = {"question": "겨울에 따뜻한 니트 있어?", "history": _history_with_bot_turn()}
    result = asyncio.run(nodes.semantic_node(state))

    assert result["raw_answer"] == "터틀넥 니트를 추천드려요."
