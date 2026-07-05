"""
tests/test_semantic_price_superlative.py
semantic_node 의 "이전 언급 상품 가격 비교" 결정적 경로 오프라인 단위테스트.

[배경]
"그 중 제일 싼 건?" 같은 질문을 LLM 자유생성에 맡기면 히스토리 속 여러 상품
가격 중 실제 최저가가 아닌 걸 고르는 오류가 재현됐다(수동 테스트로 확인).
그래서 semantic_node 진입 시 초입에서 히스토리에 실제 언급된 상품을 재조회해
파이썬으로 결정적으로 정렬하는 경로를 추가했다 — 이 테스트는 그 경로가:
1) 정확히 최저가/최고가를 고르는지
2) 매칭되는 상품이 없으면 기존 임베딩 검색+RAG 경로로 정상 폴백하는지
를 검증한다. Oracle/GPT 외부 의존은 전부 monkeypatch 로 격리한다(오프라인).
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
    # 오답이었던 상품(양털 후리스, 98,000)이 정답으로 나오면 안 됨
    assert "오버핏 양털 후리스" not in result["raw_answer"]
    # [회귀 방지] rag_hits 가 guard 가 기대하는 ChromaDB 형식(metadata.price/product_name)
    # 이 아니면 실제 배포에서 정상 답변도 환각으로 오판돼 안전문구로 대체됐던 버그가 있었다.
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

    # 히스토리 없음(첫 대화) → 매칭될 상품이 없어 결정적 경로 미작동, 기존 경로로 진행돼야 함
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
    """조합 추천 문장에 가격 없이 이름만 언급된 상품(티셔츠)은 후보에서 제외돼야 한다.

    [실제 재현된 버그] "화이트 베이직 크루넥 티셔츠 + 오버핏 양털 후리스 조합도
    좋아요"처럼 가격 없이 이름만 곁가지로 언급됐는데, DB 상 실제 가격(35,000원)이
    다른 상품들보다 낮아서 "최저가"로 잘못 뽑혔었다.
    """
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


def test_single_reference_question_uses_last_bot_turn(monkeypatch):
    """'방금 말한 거 얼마야?' → 직전 봇 발화(니트 1개만 언급)에서 결정적으로 답해야 한다."""
    monkeypatch.setattr(nodes, "fetch_all_products", lambda: _PRODUCTS, raising=True)

    history = [
        {"role": "user", "text": "겨울에 따뜻한 옷 추천해줘"},
        {"role": "bot", "text": _BOT_TURN_TEXT},
        {"role": "user", "text": "그 중에 제일 싼 건?"},
        {"role": "bot", "text": "이전에 안내해드린 상품 중 가장 저렴한 건 슬림핏 터틀넥 니트(68,000원)이에요."},
    ]
    state = {"question": "방금 말한 거 얼마라고?", "history": history}
    result = asyncio.run(nodes.semantic_node(state))

    assert "슬림핏 터틀넥 니트" in result["raw_answer"]
    assert "68,000" in result["raw_answer"]
    assert _validate_semantic_answer(result["raw_answer"], result["rag_hits"]) is True


def test_ambiguous_but_question_names_exact_product(monkeypatch):
    """실제 재현된 버그: 직전 발화에 3개가 언급돼 모호해도, 이번 질문 자체에
    정확한 상품명이 있으면 그걸로 좁혀서 답해야 한다(되묻지 않아야 함).
    """
    monkeypatch.setattr(nodes, "fetch_all_products", lambda: _PRODUCTS, raising=True)

    history = _history_with_bot_turn()   # 패딩/니트/후리스 3개 동시 언급 → 원래는 모호
    state = {"question": "오버핏 양털 후리스 재고 있어?", "history": history}
    result = asyncio.run(nodes.semantic_node(state))

    assert "오버핏 양털 후리스" in result["raw_answer"]
    assert "재고 22개" in result["raw_answer"]
    assert result["raw_answer"] != "어떤 상품을 말씀하시는지 다시 알려주시겠어요?"


def test_ambiguous_and_question_has_no_product_name_still_asks(monkeypatch):
    """질문에도 특정 상품명이 없으면(예: '재고 있어?'만) 여전히 되물어야 한다(회귀 방지)."""
    monkeypatch.setattr(nodes, "fetch_all_products", lambda: _PRODUCTS, raising=True)

    history = _history_with_bot_turn()
    state = {"question": "재고 있어?", "history": history}
    result = asyncio.run(nodes.semantic_node(state))

    assert result["raw_answer"] == "어떤 상품을 말씀하시는지 다시 알려주시겠어요?"


def test_single_reference_asks_clarifying_question_when_ambiguous(monkeypatch):
    """직전 봇 발화에 상품이 2개 이상 언급되면 검색으로 폴백하지 않고 바로 되물어야 한다.

    [버그 수정] 예전엔 기존 검색 경로로 폴백했는데, "재고 있어?" 같은 참조 질문
    자체가 새 상품 검색이 아니라서 폴백 검색이 품절 상품 등 엉뚱한 결과를
    끌고 오는 문제가 있었다. 이제 검색을 아예 시도하지 않고 바로 되묻는다.
    """
    monkeypatch.setattr(nodes, "fetch_all_products", lambda: _PRODUCTS, raising=True)

    called = {"search": False}

    async def fake_search_and_rerank(question, top_n=4, member_id=None):
        called["search"] = True
        return [{"product_id": 1, "product_name": "화이트 베이직 크루넥 티셔츠", "price": 35000}]

    import graph.rag_pipeline as rag_pipeline
    monkeypatch.setattr(rag_pipeline, "search_and_rerank", fake_search_and_rerank, raising=True)

    # 직전 봇 발화(_BOT_TURN_TEXT)에 3개 상품이 동시에 언급돼 모호함
    history = _history_with_bot_turn()
    state = {"question": "방금 말한 거 얼마야?", "history": history}
    result = asyncio.run(nodes.semantic_node(state))

    assert called["search"] is False
    assert result["raw_answer"] == "어떤 상품을 말씀하시는지 다시 알려주시겠어요?"
    assert result["rag_hits"] == []


def test_stock_question_without_reference_marker_still_works(monkeypatch):
    """실제 재현된 버그: '방금/아까/그거' 없이 '재고 있어?'만 물어도 직전 상품을
    참조해야 한다 — classify_node 가 이미 구체적 조건 없는 질문만 여기로
    보내므로, 참조어가 없어도 이전 언급 상품 질문으로 처리해야 한다.
    """
    monkeypatch.setattr(nodes, "fetch_all_products", lambda: _PRODUCTS, raising=True)

    history = [
        {"role": "user", "text": "겨울에 따뜻한 옷 추천해줘"},
        {"role": "bot", "text": _BOT_TURN_TEXT},
        {"role": "user", "text": "그 중에 제일 싼 건?"},
        {"role": "bot", "text": "이전에 안내해드린 상품 중 가장 저렴한 건 슬림핏 터틀넥 니트(68,000원)이에요."},
    ]
    state = {"question": "재고 있어?", "history": history}
    result = asyncio.run(nodes.semantic_node(state))

    assert "슬림핏 터틀넥 니트" in result["raw_answer"]
    assert "재고 30개" in result["raw_answer"]


def test_stock_reference_question_answers_in_stock(monkeypatch):
    """'그거 재고 있어?' → 직전 발화 속 단일 상품의 재고를 결정적으로 답해야 한다."""
    monkeypatch.setattr(nodes, "fetch_all_products", lambda: _PRODUCTS, raising=True)

    history = [
        {"role": "user", "text": "그 중에 제일 싼 건?"},
        {"role": "bot", "text": "이전에 안내해드린 상품 중 가장 저렴한 건 슬림핏 터틀넥 니트(68,000원)이에요."},
    ]
    state = {"question": "그거 재고 있어?", "history": history}
    result = asyncio.run(nodes.semantic_node(state))

    assert "슬림핏 터틀넥 니트" in result["raw_answer"]
    assert "재고 30개" in result["raw_answer"]
    assert _validate_semantic_answer(result["raw_answer"], result["rag_hits"]) is True


def test_stock_reference_question_reports_soldout(monkeypatch):
    """실제 재현 버그: 방금 언급한 상품이 품절이어도(stock=0) 정확히 "품절"이라고
    답해야 한다 — stock>0 필터 때문에 후보에서 빠져 엉뚱한 검색으로 새던 문제였다.
    """
    products = [dict(p) for p in _PRODUCTS]
    products[1]["stock"] = 0  # 슬림핏 터틀넥 니트를 품절 상태로 변경
    monkeypatch.setattr(nodes, "fetch_all_products", lambda: products, raising=True)

    history = [
        {"role": "user", "text": "그 중에 제일 싼 건?"},
        {"role": "bot", "text": "이전에 안내해드린 상품 중 가장 저렴한 건 슬림핏 터틀넥 니트(68,000원)이에요."},
    ]
    state = {"question": "그거 재고 있어?", "history": history}
    result = asyncio.run(nodes.semantic_node(state))

    assert "슬림핏 터틀넥 니트" in result["raw_answer"]
    assert "품절" in result["raw_answer"]


def test_non_superlative_question_unaffected(monkeypatch):
    """가격 최상급 표현이 없는 일반 질문은 기존 경로를 그대로 타야 한다(회귀 방지)."""
    monkeypatch.setattr(nodes, "fetch_all_products", lambda: _PRODUCTS, raising=True)

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