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
