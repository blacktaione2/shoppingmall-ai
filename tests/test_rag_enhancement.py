"""
tests/test_rag_enhancement.py
RAG 고도화(재랭킹 + 출처 표시) 테스트.

[검증]
1. 재랭킹 OFF(기본) → 원본 순서 top_n 컷, distance 정규화 score
2. 재랭킹 ON → Cohere 결과 순서로 재정렬 + relevance score (Cohere mock)
3. 재랭킹 API 실패 → 원본 순서 폴백 (파이프라인 안 멈춤)
4. score 통일 규칙: 높을수록 관련 (distance 정규화 / rerank_score)
5. hits_to_sources 변환: metadata → SourceItem dict
6. search_and_rerank: 후보 확대(ON) vs top_n(OFF)
"""
import asyncio

import pytest

from services import rerank_service
from graph import rag_pipeline


def _hit(pid, name, price, distance, document="doc"):
    return {
        "id": str(pid),
        "document": document,
        "metadata": {"product_id": pid, "product_name": name,
                     "category": "상의", "price": price},
        "distance": distance,
    }


# ────────────────────────────────────────────────────────────────────────
# 1) 재랭킹 OFF → 원본 순서 유지
# ────────────────────────────────────────────────────────────────────────
def test_rerank_disabled_keeps_order(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "false")
    hits = [_hit(1, "A", 1000, 0.1), _hit(2, "B", 2000, 0.2), _hit(3, "C", 3000, 0.3)]
    out = asyncio.run(rerank_service.rerank("질문", hits, top_n=2))
    assert [h["id"] for h in out] == ["1", "2"]   # 원본 순서, top_n=2 컷


# ────────────────────────────────────────────────────────────────────────
# 2) 재랭킹 ON → Cohere 결과 순서 (mock)
# ────────────────────────────────────────────────────────────────────────
def test_rerank_enabled_reorders(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("RERANK_PROVIDER", "cohere")
    monkeypatch.setenv("COHERE_API_KEY", "dummy")

    hits = [_hit(1, "A", 1000, 0.1), _hit(2, "B", 2000, 0.2), _hit(3, "C", 3000, 0.3)]

    # _rerank_cohere 를 직접 치환: index 2,0 순으로 재정렬됐다고 가정
    def fake_cohere(query, hits_in, top_n):
        order = [2, 0]   # C, A 순
        result = []
        for rank, idx in enumerate(order[:top_n]):
            h = dict(hits_in[idx])
            h["rerank_score"] = 0.9 - rank * 0.1
            result.append(h)
        return result
    monkeypatch.setattr(rerank_service, "_rerank_cohere", fake_cohere, raising=True)

    out = asyncio.run(rerank_service.rerank("질문", hits, top_n=2))
    assert [h["id"] for h in out] == ["3", "1"]   # C(idx2), A(idx0)
    assert out[0]["rerank_score"] == 0.9


# ────────────────────────────────────────────────────────────────────────
# 3) 재랭킹 API 실패 → 원본 폴백
# ────────────────────────────────────────────────────────────────────────
def test_rerank_failure_falls_back(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("RERANK_PROVIDER", "cohere")
    monkeypatch.setenv("COHERE_API_KEY", "dummy")

    def boom(query, hits_in, top_n):
        raise RuntimeError("Cohere 503")
    monkeypatch.setattr(rerank_service, "_rerank_cohere", boom, raising=True)

    hits = [_hit(1, "A", 1000, 0.1), _hit(2, "B", 2000, 0.2)]
    out = asyncio.run(rerank_service.rerank("질문", hits, top_n=2))
    assert [h["id"] for h in out] == ["1", "2"]   # 폴백: 원본 순서


# ────────────────────────────────────────────────────────────────────────
# 4) score 통일 규칙 (높을수록 관련)
# ────────────────────────────────────────────────────────────────────────
def test_attach_scores_unifies(monkeypatch):
    hits = [
        {"id": "1", "distance": 0.0},                     # → score 1.0
        {"id": "2", "distance": 1.0},                     # → score 0.5
        {"id": "3", "rerank_score": 0.8, "distance": 9},  # → score 0.8 (rerank 우선)
    ]
    out = rerank_service.attach_scores(hits)
    assert out[0]["score"] == 1.0
    assert out[1]["score"] == 0.5
    assert out[2]["score"] == 0.8


# ────────────────────────────────────────────────────────────────────────
# 5) hits_to_sources 변환
# ────────────────────────────────────────────────────────────────────────
def test_hits_to_sources():
    hits = [
        {"metadata": {"product_id": 7, "product_name": "패딩", "category": "상의", "price": 99000},
         "score": 0.95, "distance": 0.05},
    ]
    sources = rag_pipeline.hits_to_sources(hits)
    assert sources[0]["product_id"] == 7
    assert sources[0]["product_name"] == "패딩"
    assert sources[0]["price"] == 99000
    assert sources[0]["score"] == 0.95


# ────────────────────────────────────────────────────────────────────────
# 6) search_and_rerank: 후보 확대(ON) vs top_n(OFF)
# ────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────
# 7) [버그 수정] STOCK=0(품절) 후보는 최종 결과에서 제외돼야 한다
# ────────────────────────────────────────────────────────────────────────
def test_search_and_rerank_excludes_soldout(monkeypatch):
    async def fake_embed(q):
        return [0.0] * 1536

    def _hit_with_stock(pid, name, stock):
        h = _hit(pid, name, 1000 * pid, 0.1 * pid)
        h["metadata"]["stock"] = stock
        return h

    async def fake_search(emb, n_results):
        return [
            _hit_with_stock(1, "품절상품", stock=0),
            _hit_with_stock(2, "재고있는상품", stock=5),
        ]

    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setattr(rag_pipeline.embed_service, "get_embedding", fake_embed, raising=True)
    monkeypatch.setattr(rag_pipeline.chroma_service, "search_similar", fake_search, raising=True)

    hits = asyncio.run(rag_pipeline.search_and_rerank("q", top_n=4))

    ids = [h["id"] for h in hits]
    assert "1" not in ids   # 품절 상품 제외
    assert "2" in ids       # 재고 있는 상품은 유지


def test_search_and_rerank_missing_stock_metadata_not_excluded(monkeypatch):
    """stock 메타데이터가 아예 없으면(정보 없음) 품절로 오해석해 제외하면 안 된다."""
    async def fake_embed(q):
        return [0.0] * 1536

    async def fake_search(emb, n_results):
        return [_hit(1, "상품", 1000, 0.1)]   # metadata에 stock 키 자체가 없음

    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setattr(rag_pipeline.embed_service, "get_embedding", fake_embed, raising=True)
    monkeypatch.setattr(rag_pipeline.chroma_service, "search_similar", fake_search, raising=True)

    hits = asyncio.run(rag_pipeline.search_and_rerank("q", top_n=4))
    assert len(hits) == 1


def test_search_candidate_count(monkeypatch):
    captured = {}
    async def fake_embed(q):
        return [0.0] * 1536
    async def fake_search(emb, n_results):
        captured["n_results"] = n_results
        return [_hit(i, f"P{i}", 1000 * i, 0.1 * i) for i in range(1, n_results + 1)]
    monkeypatch.setattr(rag_pipeline.embed_service, "get_embedding", fake_embed, raising=True)
    monkeypatch.setattr(rag_pipeline.chroma_service, "search_similar", fake_search, raising=True)

    # OFF → 후보 = top_n
    monkeypatch.setenv("RERANK_ENABLED", "false")
    asyncio.run(rag_pipeline.search_and_rerank("q", top_n=4))
    assert captured["n_results"] == 4

    # ON → 후보 = RERANK_CANDIDATES(기본 10)
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("RERANK_CANDIDATES", "10")
    monkeypatch.setenv("COHERE_API_KEY", "dummy")
    monkeypatch.setattr(rerank_service, "_rerank_cohere",
                        lambda q, h, n: h[:n], raising=True)
    asyncio.run(rag_pipeline.search_and_rerank("q", top_n=4))
    assert captured["n_results"] == 10


# ────────────────────────────────────────────────────────────────────────
# 7) end-to-end: SEMANTIC 응답에 sources 가 실린다 (chat.py)
# ────────────────────────────────────────────────────────────────────────
def test_chat_response_includes_sources(monkeypatch):
    import importlib
    from schemas.intent_schema import IntentResult, IntentType, Entities
    from graph import nodes
    import graph.builder as builder
    import graph.rag_pipeline as rp

    fake_hits = [{
        "id": "7", "document": "롱패딩",
        "metadata": {"product_id": 7, "product_name": "롱패딩", "category": "상의", "price": 99000},
        "distance": 0.05, "score": 0.95,
    }]

    # classify=SEMANTIC 고정
    ir = IntentResult(intent=IntentType.SEMANTIC_SEARCH, entities=Entities(), confidence=0.9)
    async def fake_classify(state):
        return {"intent_result": ir}
    async def fake_search_and_rerank(query, top_n=4, **kwargs):
        return fake_hits
    async def fake_rag(question, hits, history=None):
        return "롱패딩을 추천드려요. 99,000원입니다."
    async def fake_resolve_attr(question, history):
        return None, None

    monkeypatch.setattr(nodes, "classify_node", fake_classify, raising=True)
    monkeypatch.setattr(rp, "search_and_rerank", fake_search_and_rerank, raising=True)
    monkeypatch.setattr(nodes.rag_service, "generate_rag_response", fake_rag, raising=True)
    monkeypatch.setattr(nodes, "_resolve_product_attribute_query", fake_resolve_attr, raising=True)
    builder._compiled_app = None

    import routers.chat as chat
    importlib.reload(chat)
    monkeypatch.setattr(chat, "resolve_chat_token", lambda t: 1, raising=True)
    monkeypatch.setattr(chat, "save_chat_history", lambda *a, **k: None, raising=True)

    from schemas.chat_schema import ChatRequest
    resp = asyncio.run(chat.process_chat_pipeline(
        ChatRequest(chat_token="tok", question="겨울 옷", history=[])
    ))

    assert resp.sources is not None
    assert resp.sources[0].product_id == 7
    assert resp.sources[0].product_name == "롱패딩"
    assert resp.sources[0].score == 0.95
