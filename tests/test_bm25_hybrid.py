"""
tests/test_bm25_hybrid.py
[추가] BM25 하이브리드 검색(Sparse 후보) 단위 테스트.

[전략]
- bm25_service 는 ENV 플래그(BM25_ENABLED)로 on/off 되므로, 테스트에서
  monkeypatch 로 BM25_ENABLED 를 강제 True 로 켜고 검증한다.
- 외부 의존성(Oracle/Chroma) 없이 순수 인메모리 인덱스만 검증한다.

[검증]
1. 비활성(기본) 시 search() 가 빈 리스트(회귀 가드).
2. build_index 후 키워드 검색이 정확 매칭 상품을 최상위로 올린다.
3. 반환 hit 스키마가 chroma_service 와 동일(id/document/metadata/distance).
4. distance 가 0~1 범위, 오름차순(관련 높은 순).
5. 매칭 토큰 없는 쿼리 → 빈 리스트.
6. upsert_one/delete_one 증분 갱신 동작.
7. 한국어 부분 매칭(n-gram): "나이키"가 "나이키 운동화"를 찾는다.
"""
import importlib

import pytest


_PRODUCTS = [
    {"product_id": 1, "product_name": "나이키 줌 페가수스 운동화", "category": "신발",
     "price": 129000, "description": "러닝화", "stock": 10, "image_url": ""},
    {"product_id": 2, "product_name": "아디다스 울트라부스트", "category": "신발",
     "price": 159000, "description": "쿠션 좋은 운동화", "stock": 5, "image_url": ""},
    {"product_id": 3, "product_name": "와이드 데님 팬츠", "category": "바지",
     "price": 72000, "description": "청바지", "stock": 8, "image_url": ""},
]


@pytest.fixture
def bm25(monkeypatch):
    """BM25_ENABLED=true 로 강제하고 모듈을 리로드해 플래그를 반영한다."""
    monkeypatch.setenv("BM25_ENABLED", "true")
    monkeypatch.setenv("BM25_NGRAM", "2")
    import services.bm25_service as mod
    importlib.reload(mod)
    yield mod
    # 원복: 비활성 상태로 되돌려 다른 테스트에 영향 없게
    monkeypatch.setenv("BM25_ENABLED", "false")
    importlib.reload(mod)


def test_disabled_returns_empty(monkeypatch):
    monkeypatch.setenv("BM25_ENABLED", "false")
    import services.bm25_service as mod
    importlib.reload(mod)
    assert mod.is_enabled() is False
    assert mod.build_index(_PRODUCTS) == 0
    assert mod.search("나이키", 10) == []


def test_keyword_exact_match_top(bm25):
    bm25.build_index(_PRODUCTS)
    hits = bm25.search("나이키 줌 페가수스", 10)
    assert hits, "결과가 있어야 한다"
    # 가장 관련 높은(distance 최소) 상품이 나이키(id=1)여야 한다
    assert hits[0]["metadata"]["product_id"] == 1


def test_hit_schema_matches_chroma(bm25):
    bm25.build_index(_PRODUCTS)
    hits = bm25.search("운동화", 10)
    assert hits
    h = hits[0]
    # chroma_service.search_similar 과 동일 키 집합
    assert set(h.keys()) == {"id", "document", "metadata", "distance"}
    assert isinstance(h["id"], str)
    assert "product_id" in h["metadata"]


def test_distance_range_and_order(bm25):
    bm25.build_index(_PRODUCTS)
    hits = bm25.search("운동화", 10)
    dists = [h["distance"] for h in hits]
    assert all(0.0 <= d <= 1.0 for d in dists)
    assert dists == sorted(dists), "distance 오름차순이어야 한다"


def test_no_match_returns_empty(bm25):
    bm25.build_index(_PRODUCTS)
    # 코퍼스에 전혀 없는 토큰
    assert bm25.search("xyzzy12345", 10) == []


def test_upsert_and_delete(bm25):
    bm25.build_index(_PRODUCTS)
    # 신규 상품 추가 → 검색됨
    bm25.upsert_one({"product_id": 99, "product_name": "퓨마 스웨이드", "category": "신발",
                     "price": 89000, "description": "클래식 스니커즈", "stock": 3, "image_url": ""})
    hits = bm25.search("퓨마 스웨이드", 10)
    assert any(h["metadata"]["product_id"] == 99 for h in hits)
    # 삭제 → 더 이상 검색 안 됨
    bm25.delete_one(99)
    hits2 = bm25.search("퓨마 스웨이드", 10)
    assert all(h["metadata"]["product_id"] != 99 for h in hits2)


def test_korean_partial_ngram(bm25):
    bm25.build_index(_PRODUCTS)
    # "나이키"만으로 "나이키 줌 페가수스 운동화"(id=1) 매칭
    hits = bm25.search("나이키", 10)
    assert any(h["metadata"]["product_id"] == 1 for h in hits)


# ════════════════════════════════════════════════════════════════════════
# [RRF 융합] graph.rag_pipeline._merge_text_rrf / _rrf_k 단위 테스트
# ────────────────────────────────────────────────────────────────────────
# BM25(Sparse) + Dense(벡터) 후보를 '순위'만으로 결합하는 RRF 융합의 회귀 가드.
# 핵심 증명: 절대 distance 스케일(특히 BM25 의 'Top-1 always 0.0')이 최종 순위를
#            왜곡하지 않고, '양쪽에서 검증된' 상품이 상위로 올라간다(진짜 하이브리드).
# rag_pipeline 은 services 체인을 import 하므로, 여기서는 지역 import 로 격리한다
# (융합 함수 자체는 Oracle/Chroma 무의존 순수 로직).
# ════════════════════════════════════════════════════════════════════════
def _hit(pid, distance, doc=None, category=None):
    """테스트용 hit(chroma/bm25 공통 스키마) 생성."""
    return {
        "id": str(pid),
        "document": doc if doc is not None else f"doc-{pid}",
        "metadata": {"product_id": pid, "category": category or ""},
        "distance": distance,
    }


def test_rrf_overlap_ranked_first():
    """양쪽(Dense+Sparse) 모두에 등장한 상품이 최상위로 온다(하이브리드 핵심)."""
    from graph.rag_pipeline import _merge_text_rrf
    dense = [_hit(1, 0.10), _hit(2, 0.20)]
    sparse = [_hit(2, 0.0), _hit(3, 0.50)]   # 2번이 양쪽에 등장
    merged = _merge_text_rrf(dense, sparse)
    ids = [h["id"] for h in merged]
    assert ids[0] == "2", ids                # 두 리스트 합산이 가장 큼 → 최상위
    assert set(ids) == {"1", "2", "3"}       # 합집합, 중복 제거


def test_rrf_scale_free_no_top1_bias():
    """[편향 제거 증명] Sparse Top-1(distance=0.0)이 무조건 1위가 되지 않는다.

    구 방식(_merge_dedup, distance 최소 채택)에선 sparse 의 distance=0.0 상품이
    항상 1위였다. RRF 는 절대 점수를 배제하므로, '양쪽에서 검증된' 상품이
    'sparse 단독 Top-1' 상품을 이긴다.
    """
    from graph.rag_pipeline import _merge_text_rrf
    dense = [_hit("A", 0.05)]                 # A 는 dense 에만, rank1
    sparse = [_hit("B", 0.0), _hit("A", 0.30)]  # B 는 sparse Top-1(구 편향 승자), A 는 sparse rank2
    merged = _merge_text_rrf(dense, sparse)
    ids = [h["id"] for h in merged]
    assert ids[0] == "A", ids                # 양쪽 등장 A 가 sparse-top B 를 이김
    assert ids[1] == "B"


def test_rrf_empty_sparse_fallback():
    """Sparse 가 비어도 Dense 순위를 그대로 보존한다(폴백 안전)."""
    from graph.rag_pipeline import _merge_text_rrf
    dense = [_hit(1, 0.1), _hit(2, 0.2), _hit(3, 0.3)]
    merged = _merge_text_rrf(dense, [])
    assert [h["id"] for h in merged] == ["1", "2", "3"]
    assert _merge_text_rrf([], []) == []      # 둘 다 비면 빈 리스트


def test_rrf_schema_and_order_preserved():
    """반환 hit 스키마(id/document/metadata/distance) 유지 + distance 오름차순 + 대표는 Dense."""
    from graph.rag_pipeline import _merge_text_rrf
    dense = [_hit(1, 0.10, doc="dense-doc", category="신발")]
    sparse = [_hit(1, 0.0, doc="sparse-doc", category="바지"), _hit(9, 0.4)]
    merged = _merge_text_rrf(dense, sparse)
    for h in merged:
        assert {"id", "document", "metadata", "distance"} <= set(h.keys())
        assert "rrf_score" in h               # 융합 부가 키
    dists = [h["distance"] for h in merged]
    assert dists == sorted(dists)             # distance 오름차순(관련 높은 순)
    rep1 = next(h for h in merged if h["id"] == "1")
    assert rep1["document"] == "dense-doc"    # 겹치면 Dense 쪽 dict 를 대표로 채택


def test_rrf_k_env_fallback(monkeypatch):
    """RRF_K 파싱: 정상값 반영 / 0·비정수 → 60 폴백(호출 시점 getenv 라 reload 불필요)."""
    from graph.rag_pipeline import _rrf_k
    monkeypatch.setenv("RRF_K", "30");  assert _rrf_k() == 30
    monkeypatch.setenv("RRF_K", "0");   assert _rrf_k() == 60
    monkeypatch.setenv("RRF_K", "abc"); assert _rrf_k() == 60
    monkeypatch.delenv("RRF_K", raising=False); assert _rrf_k() == 60
