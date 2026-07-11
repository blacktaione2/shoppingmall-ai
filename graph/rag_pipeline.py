"""
graph/rag_pipeline.py
SEMANTIC 검색 + 재랭킹 공통 파이프라인.

[목적]
- semantic_node(라우터 그래프)와 semantic_search(Agent tool)가 '동일한' 검색 흐름을
  공유하도록 공통 함수로 추출한다(결정 ②: 양쪽 공통 적용 → 일관성 + 비교 공정성).

[흐름]
    질문 → 임베딩 → (회원+활성화 시) 개인화 취향 벡터 혼합 → ChromaDB 후보(top-k 확대)
    → 재랭킹 → top-n → score 부여
    · 재랭킹 OFF: 후보를 그대로 top-n 컷 + distance 정규화 score
    · 재랭킹 ON : Cohere 로 재정렬 + relevance score

[개인화 — 질문 임베딩 혼합]
- member_id 가 있고 personalization_service.is_enabled() 이면, 구매 이력 기반
  취향 벡터를 질문 임베딩에 혼합(blend_vectors)한 뒤 검색한다.
- 취향 벡터 조회/혼합 실패는 예외를 삼키고 원본 질문 벡터로 안전 폴백한다
  (개인화가 검색 자체를 막아서는 안 됨).
- CLIP 이미지 검색에는 적용하지 않는다(질문 원문 그대로 인코딩 — 임베딩 공간이 다름).
- 반환 hits 에는 통일된 'score'(높을수록 관련)가 부여된다(attach_scores).

[멀티모달 — 플래그 기반 이미지 검색 병합]
- CLIP_SERVING_ENABLED=true 이면 텍스트 후보(products) + 이미지 후보(products_image)를
  product_id 기준으로 병합·중복제거한 뒤 재랭킹한다.
- false(기본)면 텍스트 후보만 사용 → 기존 동작과 100% 동일(메모리/지연 0).
- 나중에 서버 여유가 생기면 .env 플래그만 true 로 바꾸면 코드 변경 없이 활성화된다.
- 두 컬렉션의 distance 는 임베딩 공간이 달라 직접 비교가 부정확하므로,
  병합 후 재랭킹(RERANK_ENABLED=true, document 기반)을 함께 켜는 것을 권장한다.

[하이브리드 검색 — Sparse(BM25) + Dense(벡터) Fusion]
- BM25_ENABLED=true 이면 벡터 후보에 BM25 키워드 후보를 병합한다.
  · 고유명사/모델명 정확 매칭을 BM25 가 보완 → recall 향상.
  · bm25_service.search() 가 ChromaDB 와 동일한 hit 스키마를 돌려주므로
    여기서도 _merge_dedup() 을 그대로 재사용한다(코드 추가 최소화).
- false(기본)면 BM25 경로를 타지 않아 기존 동작과 100% 동일.
- 병합 후 재랭킹을 함께 켜면 Dense/Sparse 의 서로 다른 distance 스케일을
  Cohere relevance score 로 재정렬해 공정하게 최종 순위를 정한다(권장 조합).
"""
import asyncio
import logging
import os

from services import embed_service, chroma_service, rerank_service, clip_service, personalization_service, bm25_service

logger = logging.getLogger(__name__)


def _merge_dedup(text_hits: list[dict], image_hits: list[dict]) -> list[dict]:
    """텍스트/이미지 후보를 product_id(=id) 기준으로 병합·중복제거한다.

    - 같은 상품이 양쪽에서 나오면 'distance 가 더 작은(=더 가까운)' 쪽을 남긴다.
    - 한쪽에만 있으면 그대로 포함한다.
    - 반환 순서는 distance 오름차순(가까운 순). 이후 재랭킹/컷이 최종 순위를 정한다.
    """
    best: dict[str, dict] = {}
    for hit in list(text_hits) + list(image_hits):
        key = str(hit.get("id"))
        prev = best.get(key)
        if prev is None or _distance_of(hit) < _distance_of(prev):
            best[key] = hit
    merged = list(best.values())
    merged.sort(key=_distance_of)
    return merged


def _distance_of(hit: dict) -> float:
    """정렬/비교용 distance 안전 추출(없으면 매우 큰 값으로 취급)."""
    d = hit.get("distance")
    try:
        return float(d)
    except (TypeError, ValueError):
        return float("inf")


def _rrf_k() -> int:
    """RRF 상수 k(.env RRF_K, 기본 60). 파싱 실패/0 이하면 60 폴백.

    k 는 순위 1등과 2등의 점수 격차를 완만하게 만드는 상수다(원논문 기본 60).
    작을수록 상위 순위에 가중이 쏠리고, 클수록 순위 차이가 평탄해진다.
    """
    try:
        val = int(os.getenv("RRF_K", "60"))
    except (TypeError, ValueError):
        return 60
    return val if val > 0 else 60


def _merge_text_rrf(dense_hits: list[dict], sparse_hits: list[dict]) -> list[dict]:
    """Dense(벡터/CLIP 병합 결과) + BM25(Sparse) 후보를 RRF 로 융합한다(텍스트 하이브리드 전용).

        RRF(id) = Σ_L  1 / (k + rank_L(id))     (id 가 등장한 각 리스트 L 에 대해 합산)

    [왜 RRF 인가 — 융합 편향 제거]
    - BM25 자체 정규화 점수를 distance 최소값 병합에 태우면 키워드 1등이 항상
      distance=0.0 이 되어 벡터 결과를 무조건 누른다(순수 BM25 로 퇴화).
    - RRF 는 절대 점수를 융합에서 완전히 배제하고 '순위'만 사용하는 스케일-프리
      알고리즘이라, Sparse/Dense 점수의 스케일·분포가 달라도 공정하게 합쳐진다.
      두 리스트 모두에서 상위인 상품이 자연히 최상위로 올라간다(진짜 상호보완).
    - CLIP 멀티모달 병합용 _merge_dedup 은 별도 함수로 유지한다(텍스트 하이브리드 전용).

    [입력 가정] 각 리스트는 이미 관련도순(distance 오름차순)으로 정렬돼 들어온다.
                들어온 순서를 rank(1-based)로 사용한다. 리스트 내부 중복 id 는 최상위만 반영.

    [반환] [{id, document, metadata, distance, rrf_score}, ...]  distance 오름차순.
      · distance: RRF 점수를 (1 - rrf/max_rrf)로 인코딩한 '표시용' 값. 융합에 쓰인 값이
        아니라, 하류 파이프라인(rerank_service.attach_scores 의 1/(1+distance) 정규화 등)과
        스키마·정렬을 호환시키기 위한 후처리다. RERANK ON 이면 Cohere 가 다시 재정렬한다.
      · rrf_score: 원본 RRF 점수(로깅/디버그용 부가 키, 하류 스키마엔 영향 없음).
    """
    k = _rrf_k()
    rep: dict[str, dict] = {}      # id → 대표 hit dict (dense 우선)
    rrf: dict[str, float] = {}     # id → 누적 RRF 점수

    for hits in (dense_hits, sparse_hits):
        seen_in_list: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            key = str(hit.get("id"))
            if key in seen_in_list:
                continue           # 같은 리스트 내 중복은 최상위 랭크만 반영
            seen_in_list.add(key)
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (k + rank)
            if key not in rep:     # 대표 dict: 먼저 순회하는 dense 를 우선 채택
                rep[key] = hit

    if not rrf:
        return []

    max_rrf = max(rrf.values())
    merged: list[dict] = []
    for key, score in rrf.items():
        hit = dict(rep[key])
        # RRF 점수 내림차순 == distance 오름차순이 되도록 표시용 distance 인코딩.
        hit["distance"] = (1.0 - (score / max_rrf)) if max_rrf > 0 else 0.0
        hit["rrf_score"] = score
        merged.append(hit)
    merged.sort(key=lambda h: h["distance"])
    return merged


async def search_and_rerank(query: str, top_n: int = 4, member_id: int | None = None) -> list[dict]:
    """질문으로 ChromaDB 검색 후 재랭킹해 상위 top_n hits 를 반환한다.

    Args:
        query: 사용자 검색 질의(자연어).
        top_n: 최종 반환 개수(재랭킹 후).
        member_id: 로그인 회원 PK. 주어지고 개인화 ON 이면 질문 임베딩에
                   취향 벡터를 혼합한다. None(게스트/미적용)이면 기존 동작과 동일.
                   · 라우터 경로(semantic_node)만 전달한다.
                   · Agent 경로(semantic_search tool)는 비교 공정성을 위해 None 유지.
    Returns:
        [{id, document, metadata, distance, score, (rerank_score?)}, ...]
        0건이면 빈 리스트.

    [후보 확대]
        재랭킹 ON 이거나 BM25 하이브리드(BM25_ENABLED=true) ON 이면
        RERANK_CANDIDATES(기본 10)개 후보를 가져와 재정렬/융합한다.
        둘 다 OFF 면 굳이 많이 가져올 필요 없이 top_n 만 가져온다(비용/지연 절약).

    [멀티모달]
        CLIP_SERVING_ENABLED=true 면 같은 후보 개수로 이미지 컬렉션도 검색해 병합한다.
        CLIP 텍스트 인코딩/이미지 검색 실패 시 텍스트 결과로 안전 폴백(검색은 멈추지 않음).
        [경계] 개인화 혼합은 텍스트 임베딩(1536)에만 적용하고, CLIP 이미지 검색은
               질문 원문으로 수행한다(공간이 다름).
    """
    # 재랭킹 또는 BM25 하이브리드가 켜져 있으면 RERANK_CANDIDATES 만큼 넓게 가져온다.
    # RRF 는 '두 리스트 모두에서 검증된 상품'을 끌어올리는 알고리즘이라 후보 풀이
    # top_n 수준으로 얕으면 교집합 여지가 없어 융합이 무의미해지기 때문이다.
    # 최종 컷은 rerank_service.rerank() 의 top_n 슬라이스가 담당한다(응답 형식 불변).
    n_candidates = (
        rerank_service.candidates_count()
        if (rerank_service.is_rerank_enabled() or bm25_service.is_enabled())
        else top_n
    )

    # 1) 텍스트 컬렉션 검색(항상 실행)
    query_embedding = await embed_service.get_embedding(query)

    # 1-1) 개인화: 질문 임베딩에 취향 벡터 혼합(회원 + 활성화 시에만)
    if member_id is not None and personalization_service.is_enabled():
        try:
            pref = await personalization_service.get_preference_vector(member_id)
            if pref:
                query_embedding = personalization_service.blend_vectors(query_embedding, pref)
                logger.info("개인화 취향 벡터 혼합 적용: member_id=%s (질문=%r)", member_id, query)
        except Exception:
            logger.exception("개인화 혼합 실패 → 순수 질문 벡터 사용: member_id=%s", member_id)

    text_candidates = await chroma_service.search_similar(query_embedding, n_results=n_candidates)

    # 1-2) [대규모 청크 처리] id 를 product_id 기준으로 정규화한 뒤, CLIP/BM25 플래그와
    # 무관하게 항상 자체 중복 제거를 먼저 수행한다.
    # [배경] 긴 설명 상품은 청크 문서(id=f"{product_id}_chunk_{n}")로 나뉘어 저장될 수
    # 있는데, 아래 _merge_dedup 호출은 원래 CLIP_SERVING_ENABLED=true 일 때만 실행된다.
    # 그 플래그가 꺼진(기본값) 상태에서는 청크 정규화만 해봤자 병합이 안 일어나
    # 같은 상품의 여러 청크가 서로 다른 hit 로 top_n 슬롯을 나눠 차지해 검색 다양성을
    # 해칠 수 있다. 그래서 여기서 빈 이미지 후보([])와 무조건 병합해 자체 dedup만
    # 강제한다(_merge_dedup 은 "distance 최솟값 채택" 규칙이라 청크 dedup 에도 그대로
    # 맞는다). 임계값 이하 상품(현재 카탈로그 전량)은 id 가 원래도 str(product_id)라
    # 이 정규화/병합이 값을 바꾸지 않는다(회귀 없음).
    for hit in text_candidates:
        pid = hit.get("metadata", {}).get("product_id")
        if pid is not None:
            hit["id"] = str(pid)
    text_candidates = _merge_dedup(text_candidates, [])

    # 2) 이미지 컬렉션 검색(플래그 ON 일 때만, 실패 시 텍스트로 폴백)
    if clip_service.is_serving_enabled():
        try:
            clip_embedding = await asyncio.to_thread(clip_service.encode_text, query)
            image_candidates = await chroma_service.search_similar_image(
                clip_embedding, n_results=n_candidates,
            )
            candidates = _merge_dedup(text_candidates, image_candidates)
            logger.info(
                "SEMANTIC 멀티모달 후보: 텍스트 %d + 이미지 %d → 병합 %d건 (질문=%r)",
                len(text_candidates), len(image_candidates), len(candidates), query,
            )
        except Exception:
            logger.exception("CLIP 이미지 검색 실패 → 텍스트 결과로 폴백")
            candidates = text_candidates
    else:
        candidates = text_candidates

    # 2-2) [하이브리드] BM25(Sparse) 후보 병합(플래그 ON 일 때만, 실패 시 무시)
    #      벡터(Dense) 후보가 놓치는 고유명사/모델명 정확 매칭을 BM25 로 보완한다.
    #      _merge_dedup 을 그대로 재사용(반환 스키마가 동일하게 설계됨).
    if bm25_service.is_enabled():
        try:
            bm25_candidates = await asyncio.to_thread(
                bm25_service.search, query, n_candidates,
            )
            if bm25_candidates:
                before = len(candidates)
                # [RRF 융합] 점수 스케일이 다른 Dense/Sparse 를 '순위'만으로 공정 결합.
                # (CLIP 병합용 _merge_dedup 은 그대로 두고, 텍스트 하이브리드 전용 함수 사용)
                candidates = _merge_text_rrf(candidates, bm25_candidates)
                logger.info(
                    "SEMANTIC RRF 융합: Dense %d + BM25 %d → 병합 %d건 (k=%d, 질문=%r)",
                    before, len(bm25_candidates), len(candidates), _rrf_k(), query,
                )
        except Exception:
            logger.exception("BM25/RRF 융합 실패 → Dense 후보로 폴백")

    # [버그 수정] STOCK=0(품절) 상품은 스토어프론트(상품 목록)에는 안 보이는데
    # 검색 인덱스엔 그대로 있어서, 챗봇이 화면에 없는 품절 상품을 안내하는
    # 문제가 있었다. fetch_all_products() 가 품절 상품도 전부 인덱싱하는 건
    # 의도된 설계(그 함수 독스트링 참고: "검색 시 metadata로 필터링")이고,
    # 그 필터링이 여기 빠져 있었다.
    # [판매중단 제외] 기존 재고 0 필터와 같은 자리. 과거 색인된 상품은 status
    # 메타데이터가 없을 수 있어 기본값 "ACTIVE"로 하위호환 유지.
    candidates = [
        c for c in candidates
        if c.get("metadata", {}).get("stock", 1) > 0
        and c.get("metadata", {}).get("status", "ACTIVE") == "ACTIVE"
    ]

    logger.info("SEMANTIC 최종 후보 %d건 (재랭킹=%s, CLIP=%s, 질문=%r)",
                len(candidates), rerank_service.is_rerank_enabled(),
                clip_service.is_serving_enabled(), query)

    if not candidates:
        return []

    # 3) 재랭킹(비활성/실패 시 원본 순서 폴백) → 통일 score 부여
    reranked = await rerank_service.rerank(query, candidates, top_n=top_n)
    return rerank_service.attach_scores(reranked)


def hits_to_sources(hits: list[dict]) -> list[dict]:
    """rag_hits → ChatResponse.sources 용 dict 리스트로 변환.

    metadata 에서 product_id/product_name/category/price/image_url 을 꺼내고,
    통일 score(높을수록 관련)를 함께 싣는다. score 가 없으면 distance 정규화 폴백.
    """
    from services.rerank_service import _normalize_distance

    sources = []
    for h in hits:
        meta = h.get("metadata", {}) or {}
        score = h.get("score")
        if score is None:
            score = _normalize_distance(h.get("distance"))
        pid = meta.get("product_id")
        sources.append({
            "product_id": int(pid) if pid is not None else None,
            "product_name": meta.get("product_name", "") or "",
            "category": meta.get("category", "") or "",
            "price": meta.get("price"),
            # 프론트 상품 카드 썸네일용 URL(없으면 빈 문자열)
            "image_url": meta.get("image_url", "") or "",
            "score": float(score),
        })
    return sources
