"""
services/rerank_service.py
RAG 재랭킹 추상화 서비스.

[목적]
- ChromaDB 가 가져온 후보(top-k)를 재랭커로 정밀 재정렬해 최종 top-n 을 뽑는다.
  · 기본 provider = cohere (관리형 API → 인스턴스 2 의 1GB RAM 제약 회피, OOM 없음).
  · 추후 로컬 Cross-Encoder 로 교체 가능하도록 provider 추상화.

[안전장치 — 가장 중요]
- RERANK_ENABLED=false(기본) → 재랭킹 자체를 건너뛰고 원본 hits 를 그대로 반환.
  (메모리/비용 0, 기존 동작과 동일)
- 재랭킹 API 장애/키 부재/패키지 미설치 → '원본 순서 유지'로 폴백.
  RAG 파이프라인은 절대 멈추지 않는다.
- lazy import: cohere 패키지는 provider=cohere 이고 활성화됐을 때만 import.

[score 통일 규칙] (결정 ①)
- 항상 "높을수록 관련"으로 통일한다.
  · 재랭킹 적용 시  : Cohere relevance_score(0~1) 를 score 로 사용.
  · 재랭킹 미적용 시: ChromaDB distance 를 1/(1+distance) 로 정규화해 score 로 사용.
- 이 정규화는 attach_scores() 가 담당하며, sources 변환은 항상 이 score 를 쓴다.
"""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def is_rerank_enabled() -> bool:
    """재랭킹 활성화 여부 (환경변수, 기본 off)."""
    return os.getenv("RERANK_ENABLED", "false").lower() == "true"


def _get_provider() -> str:
    return os.getenv("RERANK_PROVIDER", "cohere").strip().lower()


def _candidates_count(default: int = 10) -> int:
    try:
        return int(os.getenv("RERANK_CANDIDATES", str(default)))
    except ValueError:
        return default


def _top_n(default: int = 4) -> int:
    try:
        return int(os.getenv("RERANK_TOP_N", str(default)))
    except ValueError:
        return default


def _normalize_distance(distance) -> float:
    """ChromaDB distance(낮을수록 가까움) → score(높을수록 관련)로 정규화.

    1/(1+distance): distance=0 → 1.0, distance 커질수록 0 에 수렴.
    """
    try:
        d = float(distance)
        if d < 0:
            d = 0.0
        return 1.0 / (1.0 + d)
    except (TypeError, ValueError):
        return 0.0


def attach_scores(hits: list[dict]) -> list[dict]:
    """hits 에 통일된 'score'(높을수록 관련)를 부여한다.

    이미 rerank_score 가 있으면 그것을, 없으면 distance 정규화값을 score 로 쓴다.
    원본 hits 를 변형하지 않고 score 키만 추가한 새 dict 리스트를 반환.
    """
    scored = []
    for h in hits:
        new_h = dict(h)
        if "rerank_score" in new_h and new_h["rerank_score"] is not None:
            new_h["score"] = float(new_h["rerank_score"])
        else:
            new_h["score"] = _normalize_distance(new_h.get("distance"))
        scored.append(new_h)
    return scored


def _rerank_cohere(query: str, hits: list[dict], top_n: int) -> list[dict]:
    """Cohere Rerank API 로 재정렬. 실패 시 예외 → 호출자가 폴백."""
    import cohere  # lazy import

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY 가 설정되어 있지 않습니다.")

    model = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
    client = cohere.ClientV2(api_key)

    # 재랭킹 대상 문서: document(상품 설명 텍스트) 사용
    documents = [h.get("document", "") or "" for h in hits]
    resp = client.rerank(
        model=model, query=query, documents=documents, top_n=top_n,
    )

    # 결과(index, relevance_score) → 원본 hit 매핑 + rerank_score 부여
    reranked = []
    for r in resp.results:
        hit = dict(hits[r.index])
        hit["rerank_score"] = float(r.relevance_score)
        reranked.append(hit)
    return reranked


async def rerank(query: str, hits: list[dict], top_n: int | None = None) -> list[dict]:
    """후보 hits 를 재랭킹해 상위 top_n 을 반환한다.

    Args:
        query: 사용자 검색 질의(자연어).
        hits: ChromaDB 후보 [{id, document, metadata, distance}, ...].
        top_n: 최종 반환 개수. None 이면 RERANK_TOP_N(기본 4).
    Returns:
        재랭킹된 hits[:top_n]. (각 hit 에 rerank_score 포함)
        비활성/실패 시 원본 hits[:top_n] (순서 유지).

    [폴백 정책]
        - RERANK_ENABLED=false      → 원본 hits[:top_n]
        - hits 가 비었으면           → []
        - API/키/패키지 오류         → 원본 hits[:top_n] (경고 로그)
    """
    import asyncio

    n = top_n if top_n is not None else _top_n()

    if not hits:
        return []
    if not is_rerank_enabled():
        return hits[:n]

    provider = _get_provider()
    try:
        if provider == "cohere":
            # cohere SDK 는 동기 → 스레드풀에서 실행(이벤트루프 비블로킹)
            reranked = await asyncio.to_thread(_rerank_cohere, query, hits, n)
            return reranked
        logger.warning("알 수 없는 RERANK_PROVIDER=%r → 재랭킹 건너뜀", provider)
        return hits[:n]
    except Exception:
        logger.exception("재랭킹 실패 → 원본 순서로 폴백")
        return hits[:n]


def candidates_count() -> int:
    """ChromaDB 에서 가져올 후보 개수(재랭킹 입력). 재랭킹 off 면 top_n 과 동일하게 써도 됨."""
    return _candidates_count()
