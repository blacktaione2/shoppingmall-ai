"""
services/personalization_service.py  [개인화 추천 레벨 2]
구매 이력 기반 '취향 벡터(user preference vector)' 계산 및 질문 벡터 혼합.

[핵심 아이디어]
- 회원의 구매 상품 임베딩을 '수량 × 최신성'으로 가중합산해 취향 벡터를 만든다.
  취향 벡터 = normalize( Σ embedding_i × (quantity_i × recency_i) )
    · recency_i = 0.5 ^ (경과일 / 반감기)   ← A안: 지수 감쇠(최근 구매일수록 큰 영향)
- SEMANTIC 검색 시 질문 벡터와 혼합한다.
  검색 벡터 = normalize( α × 질문벡터 + β × 취향벡터 )

[적용 경계]
- 라우터 경로(semantic_node)에서 로그인 회원에게만 적용한다.
  · 게스트(member_id=None)·이력 0건·임베딩 0건 → None 반환 → 순수 질문 검색 폴백.
  · Agent 경로(semantic_search tool)는 비교 공정성을 위해 적용하지 않는다.
- text-embedding-3-small(1536차원) 공간에서만 동작한다.
  CLIP(512차원) 이미지 검색에는 적용하지 않는다(차원/공간이 다름).

[안전장치]
- PERSONALIZATION_ENABLED=false(기본) → 호출되더라도 즉시 비활성처럼 동작(검색 영향 0).
- Oracle/ChromaDB 장애·이력 없음 → None 폴백. 개인화 실패가 검색 자체를 막지 않는다.
- member_id 별 메모리 캐시(TTL): 매 검색마다 구매이력 조인 + 임베딩 N건 조회 비용을 줄인다.
  (TTL 동안은 새 구매가 즉시 반영되지 않는 stale 을 허용 — v1 트레이드오프)

[의존성]
- numpy 미사용(의존성 최소화). 1536차원 × 최대 MAX_HISTORY 건이라 순수 Python 으로 충분.
"""
import asyncio
import logging
import math
import os
import time

from dotenv import load_dotenv

from database import oracle_db
from services import chroma_service

load_dotenv()

logger = logging.getLogger(__name__)


# ── 환경 설정 ────────────────────────────────────────────────────────────
def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


PERSONALIZATION_ENABLED = os.getenv("PERSONALIZATION_ENABLED", "false").lower() == "true"
ALPHA = _get_float("PERSONALIZATION_ALPHA", 0.7)                       # 질문 가중치
BETA = _get_float("PERSONALIZATION_BETA", 0.3)                         # 취향 가중치
HALFLIFE_DAYS = _get_float("PERSONALIZATION_RECENCY_HALFLIFE_DAYS", 90.0)  # 최신성 반감기(일)
CACHE_TTL = _get_float("PERSONALIZATION_CACHE_TTL", 300.0)             # 취향 벡터 캐시 TTL(초)
MAX_HISTORY = _get_int("PERSONALIZATION_MAX_HISTORY", 50)              # 가중합산 입력 상한
CACHE_MAX = _get_int("PERSONALIZATION_CACHE_MAX", 512)                 # 취향 벡터 캐시 항목 상한

# member_id → (취향벡터 or None, 저장시각). None 도 캐싱해 '이력 없음' 재조회를 막는다.
_cache: dict[int, tuple[list[float] | None, float]] = {}

# member_id 별 asyncio.Lock. 같은 회원에 대한 동시 캐시 미스가 각자 비싼
# DB+임베딩 파이프라인을 중복 실행(cache stampede)하는 것을 막는다.
# (이 캐시는 async 코루틴 컨텍스트에서만 갱신되므로 threading.Lock 이 아니라
#  asyncio.Lock 이 맞다 — bm25_service 는 to_thread 안 CPU 작업이라 threading.Lock.)
_locks: dict[int, asyncio.Lock] = {}


def _get_lock(member_id: int) -> asyncio.Lock:
    """member_id 전용 락을 가져온다(없으면 생성). 단일 이벤트 루프 전제라 안전."""
    lock = _locks.get(member_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[member_id] = lock
    return lock


def is_enabled() -> bool:
    """개인화 활성화 여부(.env 플래그)."""
    return PERSONALIZATION_ENABLED


# ── 벡터 유틸 (순수 Python) ──────────────────────────────────────────────
def _normalize(vec: list[float]) -> list[float]:
    """L2 정규화. 0 벡터(노름 0)는 그대로 반환(0 분할 방지)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def blend_vectors(query_vec: list[float], pref_vec: list[float],
                  alpha: float = ALPHA, beta: float = BETA) -> list[float]:
    """질문 벡터와 취향 벡터를 가중 혼합 후 재정규화.

    검색 벡터 = normalize(α·query + β·pref)
    [중요] 재정규화를 빼면 벡터 크기가 변해 cosine 거리가 왜곡되므로 반드시 정규화한다.
    길이가 다르면(이론상 없음) 짧은 쪽 기준으로 안전하게 처리.
    """
    n = min(len(query_vec), len(pref_vec))
    blended = [alpha * query_vec[i] + beta * pref_vec[i] for i in range(n)]
    return _normalize(blended)


def _recency_weight(order_date, now_ts: float) -> float:
    """주문일 → 최신성 가중치 0.5^(경과일/반감기). 지수 감쇠(A안).

    order_date 가 None/파싱불가면 1.0(중립)로 처리.
    tz-aware/naive 혼용으로 인한 빼기 오류를 피하려 tzinfo 를 제거해 통일한다.
    """
    if order_date is None:
        return 1.0
    try:
        # datetime → epoch 초. tz 정보가 있으면 제거(naive 통일)
        od = order_date.replace(tzinfo=None) if hasattr(order_date, "replace") else order_date
        order_ts = od.timestamp()
    except (AttributeError, ValueError, OSError):
        return 1.0
    elapsed_days = max(0.0, (now_ts - order_ts) / 86400.0)
    if HALFLIFE_DAYS <= 0:
        return 1.0
    return 0.5 ** (elapsed_days / HALFLIFE_DAYS)


# ── 취향 벡터 계산 ───────────────────────────────────────────────────────
def _cache_get(member_id: int):
    """캐시 조회(TTL 유효 시에만). (hit?, value) 튜플 반환."""
    entry = _cache.get(member_id)
    if entry is None:
        return False, None
    vec, ts = entry
    if (time.time() - ts) > CACHE_TTL:
        _cache.pop(member_id, None)
        return False, None
    return True, vec


def _drop_member(member_id: int) -> None:
    """캐시 항목 제거 + '잡혀있지 않은' 락만 함께 정리.

    현재 잡혀있는 락(in-flight 코루틴이 계산 중)은 남긴다. 이걸 pop 하면 뒤이어
    _get_lock 이 새 락을 만들어, 같은 회원의 동시 요청이 서로 다른 락으로 임계구역에
    동시 진입(스탬피드 방어 붕괴)할 수 있기 때문이다.
    """
    _cache.pop(member_id, None)
    lock = _locks.get(member_id)
    if lock is not None and not lock.locked():
        _locks.pop(member_id, None)


def _evict_if_needed() -> None:
    """캐시 크기 상한(CACHE_MAX) 유지. 다시 안 오는 회원 항목이 영구 잔존하는 것을 막는다.

    1) 만료된 항목을 먼저 제거하고, 2) 그래도 상한 이상이면 저장시각이 오래된 순으로
    제거해 새 항목 1개를 넣을 자리를 확보한다(간단 시각 기반 정리).
    """
    if len(_cache) < CACHE_MAX:
        return
    now = time.time()
    for mid in [m for m, (_, ts) in list(_cache.items()) if (now - ts) > CACHE_TTL]:
        _drop_member(mid)
    if len(_cache) >= CACHE_MAX:
        ordered = sorted(_cache.items(), key=lambda kv: kv[1][1])   # 저장시각 오름차순
        overflow = len(_cache) - CACHE_MAX + 1                      # 새 항목 자리 1개 확보
        for mid, _ in ordered[:overflow]:
            _drop_member(mid)


def _cache_put(member_id: int, value) -> None:
    """캐시 저장(상한 관리 포함). value 는 취향벡터(list) 또는 None(이력 없음)."""
    _evict_if_needed()
    _cache[member_id] = (value, time.time())


async def get_preference_vector(member_id: int) -> list[float] | None:
    """회원의 구매 이력으로 취향 벡터(정규화)를 계산. 이력/임베딩 없으면 None.

    흐름:
      1) 캐시 확인(TTL)
      2) 구매 이력 조회(ORDER_ITEM ⋈ ORDERS) — 최신순 MAX_HISTORY 건
      3) 같은 상품 반복 구매는 가중치(수량×최신성) 누적
      4) 상품 임베딩 일괄 조회(ChromaDB products) — 없는 상품은 스킵
      5) 가중합산 → 정규화 → 캐시 저장 → 반환

    [폴백] 어떤 단계에서 장애가 나도 예외를 호출 측으로 던지지 않고 None 을 반환한다.
           (개인화 실패가 검색을 막지 않도록)
    """
    if member_id is None:
        return None

    # 빠른 경로: 락 없이 캐시 확인(히트면 즉시 반환).
    hit, cached = _cache_get(member_id)
    if hit:
        return cached

    # 캐시 미스: 같은 회원의 동시 요청이 비싼 계산을 중복하지 않도록 락 안에서 처리.
    async with _get_lock(member_id):
        # 더블체크: 락을 기다리는 동안 다른 요청이 이미 채웠을 수 있다.
        hit, cached = _cache_get(member_id)
        if hit:
            return cached

        try:
            history = await asyncio.to_thread(
                oracle_db.fetch_purchase_history, member_id, MAX_HISTORY,
            )
        except Exception:
            logger.exception("구매 이력 조회 실패 → 개인화 폴백(None): member_id=%s", member_id)
            # 실패도 None 으로 캐싱한다 — 락 정리(_evict_if_needed/clear_cache)가
            # _cache 순회 기반이라, 캐싱하지 않으면 이 경로의 _locks 항목이 영구
            # 잔존한다(장애 시 재조회 폭주 방지 효과도 겸함).
            _cache_put(member_id, None)
            return None

        if not history:
            _cache_put(member_id, None)   # 이력 없음도 캐싱(재조회 방지)
            return None

        # 3) 상품별 가중치 누적 (product_id → weight)
        now_ts = time.time()
        weight_by_pid: dict[str, float] = {}
        for item in history:
            pid = item.get("product_id")
            if pid is None:
                continue
            qty = item.get("quantity") or 1
            w = qty * _recency_weight(item.get("order_date"), now_ts)
            key = str(pid)
            weight_by_pid[key] = weight_by_pid.get(key, 0.0) + w

        if not weight_by_pid:
            _cache_put(member_id, None)
            return None

        # 4) 상품 임베딩 일괄 조회 (없는 상품은 dict 에서 빠짐)
        try:
            emb_map = await chroma_service.get_embeddings_by_ids(list(weight_by_pid.keys()))
        except Exception:
            logger.exception("취향 상품 임베딩 조회 실패 → 개인화 폴백(None): member_id=%s", member_id)
            # 실패도 캐싱(_locks 잔존 방지) — 위 구매 이력 예외 경로와 동일한 이유.
            _cache_put(member_id, None)
            return None

        if not emb_map:
            _cache_put(member_id, None)
            return None

        # 5) 가중합산 (임베딩이 있는 상품만)
        acc: list[float] | None = None
        for key, weight in weight_by_pid.items():
            emb = emb_map.get(key)
            if emb is None:
                continue
            if acc is None:
                acc = [weight * v for v in emb]
            else:
                # 차원 불일치 방어(이론상 동일하지만 안전하게 짧은 쪽 기준)
                m = min(len(acc), len(emb))
                for i in range(m):
                    acc[i] += weight * emb[i]

        if acc is None:
            _cache_put(member_id, None)
            return None

        pref = _normalize(acc)
        _cache_put(member_id, pref)
        logger.info("취향 벡터 생성: member_id=%s (상품 %d종, 이력 %d건)",
                    member_id, len(emb_map), len(history))
        return pref


def clear_cache(member_id: int | None = None) -> None:
    """캐시 무효화. member_id 지정 시 해당 회원만, None 이면 전체.

    구매 직후 즉시 반영이 필요하면 Spring Boot 측에서 별도 엔드포인트로 호출하거나
    추후 admin 훅에 연결할 수 있다(v1 에서는 TTL 만료에 의존).
    """
    if member_id is None:
        _cache.clear()
        # 잡혀있지 않은 락만 정리(in-flight 락은 남겨 스탬피드 방어 유지).
        for mid in list(_locks.keys()):
            lock = _locks.get(mid)
            if lock is not None and not lock.locked():
                _locks.pop(mid, None)
    else:
        _cache.pop(member_id, None)
        lock = _locks.get(member_id)
        if lock is not None and not lock.locked():
            _locks.pop(member_id, None)
