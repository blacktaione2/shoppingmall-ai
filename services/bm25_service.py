"""
services/bm25_service.py  [하이브리드 검색 — Sparse 후보]
==========================================================
BM25(Sparse, 키워드 정밀 매칭) 검색 전담 모듈.

[왜 추가하나 — 진짜 하이브리드]
- 현재 인텐트 라우팅은 STRUCTURED(키워드) vs SEMANTIC(벡터)으로 '분기'할 뿐,
  하나의 SEMANTIC 쿼리 안에서 두 기법을 '융합'하지는 않는다.
- "나이키 줌 X 같은 거" 처럼 고유명사/모델명이 섞인 쿼리는 Dense(벡터) 임베딩
  공간에서 의미 유사도에 파묻혀 정밀 매칭이 깨지기 쉽다. BM25 는 이런 정확 토큰
  매칭에 강하므로, 벡터 후보에 BM25 후보를 '병합'해 recall 을 보완한다.

[설계 — 기존 패턴 재사용]
- clip_service / rerank_service 와 똑같이 ENV 플래그로 on/off 한다.
    · BM25_ENABLED=false(기본) → 이 모듈은 어떤 경로에서도 호출되지 않아 동작/메모리 0.
- search() 반환 스키마를 chroma_service.search_similar 과 '동일'하게 맞춘다:
    [{id, document, metadata, distance}, ...]  (distance 오름차순 = 관련 높은 순)
  → rag_pipeline._merge_dedup() 을 한 글자도 바꾸지 않고 그대로 재사용한다.

[score → distance 변환 규칙]
- BM25 는 '점수 높을수록 관련'이지만 ChromaDB 는 'distance 낮을수록 관련'이라 방향이 반대다.
- 후보 내 최대 점수로 정규화한 뒤 distance = 1 - (score / max_score) 로 변환해
  기존 distance 기반 정렬/병합과 호환시킨다(0 = 최상위, 1 = 최하위).
  · 모든 점수가 0(매칭 토큰 없음)이면 그 쿼리에 대해 BM25 후보를 만들지 않는다(빈 리스트).

[한국어 토크나이징]
- rank_bm25 는 토큰 분리를 직접 하지 않는다. 한국어는 형태소 분석기 없이는
  "나이키운동화"가 1토큰이 되어 부분 매칭이 안 된다. konlpy 같은 무거운 의존성을
  피하기 위해, 문자 단위 n-gram(기본 2-gram) + 공백 토큰을 함께 사용한다.
  · 가볍고(추가 의존성 0) "나이키"가 "나이키 운동화"와 부분 매칭된다.

[인덱스 수명]
- 인메모리 인덱스다. 상품 22개 규모라 전체 구축이 수십 ms 내 끝나므로 디스크 영속화는
  하지 않는다. 서버 재시작 시 main.py lifespan 에서 1회 build_index() 한다.
- 상품 CRUD(관리자) 시 upsert_one/delete_one 으로 증분 갱신한다(실패해도 메인 경로 불방해).
"""
import logging
import os
import re
import threading

logger = logging.getLogger(__name__)

# ── 환경 설정 ────────────────────────────────────────────────────────────
BM25_ENABLED = os.getenv("BM25_ENABLED", "false").lower() == "true"
# 한국어 부분 매칭용 문자 n-gram 크기(기본 2). 2면 "나이키"→"나이","이키".
_NGRAM = int(os.getenv("BM25_NGRAM", "2") or "2")

# ── 모듈 싱글톤(지연 초기화) ─────────────────────────────────────────────
_bm25 = None                     # BM25Okapi 인스턴스
_corpus_rows: list[dict] = []    # 색인된 상품 메타(검색 결과 복원용)
_corpus_tokens: list[list[str]] = []  # 행별 토큰(증분 갱신 시 재구축에 사용)
_lock = threading.Lock()         # 증분 갱신/검색 동시성 보호(인메모리 객체 교체)


def is_enabled() -> bool:
    """하이브리드 BM25 후보를 사용할지 여부(.env 플래그)."""
    return BM25_ENABLED


def _tokenize(text: str) -> list[str]:
    """공백 토큰 + 문자 n-gram 혼합 토크나이저(형태소 분석기 불필요).

    - 영문/숫자는 공백 단위로도 의미가 있으므로 공백 토큰을 함께 넣는다.
    - 한글/연속 문자열은 n-gram 으로 잘라 부분 매칭을 가능하게 한다.
    """
    if not text:
        return []
    low = text.lower()
    tokens: list[str] = []

    # 1) 공백/기호 기준 단어 토큰 (영문 모델명, 숫자 등 정확 매칭에 유리)
    words = re.findall(r"[0-9a-z가-힣]+", low)
    tokens.extend(words)

    # 2) 각 단어를 문자 n-gram 으로도 분해(한국어 부분 매칭)
    n = _NGRAM if _NGRAM >= 1 else 2
    for w in words:
        if len(w) <= n:
            continue  # 단어 자체가 이미 토큰으로 들어가 있음
        for i in range(len(w) - n + 1):
            tokens.append(w[i:i + n])
    return tokens


def _doc_text_of(row: dict) -> str:
    """색인 대상 텍스트: 상품명 + 카테고리 + 설명(가중치는 단순 연결로 표현)."""
    name = (row.get("product_name") or "").strip()
    category = (row.get("category") or "").strip()
    desc = (row.get("description") or "").strip()
    # 상품명을 한 번 더 넣어 키워드 가중(BM25 TF 증가 효과)
    return " ".join(p for p in (name, name, category, desc) if p)


def _to_price(value) -> float:
    """가격을 float 로 정규화(Oracle NUMBER/Decimal → float, NULL → 0.0).

    ChromaDB 색인 규칙(index_products._to_float: Decimal→float, None→0.0)과
    동일하게 맞춰, 같은 상품이 Dense/BM25 어느 경로로 나와도 표기가 일치하게 한다
    (Decimal 이 남으면 build_product_context 의 isinstance 검사에 걸려
    '(가격 정보 없음)'으로 새고, None 처리도 경로별로 갈린다).
    """
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _row_to_hit(row: dict, distance: float) -> dict:
    """상품 행 → chroma_service.search_similar 과 동일한 hit 스키마로 변환."""
    pid = row.get("product_id")
    return {
        "id": str(pid) if pid is not None else "",
        "document": _doc_text_of(row),
        "metadata": {
            "product_id": int(pid) if pid is not None else None,
            "product_name": row.get("product_name") or "",
            "category": row.get("category") or "",
            "price": _to_price(row.get("price")),
            "description": row.get("description") or "",
            "stock": row.get("stock") if row.get("stock") is not None else 0,
            "image_url": row.get("image_url") or "",
        },
        "distance": distance,
    }


def build_index(products: list[dict]) -> int:
    """전체 상품으로 BM25 인덱스를 (재)구축한다. 반환: 색인된 문서 수.

    비활성(BM25_ENABLED=false)이면 아무것도 하지 않고 0 을 반환한다.
    빈 코퍼스면 인덱스를 비워 검색이 항상 빈 리스트를 반환하게 한다(예외 없음).

    [동시성] 이 함수 자체가 _lock 을 잡는다. upsert_one/delete_one 은 '스냅샷 계산 +
    재구축'을 하나의 원자적 구간으로 묶기 위해 자신들의 락 안에서 이 함수를
    호출하지 않고, _build_index_locked() 를 직접 쓴다(재진입 불가 Lock 이라 이중
    획득 시 데드락이 나기 때문). 외부(main.py lifespan 등)에서 단독 호출할 때는
    이 함수를 그대로 쓰면 된다.
    """
    if not BM25_ENABLED:
        return 0
    with _lock:
        return _build_index_locked(products)


def _build_index_locked(products: list[dict]) -> int:
    """_lock 을 이미 잡고 있는 호출자를 위한 내부 재구축 로직(락 재획득 없음)."""
    global _bm25, _corpus_rows, _corpus_tokens

    # 지연 import: 비활성 환경에선 rank_bm25 가 없어도 된다.
    from rank_bm25 import BM25Okapi

    rows = [r for r in (products or []) if r.get("product_id") is not None]
    tokens = [_tokenize(_doc_text_of(r)) for r in rows]

    _corpus_rows = rows
    _corpus_tokens = tokens
    # rank_bm25 는 빈 코퍼스에서 ZeroDivision 을 낼 수 있어 가드한다.
    _bm25 = BM25Okapi(tokens) if tokens else None
    logger.info("BM25 인덱스 구축 완료: %d건 (ngram=%d)", len(rows), _NGRAM)
    return len(rows)


def search(query: str, n_results: int = 10) -> list[dict]:
    """BM25 로 상위 n_results 후보를 검색한다.

    반환: chroma_service.search_similar 과 동일 스키마
          [{id, document, metadata, distance}, ...]  (distance 오름차순)
    비활성/미구축/빈 결과면 빈 리스트.
    """
    if not BM25_ENABLED:
        return []
    with _lock:
        bm25 = _bm25
        rows = _corpus_rows
    if bm25 is None or not rows:
        return []

    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scores = bm25.get_scores(q_tokens)  # 행별 점수(numpy array 형태)
    max_score = float(max(scores)) if len(scores) else 0.0
    # 매칭 토큰이 하나도 없으면(전부 0) 이 쿼리에 BM25 후보 없음 → 빈 리스트
    if max_score <= 0.0:
        return []

    # 점수 내림차순 상위 n 선정
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits: list[dict] = []
    for idx in ranked[:n_results]:
        s = float(scores[idx])
        if s <= 0.0:
            break  # 0점부터는 매칭 없음 → 후보에서 제외
        distance = 1.0 - (s / max_score)   # 0(최상위)~1(최하위)로 변환
        hits.append(_row_to_hit(rows[idx], distance))
    return hits


def upsert_one(product: dict) -> None:
    """상품 1건 등록/수정 시 인덱스를 증분 갱신한다(전체 재구축으로 단순화).

    상품 22개 규모라 단건 변경 시 전체 재토큰화 비용이 무시할 수준이다.
    스냅샷 계산과 재구축은 _build_index_locked() 로 하나의 락 구간에 묶어
    원자적으로 처리한다(동시 CRUD 의 lost update 방지).
    """
    if not BM25_ENABLED or product.get("product_id") is None:
        return
    pid = int(product["product_id"])
    with _lock:
        rows = [r for r in _corpus_rows if int(r.get("product_id")) != pid]
        rows.append(product)
        _build_index_locked(rows)


def delete_one(product_id) -> None:
    """상품 1건 삭제 시 인덱스에서 제거(전체 재구축). 존재하지 않아도 안전.
    upsert_one 과 동일하게 스냅샷+재구축을 한 락 구간으로 묶는다(원자성).
    """
    if not BM25_ENABLED or product_id is None:
        return
    pid = int(product_id)
    with _lock:
        rows = [r for r in _corpus_rows if int(r.get("product_id")) != pid]
        _build_index_locked(rows)


def indexed_count() -> int:
    """현재 색인된 문서 수(헬스/디버그용)."""
    with _lock:
        return len(_corpus_rows)
