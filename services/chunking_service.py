"""
services/chunking_service.py  [대규모 청크 처리 — Context-Rich Chunking]
==========================================================================
상품 1건을 ChromaDB에 넣을 문서 1~N개로 변환하는 단일 출처(single source) 모듈.

[설계]
- description 이 짧으면(기본 500자 이하) 지금까지와 동일하게 문서 1개만 반환한다.
  → 현재 카탈로그(상품 설명 200~250자)는 전부 이 경로라 회귀가 없다.
- 임계값을 넘는 긴 설명만 RecursiveCharacterTextSplitter 로 분할하고,
  청크마다 "[상품명 | 카테고리]" 프리픽스를 앞에 강제 주입해 임베딩한다
  (문맥 없이 잘린 청크만 벡터화되는 것을 방지 — Context-Rich Chunking).
- 청크 문서의 id 는 f"{product_id}_chunk_{n}" 으로 원본 상품 id 와 구분한다.
  metadata 는 원본 상품 것을 그대로 복사(product_id/stock/status/price/image_url 등)해
  하류(재고 필터·sources 변환)가 청크 여부를 몰라도 그대로 동작하게 한다.
  · chunk_index/chunk_count: 디버깅/재구성용 부가 정보.
  · chunk_text_raw: 프리픽스가 없는 순수 청크 원문. LLM 답변 생성 프롬프트
    (rag_service.build_product_context)가 상품명/카테고리 중복 노출 없이
    이 필드를 우선 사용한다.

[호출 지점 — 배치/단건 동일 함수 사용]
- scripts/index_products.py(전체 재색인)와 routers/admin.py 의 reindex_product
  (단건 재색인) 둘 다 이 함수 하나만 호출한다. 로직이 두 곳에 따로 있으면
  한쪽만 고쳐서 동작이 갈라지는 문제(청킹이 배치에만 적용되는 등)가 생기므로
  단일 출처로 강제한다.

[삭제/재색인 시 orphan 청크는 이 모듈의 책임이 아님]
- 이 함수는 '어떤 문서를 새로 넣을지'만 결정한다. 기존 청크(개수가 바뀌었을 수 있음)를
  지우는 책임은 호출 측(index_products.py/admin.py)이 upsert 전에
  chroma_service.delete_product(product_id) 를 먼저 호출하는 것으로 진다
  (delete-then-upsert 패턴 — chroma_service.delete_product 자체도 이제
  id 하나가 아니라 product_id 메타데이터 기준으로 삭제하도록 바뀌었다).
"""
import os
from decimal import Decimal

from langchain_text_splitters import RecursiveCharacterTextSplitter


# [순환 import 방지] 이 두 헬퍼는 원래 scripts/index_products.py 에 있었다.
# index_products.py 가 이 모듈의 build_chunk_documents() 를 쓰게 되면서
# "index_products → chunking_service → index_products" 순환이 생기므로,
# 정의를 이 모듈로 옮기고 index_products.py 가 여기서 다시 import 하도록 바꿨다.
# scripts.index_products._embed_text_for 처럼 기존 경로로 import 하던 다른 모듈
# (scripts/ci_index_fixtures.py, scripts/index_products_image.py)은 index_products.py가
# 이 이름들을 그대로 재노출(re-export)하므로 영향 없다.

def _to_float(value) -> float:
    """Decimal/None/숫자 → float. NULL 은 0.0 으로 처리."""
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _embed_text_for(row: dict) -> str:
    """
    임베딩 대상 텍스트 결정.
    - 우선: description (LOB 변환 완료 문자열)
    - 폴백: product_name + category (description 이 NULL/공백인 경우)
    - [Vision 태깅] image_caption 이 있으면 뒤에 이어붙인다 — 색상/소재/스타일
      키워드가 텍스트 검색(BM25)·임베딩(Dense) 둘 다에 반영되도록.
    """
    desc = row.get("description")
    if desc and str(desc).strip():
        base = str(desc).strip()
    else:
        name = (row.get("product_name") or "").strip()
        category = (row.get("category") or "").strip()
        fallback = f"{name} {category}".strip()
        base = fallback or name or "상품"

    caption = (row.get("image_caption") or "").strip()
    return f"{base} {caption}".strip() if caption else base


def _chunk_threshold_chars() -> int:
    """청킹을 시작할 글자 수 임계값(.env CHUNK_THRESHOLD_CHARS, 기본 500).

    파싱 실패/0 이하면 500 으로 폴백(청킹이 조용히 꺼지는 것 방지).
    """
    try:
        val = int(os.getenv("CHUNK_THRESHOLD_CHARS", "500"))
    except (TypeError, ValueError):
        return 500
    return val if val > 0 else 500


def _chunk_overlap_chars() -> int:
    """인접 청크 간 중첩 글자 수(.env CHUNK_OVERLAP_CHARS, 기본 50)."""
    try:
        val = int(os.getenv("CHUNK_OVERLAP_CHARS", "50"))
    except (TypeError, ValueError):
        return 50
    return val if val >= 0 else 50


def _base_metadata(row: dict) -> dict:
    """index_products.py/admin.py 와 동일 규칙의 상품 메타데이터(청크 부가 필드 제외)."""
    pid = row.get("product_id")
    return {
        "product_id":   int(pid),
        "product_name": row.get("product_name") or "",
        "category":     row.get("category") or "",
        "price":        _to_float(row.get("price")),
        "description":  str(row.get("description")) if row.get("description") else "",
        "stock":        int(row.get("stock")) if row.get("stock") is not None else 0,
        "status":       row.get("status") or "ACTIVE",
        "image_url":    row.get("image_url") or "",
    }


def build_chunk_documents(row: dict) -> list[dict]:
    """상품 1건(row) → ChromaDB 문서 1~N개.

    Args:
        row: oracle_db.fetch_all_products()/fetch_product_by_id() 반환 형태의 dict.
    Returns:
        [{"id": str, "document": str, "metadata": dict}, ...]
        - 임계값 이하: 길이 1(id=str(product_id)), document 는 기존 _embed_text_for() 결과
          그대로(프리픽스 없음) — 기존 인덱싱과 100% 동일한 데이터.
        - 임계값 초과: 길이 N(id=f"{product_id}_chunk_{n}"), 각 document 는
          "[상품명 | 카테고리] 청크원문" 형태.
    """
    pid = row.get("product_id")
    text = _embed_text_for(row)
    base_meta = _base_metadata(row)

    threshold = _chunk_threshold_chars()
    if len(text) <= threshold:
        # 짧은 설명(현재 카탈로그 전량 포함) — 청킹 없이 기존과 동일하게 문서 1개.
        return [{
            "id": str(pid),
            "document": text,
            "metadata": base_meta,
        }]

    name = base_meta["product_name"] or "(이름 없음)"
    category = base_meta["category"] or "(카테고리 없음)"
    prefix = f"[{name} | {category}] "

    # overlap이 threshold 이상이면 RecursiveCharacterTextSplitter가
    # ValueError("chunk_overlap > chunk_size")로 죽는다. .env에서 THRESHOLD만
    # 낮추고 OVERLAP은 그대로 둔 조합에서 실제로 재현됨 — threshold 미만으로 clamp.
    overlap = min(_chunk_overlap_chars(), max(threshold - 1, 0))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=threshold,
        chunk_overlap=_chunk_overlap_chars(),
    )
    raw_chunks = splitter.split_text(text)
    chunk_count = len(raw_chunks)

    documents = []
    for idx, raw_chunk in enumerate(raw_chunks):
        meta = dict(base_meta)
        meta["chunk_index"] = idx
        meta["chunk_count"] = chunk_count
        meta["chunk_text_raw"] = raw_chunk   # 프리픽스 없는 순수 원문(LLM 프롬프트용)
        documents.append({
            "id": f"{pid}_chunk_{idx}",
            "document": f"{prefix}{raw_chunk}",
            "metadata": meta,
        })
    return documents
