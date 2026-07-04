"""
scripts/index_products.py
Oracle PRODUCT 테이블 → ChromaDB 'products' 컬렉션 최초/재인덱싱 스크립트.

실행(프로젝트 루트에서):  python -m scripts.index_products
사전조건:
  - .env 에 OPENAI_API_KEY / DB_* / CHROMA_HOST / CHROMA_PORT 설정

처리:
  1) oracle_db.fetch_all_products() 로 전체 상품 로드
     반환 형태: [{"product_id": 1, "product_name": ..., "category": ...,
                  "price": ..., "description": ..., "stock": ...}, ...]
     (컬럼명은 소문자 확정 — cursor.description lower() 변환)
  2) 임베딩 대상 텍스트 = description (NULL/공백이면 product_name + category 폴백)
  3) text-embedding-3-small 배치 임베딩 (1회 호출)
  4) ChromaDB upsert (id = product_id 문자열) → 재실행해도 중복 없이 갱신

주의:
  - price(Decimal) → float 강제 변환 (Chroma 메타데이터는 str/int/float/bool 만 허용)
"""
import asyncio
import logging
from decimal import Decimal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from database import oracle_db
from services import embed_service, chroma_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("index_products")


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
    """
    desc = row.get("description")
    if desc and str(desc).strip():
        return str(desc).strip()
    name = (row.get("product_name") or "").strip()
    category = (row.get("category") or "").strip()
    fallback = f"{name} {category}".strip()
    return fallback or name or "상품"


async def main():
    # 0) ChromaDB 연결 확인
    try:
        hb = await chroma_service.heartbeat()
        logger.info("ChromaDB 연결 OK (heartbeat=%s)", hb)
    except Exception as e:
        logger.error("ChromaDB 연결 실패 — ChromaDB 서버(chroma run) 기동/포트(CHROMA_PORT) 확인 필요: %s", e)
        raise

    # 1) Oracle 상품 로드 (동기 함수 → to_thread 로 비블로킹)
    rows: list[dict] = await asyncio.to_thread(oracle_db.fetch_all_products)
    if not rows:
        logger.warning("PRODUCT 테이블에 상품이 없습니다. 인덱싱 중단.")
        return
    logger.info("Oracle 상품 %d건 로드", len(rows))

    # 2) 임베딩 입력 / 메타데이터 / ID 구성
    ids: list[str] = []
    embed_texts: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for row in rows:
        pid = row.get("product_id")
        if pid is None:
            logger.warning("product_id 없는 행 스킵: %r", row)
            continue

        text = _embed_text_for(row)

        ids.append(str(pid))
        embed_texts.append(text)
        documents.append(text)          # 임베딩 원문을 document 로 저장 (RAG 설명에 재사용)
        metadatas.append({
            "product_id":   int(pid),
            "product_name": row.get("product_name") or "",
            "category":     row.get("category") or "",
            "price":        _to_float(row.get("price")),   # Decimal → float 변환
            "description":  str(row.get("description")) if row.get("description") else "",
            "stock":        int(row.get("stock")) if row.get("stock") is not None else 0,
            # 검색 결과 카드(프론트)에서 상품 이미지를 띄우기 위한 URL.
            # CLIP 검색을 안 켜도(텍스트 검색만) 이 필드로 이미지가 표시된다.
            "image_url":    row.get("image_url") or "",
        })

    if not ids:
        logger.warning("인덱싱할 유효 상품이 없습니다.")
        return

    # 3) 배치 임베딩 (text-embedding-3-small, 1회 호출)
    logger.info("임베딩 생성 중... (%d건, 모델=%s)", len(embed_texts), embed_service.EMBED_MODEL)
    embeddings = await embed_service.get_embeddings(embed_texts)
    logger.info("임베딩 완료: %d벡터 x %d차원",
                len(embeddings), len(embeddings[0]) if embeddings else 0)

    # 4) ChromaDB upsert (idempotent — 재실행 시 중복 없이 갱신)
    total = await chroma_service.upsert_products(ids, embeddings, documents, metadatas)
    logger.info("ChromaDB upsert 완료. 컬렉션 '%s' 총 %d건",
                chroma_service.COLLECTION_NAME, total)


if __name__ == "__main__":
    asyncio.run(main())
