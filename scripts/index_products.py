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
  2) 상품별 문서 구성(services.chunking_service.build_chunk_documents):
     - 임베딩 대상 텍스트(description, 없으면 product_name+category 폴백)가
       CHUNK_THRESHOLD_CHARS(.env, 기본 500자) 이하면 기존과 동일하게 문서 1개.
     - 초과하면 RecursiveCharacterTextSplitter 로 분할하고, 청크마다
       "[상품명 | 카테고리]" 프리픽스를 삽입해 문서 N개(id=f"{product_id}_chunk_{n}")로 확장.
  3) text-embedding-3-small 배치 임베딩 (문서 전체 대상 1회 호출)
  4) 상품별 delete_product() 선행 → ChromaDB upsert
     (청크 개수가 이전 인덱싱과 달라져도 옛 청크가 orphan 으로 안 남도록 delete-then-upsert)

주의:
  - price(Decimal) → float 강제 변환 (Chroma 메타데이터는 str/int/float/bool 만 허용)
"""
import asyncio
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from database import oracle_db
from services import embed_service, chroma_service
# [순환 import 방지] _to_float/_embed_text_for 는 services/chunking_service.py 가
# 정의를 소유한다(그 모듈 docstring 참고). 여기서 재노출(re-export)해
# "from scripts.index_products import _embed_text_for" 처럼 기존 경로로 이 함수들을
# import 하던 다른 모듈(ci_index_fixtures.py, index_products_image.py 등)이
# 그대로 동작하게 한다.
from services.chunking_service import _to_float, _embed_text_for, build_chunk_documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("index_products")


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

    # 2) 상품별 문서 구성 — [대규모 청크 처리] build_chunk_documents() 가 설명 길이에
    #    따라 문서 1개(짧은 설명, 현재 카탈로그 전량) 또는 N개(긴 설명 — 청크마다
    #    "[상품명 | 카테고리]" 프리픽스 삽입)를 돌려준다. routers/admin.py 의 단건
    #    재색인(reindex_product)과 동일 함수를 써서 두 경로의 청킹 동작을 통일한다.
    ids: list[str] = []
    embed_texts: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    product_ids: list[int] = []      # delete-then-upsert 대상(상품 단위, 청크 아님)

    for row in rows:
        pid = row.get("product_id")
        if pid is None:
            logger.warning("product_id 없는 행 스킵: %r", row)
            continue

        product_ids.append(pid)
        for doc in build_chunk_documents(row):
            ids.append(doc["id"])
            embed_texts.append(doc["document"])
            documents.append(doc["document"])   # 임베딩 원문을 document 로 저장 (RAG 컨텍스트에 재사용)
            metadatas.append(doc["metadata"])

    if not ids:
        logger.warning("인덱싱할 유효 상품이 없습니다.")
        return

    # 3) 배치 임베딩 (text-embedding-3-small, 1회 호출 — 청크 포함 전체 문서 대상)
    logger.info("임베딩 생성 중... (%d건 문서 / 상품 %d건, 모델=%s)",
                len(embed_texts), len(product_ids), embed_service.EMBED_MODEL)
    embeddings = await embed_service.get_embeddings(embed_texts)
    logger.info("임베딩 완료: %d벡터 x %d차원",
                len(embeddings), len(embeddings[0]) if embeddings else 0)

    # 4) [delete-then-upsert] upsert 전에 상품별 기존 문서를 전부 삭제한다.
    #    청크 개수가 이전 인덱싱 이후 바뀐 상품(예: 설명을 줄여 3개→1개)의 옛 청크가
    #    orphan 으로 남지 않도록 하기 위함 — chroma_service.delete_product() 가
    #    id 하나가 아니라 product_id 메타데이터 필터로 삭제하므로 청크 개수를 몰라도 안전하다.
    for pid in product_ids:
        await chroma_service.delete_product(pid)

    # 5) ChromaDB upsert (idempotent — 재실행해도 동일 id 는 갱신)
    total = await chroma_service.upsert_products(ids, embeddings, documents, metadatas)
    logger.info("ChromaDB upsert 완료. 컬렉션 '%s' 총 %d건 (문서 %d건, 상품 %d건)",
                chroma_service.COLLECTION_NAME, total, len(ids), len(product_ids))


if __name__ == "__main__":
    asyncio.run(main())
