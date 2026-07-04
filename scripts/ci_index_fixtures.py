"""
scripts/ci_index_fixtures.py
[CI 전용] 고정 픽스처 상품을 '로컬' ChromaDB 에 색인한다.

[왜 필요한가 — Option 2: 격리된 평가 환경]
- RAGAs CI 게이트는 GitHub Actions 러너에서 돈다. 러너는 운영 서버의
  Oracle Autonomous DB / 운영 ChromaDB 에 네트워크로 닿을 수 없고, 닿게 만들면
  운영 인프라를 외부에 노출하는 보안 부담이 생긴다.
- 그래서 CI 에서는 운영 데이터 대신 '고정 픽스처(products.json)'를 러너 안에서
  띄운 임시 ChromaDB 에 색인하고, 그 안에서만 평가한다. 운영 데이터/네트워크에
  전혀 의존하지 않는 '재현 가능한' 품질 게이트가 된다.

[하는 일]
1. scripts/ci_fixtures/products.json 로드
2. text-embedding-3-small 로 배치 임베딩(OPENAI_API_KEY 필요)
3. 로컬 ChromaDB(CHROMA_HOST/PORT, 기본 localhost:8001)에 upsert
   → evaluate_rag.py 의 search_and_rerank() 가 이 컬렉션을 그대로 검색한다.

[주의]
- 이 스크립트는 CI 워크플로우에서만 호출한다. 운영 서버에서 실행하면
  운영 ChromaDB 컬렉션을 픽스처로 덮어쓸 수 있으므로 절대 실행하지 말 것.
  (안전장치로 CI=true 환경에서만 동작하도록 가드한다)
"""
import asyncio
import json
import logging
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from services import embed_service, chroma_service
# 최초 인덱싱과 동일한 텍스트/메타 규칙 재사용(불일치 방지)
from scripts.index_products import _embed_text_for, _to_float

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ci_index_fixtures")

_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "ci_fixtures", "products.json")


async def main() -> None:
    # 안전장치: CI 환경에서만 실행(운영 ChromaDB 오염 방지)
    if os.getenv("CI", "").lower() != "true" and os.getenv("ALLOW_FIXTURE_INDEX", "").lower() != "true":
        logger.error(
            "이 스크립트는 CI 전용입니다(운영 ChromaDB 오염 방지). "
            "로컬에서 강제 실행하려면 ALLOW_FIXTURE_INDEX=true 를 설정하세요."
        )
        sys.exit(1)

    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    logger.info("픽스처 상품 %d건 로드", len(rows))

    # ChromaDB 연결 확인
    hb = await chroma_service.heartbeat()
    logger.info("ChromaDB 연결 OK (heartbeat=%s)", hb)

    ids, embed_texts, documents, metadatas = [], [], [], []
    for row in rows:
        pid = row.get("product_id")
        text = _embed_text_for(row)
        ids.append(str(pid))
        embed_texts.append(text)
        documents.append(text)
        metadatas.append({
            "product_id": int(pid),
            "product_name": row.get("product_name") or "",
            "category": row.get("category") or "",
            "price": _to_float(row.get("price")),
            "description": str(row.get("description")) if row.get("description") else "",
            "stock": int(row.get("stock")) if row.get("stock") is not None else 0,
            "image_url": row.get("image_url") or "",
        })

    embeddings = await embed_service.get_embeddings(embed_texts)
    total = await chroma_service.upsert_products(ids, embeddings, documents, metadatas)
    logger.info("CI 픽스처 색인 완료: 총 %d건", total)


if __name__ == "__main__":
    asyncio.run(main())
