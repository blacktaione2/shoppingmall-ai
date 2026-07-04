"""
scripts/dev_checks/check_semantic.py
의미 검색(임베딩 → ChromaDB top-k) 점검용 개발 스크립트.

샘플 질문을 임베딩해 유사 상품 top-k 와 거리(distance)를 출력한다.
검색 파이프라인이 끝단까지 정상 동작하는지 빠르게 확인하는 용도다.
운영 코드가 아니라 배포/디버깅 시 수동 점검용이다.

실행(프로젝트 루트에서):
    python -m scripts.dev_checks.check_semantic
"""
import asyncio

from dotenv import load_dotenv

load_dotenv()

from services import embed_service, chroma_service


async def main():
    emb = await embed_service.get_embedding("겨울에 따뜻하게 입을만한 옷 추천해줘")
    hits = await chroma_service.search_similar(emb, n_results=4)
    for h in hits:
        m = h["metadata"]
        print(f"{m['product_name']} | {m['price']}원 | distance={h['distance']:.4f}")


asyncio.run(main())
