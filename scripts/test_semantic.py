"""
scripts/test_semantic.py
SEMANTIC 핸들러 실 통합 테스트 (실제 ChromaDB + 실제 OpenAI 필요).

실행(프로젝트 루트):  python -m scripts.test_semantic
사전조건:
  - python -m scripts.index_products 로 인덱싱 완료
  - .env 설정 완료, ChromaDB 서버(chroma run) 기동
"""
import asyncio
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from services import chroma_service
from pipeline.semantic_handler import handle_semantic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# 실제 상품 데이터(샘플 15개)에 맞춰 질문을 자유롭게 바꿔 테스트하세요.
TEST_QUESTIONS = [
    "겨울에 따뜻하게 입을 만한 옷 추천해줘",
    "운동할 때 신기 좋은 가벼운 신발 있어?",
    "선물하기 좋은 향수 같은 거 없을까?",
    "비 오는 날 신을 수 있는 방수 제품 보여줘",
    "양자컴퓨터 작동 원리 알려줘",   # 상품과 무관 → 정중히 거절 기대
]


async def main():
    # 0) 사전 점검: 컬렉션 건수
    cnt = await chroma_service.count()
    print(f"\n=== ChromaDB '{chroma_service.COLLECTION_NAME}' 컬렉션 건수: {cnt} ===")
    if cnt == 0:
        print("⚠️  컬렉션이 비어 있습니다. 먼저 `python -m scripts.index_products` 를 실행하세요.")
        return

    for q in TEST_QUESTIONS:
        print("\n" + "=" * 60)
        print(f"[질문] {q}")
        answer = await handle_semantic(q, {"intent": "SEMANTIC_SEARCH"})
        print(f"[응답]\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
