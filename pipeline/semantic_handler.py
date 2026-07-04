"""
pipeline/semantic_handler.py
SEMANTIC_SEARCH 인텐트 핸들러.

흐름: 질문 임베딩 → ChromaDB top-k 검색 → GPT RAG 응답 (gpt_service.GPT_MODEL_MAIN="gpt-5.4")
- top_k = 4 (확정)
- 거리 임계값 컷 없음 → 검색 결과를 그대로 GPT 에 넘기고 관련성 판단은 프롬프트가 담당
- 반환: 문자열 응답 (FAQ 핸들러와 동일 계약, [추천] 프리픽스)
"""
import logging
from services import embed_service, chroma_service, rag_service
from schemas.intent_schema import IntentResult
from pipeline.pipeline_context import set_rag_context

logger = logging.getLogger(__name__)

SEMANTIC_TOP_K = 4   # 확정: 벡터 검색 결과 수


async def handle_semantic(question: str, intent_result: IntentResult) -> str:
    """
    question      : 사용자 원본 질문 (임베딩 쿼리로 사용)
    intent_result : 인텐트 분류 결과(IntentResult). 현재 SEMANTIC 경로는 entities 미사용.
                    (category/price 구조화 필터는 STRUCTURED 핸들러가 담당)
                    라우터 호출 일관성을 위해 인자로 받아둔다.
    """
    # 1) 질문 임베딩
    query_embedding = await embed_service.get_embedding(question)

    # 2) ChromaDB top-k 벡터 검색 (필터 없음 = 순수 의미 검색)
    hits = await chroma_service.search_similar(query_embedding, n_results=SEMANTIC_TOP_K)
    logger.info("SEMANTIC 검색 %d건 (질문=%r)", len(hits), question)

    # 2-1) hallucination_guard 가 동일 hits 로 가격/상품명 검증을 수행할 수 있도록
    #      ContextVar 사이드채널에 저장 (0건이어도 저장 → guard가 빈 리스트로 판단)
    set_rag_context(hits)

    # 3) 검색 결과 0건(컬렉션 미인덱싱 등) → GPT 호출 없이 고정 안내
    if not hits:
        return "[추천] 죄송합니다, 조건에 맞는 상품을 찾지 못했어요. 다른 검색어로 다시 시도해 주세요."

    # 4) GPT RAG 응답 생성 (관련성 판단/거절은 프롬프트가 담당)
    answer = await rag_service.generate_rag_response(question, hits)
    return f"[추천] {answer}"
