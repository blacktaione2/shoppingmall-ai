"""
graph/guard.py
환각 방어 가드 (State 기반 재설계).

[기존 대비 변경점]
- 기존 pipeline/hallucination_guard.py 는 hits 를 ContextVar(get_rag_context)로
  읽었다. LangGraph 에서는 semantic_node 가 state["rag_hits"] 에 담아 넘기므로
  ContextVar 의존을 제거하고 rag_hits 를 인자로 직접 받는다.
- 순수 검증 로직(_validate_semantic_answer, _is_rejection, 가격/상품명 추출 등)은
  기존 hallucination_guard 의 함수를 그대로 재사용한다(중복 구현 방지, 단일 출처).
- 프리픽스([추천]) 제거 방침에 맞춰, SEMANTIC 폴백 문구도 프리픽스 없이 정의한다.
- SEMANTIC 재시도는 get_intent_llm()(저비용 역할)으로 1회 수행한다. 1차 생성
  (rag_service.generate_rag_response)과 마찬가지로 멀티모델 팩토리를 경유하므로
  .env LLM_PROVIDER 에 따라 재시도 모델도 함께 전환된다 → provider 벤치마크 시
  '1차 생성은 DeepSeek, 재시도만 OpenAI' 같은 비대칭(메트릭 오염)을 방지한다.
  재시도 프롬프트에는 이력을 넣지 않는다(재시도는 '컨텍스트만 근거' 보정이 목적).
"""
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from schemas.intent_schema import IntentResult, IntentType
from services import rag_service
from graph.llm import get_intent_llm, date_system_prefix

# 기존 환각 가드의 순수 검증 로직 재사용 (단일 출처)
from pipeline.hallucination_guard import (
    _validate_semantic_answer,
    _has_factual_claim,
    GUIDANCE_SUFFIX,
    _RETRY_SYSTEM_SUFFIX,
)

logger = logging.getLogger(__name__)

# 프리픽스 제거 방침에 맞춘 SEMANTIC 최종 폴백 문구 ([추천] 프리픽스 없음)
SEMANTIC_FALLBACK_MESSAGE = (
    "죄송합니다, 정확한 상품 정보를 확인하기 어려워요. 다른 검색어로 다시 시도해 주세요."
)


async def _retry_semantic_answer(question: str, hits: list[dict]) -> str:
    """1차 검증 실패 시 get_intent_llm()(저비용 역할)으로 1회 재시도 (보정 호출).

    1차 생성과 동일하게 멀티모델 팩토리를 경유한다(provider 전환됨). 재시도 프롬프트는
    rag_service.SYSTEM_PROMPT + _RETRY_SYSTEM_SUFFIX 를 system 으로 쓰고, 질문/컨텍스트는
    rag_service.RAG_HUMAN_TEMPLATE 자리표시자로 안전하게 채운다(중괄호 KeyError 방지).
    이력은 넣지 않는다('컨텍스트만 근거' 보정 목적).
    """
    context = rag_service.build_product_context(hits)
    retry_system_prompt = (
        f"{date_system_prefix()}\n\n{rag_service.SYSTEM_PROMPT}{_RETRY_SYSTEM_SUFFIX}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", retry_system_prompt),
        ("human", rag_service.RAG_HUMAN_TEMPLATE),
    ])
    # 재시도는 저비용 역할(get_intent_llm) + temperature 0.0(결정적 보정)
    chain = prompt | get_intent_llm(temperature=0.0) | StrOutputParser()
    return await chain.ainvoke({"question": question, "context": context})


async def _guard_semantic(question: str, answer: str, hits: list[dict]) -> str:
    """SEMANTIC_SEARCH 환각 가드 본체 (rag_hits 를 인자로 받음)."""
    # hits 0건 → semantic_node 의 고정 안내(GPT 미호출) → 검증 불필요
    if not hits:
        return answer

    if _validate_semantic_answer(answer, hits):
        return answer

    logger.warning("SEMANTIC 환각 의심 → 저비용 모델(get_intent_llm) 1회 재시도. question=%r", question)
    try:
        retried = await _retry_semantic_answer(question, hits)
    except Exception:
        logger.exception("환각 재시도 호출 중 오류 발생")
        return SEMANTIC_FALLBACK_MESSAGE

    retried_answer = (retried or "").strip()
    # 재검증: 기존 _validate_semantic_answer 는 프리픽스 유무에 영향받지 않음
    if retried_answer and _validate_semantic_answer(retried_answer, hits):
        return retried_answer

    logger.warning("재시도 후에도 환각 의심 → 안전 문구로 폴백. question=%r", question)
    return SEMANTIC_FALLBACK_MESSAGE


async def guard_answer_state(
    question: str,
    answer: str,
    intent_result: IntentResult,
    rag_hits: list[dict],
    history: list[dict] | None = None,
) -> str:
    """인텐트별 환각 방어 진입점 (State 기반).

    Args:
        question      : 사용자 원본 질문
        answer        : 핸들러 노드가 만든 raw_answer
        intent_result : 분류 결과
        rag_hits      : semantic_node 가 넘긴 검색 컨텍스트(State 경유)
        history       : 멀티턴 이력 (현재 가드 재시도에는 미사용, 시그니처 확장 대비)
    """
    if intent_result.intent == IntentType.SEMANTIC_SEARCH:
        return await _guard_semantic(question, answer, rag_hits)

    if intent_result.intent in (IntentType.COMPLAINT, IntentType.SMALL_TALK):
        if _has_factual_claim(answer):
            return answer + GUIDANCE_SUFFIX
        return answer

    # STRUCTURED_QUERY / FAQ / ORDER_INQUIRY: GPT 미사용 → pass-through
    return answer
