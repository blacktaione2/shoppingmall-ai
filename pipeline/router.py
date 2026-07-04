"""
[LEGACY] pipeline/router.py
LangGraph 전환 이후 '진입점'으로는 사용되지 않는다.
아래 함수들은 graph/ 노드에서 재사용되거나, 기존 테스트(test_strip_prefix.py)가
참조하므로 파일 자체는 유지한다.
  - strip_known_prefix(): graph/ 에서 프리픽스를 안 붙이므로 사실상 no-op.
  - route_intent(): scripts/test_integration.py 에서만 참조.
삭제 타이밍: 레거시 테스트 정리 시 함께 제거 가능.
----------------------------------------------------------------------
파이프라인 라우터: IntentResult → 핸들러 분기

[설계 메모]
- COMPLAINT / SMALL_TALK 은 별도 핸들러 파일이 없는 구조(확정 구조 기준)이므로
  본 파일 내부의 비공개 함수로 정의했다. gpt_service.chat_completion() 기반
  실제 응답으로 구현 완료.
- 핸들러 시그니처는 전부 동일: async (question: str, intent_result: IntentResult) -> str
- route_intent() 진입 시 reset_rag_context() 호출:
  SEMANTIC_SEARCH 핸들러가 ContextVar에 남긴 이전 hits를 다음 호출로 새지 않도록
  방어적으로 초기화한다 (pipeline_context.py 모듈 docstring 참고).
"""
from schemas.intent_schema import IntentResult, IntentType
from services import gpt_service

from pipeline.structured_handler import handle_structured
from pipeline.semantic_handler import handle_semantic
from pipeline.faq_handler import handle_faq
from pipeline.order_handler import handle_order
from pipeline.pipeline_context import reset_rag_context


# ── COMPLAINT 핸들러 ────────────────────────────────────────────────────
_COMPLAINT_SYSTEM_PROMPT = (
    "당신은 온라인 쇼핑몰의 고객 응대 상담사입니다. "
    "고객이 불만이나 항의를 표현하고 있습니다. "
    "고객의 감정에 깊이 공감하며 진심으로 사과하는 톤으로 답변하세요. "
    "단, 당신은 실제 주문/배송/환불 데이터에 접근할 수 없으므로 "
    "'환불 처리했습니다', '내일 도착합니다'처럼 구체적인 사실을 단정적으로 말하지 마세요. "
    "공감 표현 후, 필요하다면 '주문조회' 기능이나 고객센터(1234-5678, 평일 09:00~18:00) "
    "이용을 안내하세요. 답변은 2~3문장으로 간결하게 작성하세요."
)

# 라우팅 검증용 프리픽스 ([LEGACY] graph/ 경로는 프리픽스를 붙이지 않으므로 미사용)
_COMPLAINT_PREFIX = "[상담]"

# GPT 응답이 비었을 때(드묾) 사용할 안전 문구
_COMPLAINT_FALLBACK = "불편을 드려 죄송합니다. 자세한 사항은 고객센터(1234-5678)로 문의해 주세요."


async def _handle_complaint(question: str, intent_result: IntentResult) -> str:
    """COMPLAINT 인텐트: gpt-5.4 기반 감정 공감 응답.

    intent_result.emotion (분류기가 추출한 감정, 예: 분노/실망/불안)을
    프롬프트에 함께 전달하여 응답 톤을 보정한다.
    """
    emotion = intent_result.emotion or "불편함"
    user_prompt = f"[고객 감정: {emotion}]\n[고객 메시지]\n{question}"
    answer = await gpt_service.chat_completion(
        system_prompt=_COMPLAINT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=gpt_service.GPT_MODEL_MAIN,
        temperature=0.7,
    )
    answer = (answer or "").strip() or _COMPLAINT_FALLBACK
    return f"{_COMPLAINT_PREFIX} {answer}"


# ── SMALL_TALK 핸들러 ───────────────────────────────────────────────────
_SMALL_TALK_SYSTEM_PROMPT = (
    "당신은 온라인 쇼핑몰의 친근한 AI 챗봇입니다. "
    "고객의 인사나 잡담에 가볍고 친근하게 응답하세요. "
    "쇼핑몰과 무관한 질문에도 정중히 답하면서, 자연스럽다면 쇼핑 관련 도움을 줄 수 있다고 "
    "안내해도 좋습니다. 답변은 1~2문장으로 짧게 작성하세요."
)

_SMALL_TALK_PREFIX = "[잡담]"

# GPT 응답이 비었을 때(드묾) 사용할 안전 문구
_SMALL_TALK_FALLBACK = "안녕하세요! 무엇을 도와드릴까요?"


async def _handle_small_talk(question: str, intent_result: IntentResult) -> str:
    """SMALL_TALK 인텐트: gpt-5.4 기반 가벼운 응답."""
    answer = await gpt_service.chat_completion(
        system_prompt=_SMALL_TALK_SYSTEM_PROMPT,
        user_prompt=question,
        model=gpt_service.GPT_MODEL_MAIN,
        temperature=0.8,
    )
    answer = (answer or "").strip() or _SMALL_TALK_FALLBACK
    return f"{_SMALL_TALK_PREFIX} {answer}"


HANDLER_MAP = {
    IntentType.STRUCTURED_QUERY: handle_structured,
    IntentType.SEMANTIC_SEARCH: handle_semantic,
    IntentType.FAQ: handle_faq,
    IntentType.ORDER_INQUIRY: handle_order,
    IntentType.COMPLAINT: _handle_complaint,
    IntentType.SMALL_TALK: _handle_small_talk,
}


# ── 프리픽스 제거 유틸 ─────────────────────────────────────────────────
# [배경] 각 핸들러는 라우팅 검증 편의를 위해 답변 맨 앞에 인텐트 프리픽스를 붙인다
#        (예: "[추천] ...", "[상담] ..."). 프론트는 ChatResponse.intent 필드로
#        인텐트를 분기하므로, 사용자에게 보이는 answer 텍스트에서는 프리픽스를
#        제거해 깨끗한 본문만 전달한다.
# [설계] blanket 정규식(^\[...\]) 대신 '알려진 6개 프리픽스'만 화이트리스트로 제거한다.
#        GPT가 우연히 "[중요]" 같은 자체 강조 표현으로 답변을 시작해도 본문이
#        잘못 잘리지 않도록 하기 위함. (false strip 방지)
# [순서] chat.py 에서 반드시 guard_answer() 통과 '이후'에 호출할 것.
#        hallucination_guard 는 answer 전체를 검사하므로 프리픽스 유무에 영향받지 않지만,
#        가드의 SEMANTIC 재시도/폴백 문구도 "[추천]" 으로 시작하기에 가드를 모두 거친
#        최종 문자열에 한 번만 적용해야 일관성이 보장된다.
_STRUCTURED_PREFIX = "[검색결과]"  # STRUCTURED_QUERY
_SEMANTIC_PREFIX = "[추천]"        # SEMANTIC_SEARCH (hallucination_guard.SEMANTIC_FALLBACK_MESSAGE 와 동일)
_FAQ_PREFIX = "[FAQ]"             # FAQ
_ORDER_PREFIX = "[주문조회]"       # ORDER_INQUIRY
# _COMPLAINT_PREFIX("[상담]") / _SMALL_TALK_PREFIX("[잡담]") 는 위에서 이미 정의됨 → 재사용

# 제거 대상 프리픽스 집합. (튜플 순서는 상관없음 - startswith 단일 매칭)
_KNOWN_PREFIXES = (
    _STRUCTURED_PREFIX,
    _SEMANTIC_PREFIX,
    _FAQ_PREFIX,
    _ORDER_PREFIX,
    _COMPLAINT_PREFIX,
    _SMALL_TALK_PREFIX,
)


def strip_known_prefix(answer: str) -> str:
    """답변 맨 앞의 '알려진 인텐트 프리픽스' 1개를 제거한다.

    - answer 가 _KNOWN_PREFIXES 중 하나로 시작하면 그 프리픽스 + 뒤따르는 공백을 제거.
    - 어떤 프리픽스로도 시작하지 않으면 원본을 그대로 반환(안전한 no-op).
    - 프리픽스는 답변당 최대 1개만 부착되므로 첫 매칭에서 즉시 반환한다.

    Args:
        answer: 핸들러 + 환각 가드를 모두 거친 최종 답변 문자열

    Returns:
        프리픽스가 제거된(또는 없으면 원본 그대로) 문자열
    """
    if not answer:
        return answer
    for prefix in _KNOWN_PREFIXES:
        if answer.startswith(prefix):
            # 프리픽스 길이만큼 잘라낸 뒤 앞쪽 공백/개행만 제거(본문 끝 공백은 보존)
            return answer[len(prefix):].lstrip()
    return answer


async def route_intent(question: str, intent_result: IntentResult) -> str:
    """분류된 인텐트에 해당하는 핸들러를 호출해 답변 문자열을 반환"""
    # 이전 호출의 잔류 RAG 컨텍스트 제거 (pipeline_context.py 참고)
    reset_rag_context()

    handler = HANDLER_MAP.get(intent_result.intent)
    if handler is None:
        # IntentType Enum 검증을 통과하면 사실상 도달 불가하지만 방어적으로 처리
        return "죄송합니다. 지원하지 않는 요청 유형입니다."
    return await handler(question, intent_result)
