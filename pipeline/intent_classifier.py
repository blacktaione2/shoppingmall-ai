import logging

from schemas.intent_schema import IntentResult, IntentType
from services.gpt_service import structured_completion

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """당신은 쇼핑몰 챗봇의 인텐트 분류기입니다.
사용자 질문을 아래 6개 인텐트 중 정확히 하나로 분류하고 엔티티를 추출하세요.

[인텐트 정의]
1. STRUCTURED_QUERY: 가격/카테고리/정렬 등 명확한 조건 기반 상품 검색
   예) "5만원 이하 신발 보여줘", "300만원 이상 가전 있어?", "상의 중에 제일 싼 거", "10만원대 가전 있어?"
2. SEMANTIC_SEARCH: 추상적/감성적 표현의 상품 탐색 (조건이 아닌 의미 기반)
   예) "겨울에 따뜻하게 입을 만한 거 추천해줘", "선물하기 좋은 상품 있을까?"
3. FAQ: 배송 정책, 환불/교환 규정, 회원 혜택 등 쇼핑몰 정책 질문
   예) "배송은 며칠 걸려요?", "환불 규정이 어떻게 되나요?"
4. ORDER_INQUIRY: 본인 주문/배송 상태 조회
   예) "내 주문 어디까지 왔어?", "어제 시킨 거 배송 조회해줘"
   예) "3번 주문 배송 상태 알려줘" → entities.order_id = "3"
5. COMPLAINT: 불만, 항의, 부정적 감정 표출
   예) "배송이 일주일째 안 와요 진짜 화나네요", "상품이 파손돼서 왔어요"
6. SMALL_TALK: 인사, 잡담, 쇼핑과 무관한 대화
   예) "안녕", "너 누구야?", "오늘 날씨 좋다"

[분류 규칙]
- 가격/정렬 같은 '구체적 조건'이 하나라도 있으면 STRUCTURED_QUERY 우선
- 조건 없이 의미/상황 중심이면 SEMANTIC_SEARCH
- 불만 + 주문조회가 섞이면 감정 표출이 주된 목적일 때만 COMPLAINT
- COMPLAINT일 때는 emotion 필드를 반드시 채울 것 (예: 분노, 실망, 불안)
- 가격은 원 단위 정수로 변환 (5만원 → 50000)
- 가격 비교 표현 매핑: '이상'/'넘는'은 price_min, '이하'/'안 되는'은 price_max로 추출
  (금액 크기와 무관하게 항상 이 규칙 적용 — 이상↔이하 혼동 주의)
- 질문에 특정 주문번호(숫자, 예: 3, 12)가 명시되어 있으면 entities.order_id 에
  그 숫자를 문자열로 추출(예: "3번 주문" → "3"). 주문번호 언급이 없으면 null
- 이전 대화가 있다면 참고해서 지시어("그 중", "그거", "이 중에서" 등)를 해석할 것.
  단, 지시어가 가리키는 대상이 이전 대화 속 특정 상품/추천 목록이라 새 조건이
  구체적 필터(카테고리/가격 범위)로 확정되지 않으면 STRUCTURED_QUERY로 섣불리
  분류하지 말고 SEMANTIC_SEARCH로 분류할 것(그래야 맥락을 아는 답변 생성 가능)
- confidence는 0.0~1.0, 애매하면 낮게 책정"""


async def classify_intent(question: str) -> IntentResult:
    """사용자 질문을 GPT Structured Outputs(gpt_service.GPT_MODEL_INTENT="gpt-5.4-mini")로 인텐트 분류한다."""
    try:
        result = await structured_completion(
            system_prompt=INTENT_SYSTEM_PROMPT,
            user_message=question,
            response_model=IntentResult,
        )
        logger.info(
            "intent=%s confidence=%.2f question=%s",
            result.intent.value, result.confidence, question,
        )
        return result
    except Exception:
        logger.exception("인텐트 분류 실패, SMALL_TALK 폴백: %s", question)
        return IntentResult(
            intent=IntentType.SMALL_TALK,
            confidence=0.0,
        )