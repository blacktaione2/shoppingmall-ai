from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class IntentType(str, Enum):
    STRUCTURED_QUERY = "STRUCTURED_QUERY"
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"
    FAQ = "FAQ"
    ORDER_INQUIRY = "ORDER_INQUIRY"
    COMPLAINT = "COMPLAINT"
    SMALL_TALK = "SMALL_TALK"


class SortType(str, Enum):
    PRICE_ASC = "PRICE_ASC"
    PRICE_DESC = "PRICE_DESC"
    LATEST = "LATEST"
    DEFAULT = "DEFAULT"


class Entities(BaseModel):
    category: Optional[str] = Field(
        None,
        description="상품 카테고리. 질문에 명시된 경우만 추출 (예: 상의, 하의, 신발, 가전 등)",
    )
    price_max: Optional[int] = Field(
        None,
        description="최대 가격 조건 (원 단위 정수). '5만원 이하' → 50000. 언급 없으면 null",
    )
    price_min: Optional[int] = Field(
        None,
        description="최소 가격 조건 (원 단위 정수). '3만원 이상' → 30000. 언급 없으면 null",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="검색에 사용할 핵심 키워드 목록. 조사/불용어 제외 (예: ['겨울', '패딩'])",
    )
    sort_by: Optional[SortType] = Field(
        None,
        description="정렬 조건. '싼 순서'→PRICE_ASC, '비싼 순서'→PRICE_DESC, '최신'→LATEST, 명시적 정렬 언급 없으면 null",
    )
    # === ORDER_INQUIRY 전용 ===
    order_id: Optional[str] = Field(
        None,
        description="주문번호(숫자). '3번 주문'처럼 특정 주문이 명시된 경우만 "
        "그 숫자를 문자열로 추출(예: '3'). 언급 없으면 null",
    )


class IntentResult(BaseModel):
    intent: IntentType = Field(
        ...,
        description="사용자 질문의 인텐트 분류 결과",
    )
    entities: Entities = Field(
        default_factory=Entities,
        description="질문에서 추출한 엔티티. STRUCTURED_QUERY/SEMANTIC_SEARCH에서 주로 사용",
    )
    emotion: Optional[str] = Field(
        None,
        description="감정 상태. COMPLAINT일 때 필수 (예: 분노, 실망, 불안). 그 외 인텐트는 null 가능",
    )
    confidence: float = Field(
        0.0,
        description="분류 확신도 0.0~1.0. 애매한 질문일수록 낮게 책정",
    )


def coerce_intent_result(value) -> IntentResult:
    """State 의 intent_result 를 IntentResult 인스턴스로 정규화한다(경계선 검증).

    체크포인트(MemorySaver/Redis)에는 직렬화 안전성을 위해 dict(primitive)만
    저장하고, 노드/라우터 진입점에서 이 함수로 복원해 Pydantic 타입 안정성을
    유지한다. IntentResult 가 직접 들어오는 경로(테스트 주입 등)도 그대로 허용.
    """
    if isinstance(value, IntentResult):
        return value
    return IntentResult.model_validate(value)
