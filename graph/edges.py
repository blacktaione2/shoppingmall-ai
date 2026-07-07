"""
graph/edges.py
LangGraph 조건부 분기(conditional edge) 함수.

[설계]
- classify_node 이후 intent_result.intent 값에 따라 핸들러 노드로 분기한다.
  기존 router.HANDLER_MAP 딕셔너리 룩업을 LangGraph conditional_edge 로 대체.
- 게스트 주문조회 차단:
  기존 chat.py 가 분류 직후 ORDER_INQUIRY + is_guest 를 막았는데,
  그래프 안에서 처리하기 위해 별도 'guest_block_node' 로 분기시킨다.
  → 그래프 흐름의 일관성(모든 응답이 동일 경로로 guard 까지 가도록) 확보.
"""
from schemas.intent_schema import IntentType, coerce_intent_result
from graph.state import ShoppingState


# intent → 핸들러 노드 이름 매핑 (builder 의 add_node 이름과 일치해야 함)
_INTENT_TO_NODE = {
    IntentType.STRUCTURED_QUERY: "structured",
    IntentType.SEMANTIC_SEARCH: "semantic",
    IntentType.FAQ: "faq",
    IntentType.ORDER_INQUIRY: "order",
    IntentType.COMPLAINT: "complaint",
    IntentType.SMALL_TALK: "small_talk",
}


# [되묻기] confidence 가 이 값 미만이면 intent 판단과 무관하게 무조건 되묻는다.
# 분류 프롬프트 자체가 "애매하면 confidence 를 낮게 책정"하라고 지시하는데,
# 지금까지는 이 값이 로그에만 쓰이고 라우팅에는 반영되지 않았다.
# 배포 후 실제 confidence 분포를 보고 조정 필요(초기값은 상식적 추정치).
_CONFIDENCE_THRESHOLD = 0.5


def route_by_intent(state: ShoppingState) -> str:
    """분류 결과로 다음 노드 이름을 반환한다.

    - confidence 가 임계값 미만이면 intent 무관하게 'clarify'로 보낸다.
    - 게스트(is_guest=True)가 ORDER_INQUIRY 를 요청하면 'guest_block' 으로 보낸다.
    - 그 외에는 인텐트에 매핑된 핸들러 노드명을 반환.
    - 매핑 밖 값(이론상 도달 불가)은 방어적으로 small_talk 로 폴백.
    """
    intent_result = coerce_intent_result(state["intent_result"])

    if intent_result.confidence < _CONFIDENCE_THRESHOLD:
        return "clarify"

    intent = intent_result.intent
    if intent == IntentType.ORDER_INQUIRY and state.get("is_guest"):
        return "guest_block"

    return _INTENT_TO_NODE.get(intent, "small_talk")
