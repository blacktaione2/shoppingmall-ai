"""
pipeline/order_handler.py [LEGACY — graph/nodes.py 의 order_node 가 실DB 기반으로 대체]
ORDER_INQUIRY 인텐트 핸들러 (레거시, mock/order_mock.py 고정 데이터 기반)

- mock/order_mock.py 의 고정 데이터를 Markdown 템플릿 문자열로 포맷팅
- GPT 미사용 (환각 가능성 없음 → 환각 방어 대상 아님)
- router.py HANDLER_MAP 시그니처 규약 준수:
      async (question: str, intent_result: IntentResult) -> str
- DB/네트워크 I/O 없음 → asyncio.to_thread 불필요

[조회 트랙 2종]
  1) intent_result.entities.order_id 있음 → 단건 상세 (배송상태 강조)
  2) order_id 없음 → 전체 목록 (get_mock_orders)
"""
from schemas.intent_schema import IntentResult
from mock.order_mock import get_mock_orders, get_order_by_id
from pipeline.pipeline_context import get_current_member_id

# MOCK_MEMBER_ID 하드코딩 제거 → pipeline_context.get_current_member_id() 로 교체.
# routers/chat.py 가 chat_token(Token Bridge)을 역조회한 실제 member_id 를
# set_current_member_id() 로 저장해두면 여기서 그 값을 그대로 읽는다.
# 컨텍스트가 세팅되지 않은 경로(예: route_intent 를 직접 호출하는 오프라인 테스트)는
# get_current_member_id() 의 기본값 1 로 기존 동작과 동일하게 유지된다.

# 라우팅 검증용 프리픽스 ([LEGACY] graph/ 경로는 프리픽스를 붙이지 않으므로 미사용)
PREFIX = "[주문조회]"

# 배송상태 강조용 이모지
_STATUS_EMOJI = {
    "배송완료": "✅",
    "배송중": "🚚",
    "배송준비중": "⏳",
}


def _format_items(items: list[dict]) -> str:
    """주문 내 상품 목록 포맷 (한 주문에 여러 상품 가능)"""
    lines = [
        f"   - {item['product_name']} x{item['quantity']} ({item['price']:,}원)"
        for item in items
    ]
    return "\n".join(lines)


def _format_order_list(orders: list[dict]) -> str:
    """전체 주문 목록 포맷"""
    if not orders:
        return "주문 내역이 없습니다."

    blocks = [f"회원님의 주문 내역 {len(orders)}건입니다."]
    for order in orders:
        emoji = _STATUS_EMOJI.get(order["status"], "📦")
        blocks.append(
            f"📦 **{order['order_id']}** ({order['order_date']})\n"
            f"{_format_items(order['items'])}\n"
            f"   - 총 결제금액: {order['total_price']:,}원\n"
            f"   - 배송상태: {order['status']} {emoji}"
        )
    return "\n\n".join(blocks)


def _format_order_detail(order: dict) -> str:
    """단건 상세 포맷 (배송상태 강조)"""
    emoji = _STATUS_EMOJI.get(order["status"], "📦")
    return (
        f"주문번호 **{order['order_id']}** 조회 결과입니다.\n\n"
        f"📦 주문일: {order['order_date']}\n"
        f"{_format_items(order['items'])}\n"
        f"- 총 결제금액: {order['total_price']:,}원\n"
        f"- 배송상태: **{order['status']}** {emoji}"
    )


def _format_not_found(order_id: str) -> str:
    """주문번호 미존재 포맷 (예시는 실제 order_id 포맷 기준)"""
    return (
        f"주문번호 '{order_id}'에 해당하는 주문 내역을 찾을 수 없습니다.\n"
        f"주문번호를 다시 확인해 주세요. (예: ORD-20260601-0001)"
    )


async def handle_order(question: str, intent_result: IntentResult) -> str:
    """ORDER_INQUIRY 처리 진입점

    question      : 사용자 원본 질문 (현재 미사용 - 다른 핸들러와 시그니처 통일 +
                     향후 확장 대비 수신)
    intent_result : 분류기 결과. entities.order_id 로 단건/전체 분기.
                     member_id 는 IntentResult 에 없으므로 pipeline_context 의
                     ContextVar(get_current_member_id())로 조회한다.
    """
    member_id = get_current_member_id()

    order_id = intent_result.entities.order_id
    order_id = order_id.strip() if isinstance(order_id, str) else None

    if order_id:
        # 단건 상세 조회 트랙
        order = get_order_by_id(member_id, order_id)
        body = _format_order_detail(order) if order else _format_not_found(order_id)
    else:
        # 전체 목록 조회 트랙
        body = _format_order_list(get_mock_orders(member_id))

    return f"{PREFIX}\n{body}"
