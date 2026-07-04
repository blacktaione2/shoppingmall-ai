"""
주문 Mock 데이터 (레거시 — 현재 운영 경로는 database.oracle_db 의 실DB 조회를 쓴다)
- memberId 1 고정
- pipeline/order_handler.py 가 이 데이터를 문장으로 포맷팅(레거시 경로 전용)
"""

MOCK_ORDERS: dict[int, list[dict]] = {
    1: [
        {
            "order_id": "ORD-20260601-0001",
            "order_date": "2026-06-01",
            "status": "배송완료",
            "items": [
                {"product_name": "클래식 화이트 스니커즈", "quantity": 1, "price": 59000},
            ],
            "total_price": 59000,
        },
        {
            "order_id": "ORD-20260610-0002",
            "order_date": "2026-06-10",
            "status": "배송중",
            "items": [
                {"product_name": "여름용 린넨 셔츠", "quantity": 2, "price": 39000},
                {"product_name": "코튼 반바지", "quantity": 1, "price": 29000},
            ],
            "total_price": 107000,
        },
    ],
}


def get_mock_orders(member_id: int) -> list[dict]:
    """회원 ID 기준 Mock 주문 목록 반환 (없으면 빈 리스트)"""
    return MOCK_ORDERS.get(member_id, [])


def get_order_by_id(member_id: int, order_id: str) -> dict | None:
    """회원 ID + 주문번호로 단건 조회

    - 대소문자/공백 무시: 'ord-20260601-0001', ' ORD-20260601-0001 ' 등도 매칭
    - 해당 회원의 주문 목록(get_mock_orders) 내에서만 검색 (member_id 분리 유지)
    - 없으면 None
    """
    if not order_id:
        return None
    normalized = order_id.strip().upper()
    for order in get_mock_orders(member_id):
        if order["order_id"] == normalized:
            return order
    return None
