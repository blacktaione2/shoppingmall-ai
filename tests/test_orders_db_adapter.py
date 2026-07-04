"""
tests/test_orders_db_adapter.py  [주문조회 실DB 전환]
oracle_db 의 주문 조회 어댑터(_build_orders_from_rows) 단위 테스트.

DB 연결 없이 '조인 결과 행 → Mock 동일 구조 dict' 변환만 검증한다.
(fetch_orders/fetch_order_by_id 의 SQL 실행은 통합 환경에서 별도 확인)

검증 항목:
  1) 다중 item 주문 그룹핑 + 최신순 보존
  2) order_date(TIMESTAMP) → 'YYYY-MM-DD' 포맷
  3) TOTAL_PRICE NULL → items 합산 폴백
  4) ORDER_STATUS NULL → '상태 미확인' 폴백
  5) item 0개 주문(LEFT JOIN 빈 행) 처리
  6) 포맷 함수(_format_order_*) 무수정 재사용
  7) fetch_order_by_id 의 비숫자 order_id → None
"""
from datetime import datetime

import database.oracle_db as odb
from graph.nodes import _format_order_list, _format_order_detail


# rows 컬럼: (ORDER_ID, ORDER_DATE, ORDER_STATUS, TOTAL_PRICE, PRODUCT_NAME, QUANTITY, PRICE)
def test_group_multi_item_and_order_preserved():
    rows = [
        (2, datetime(2026, 6, 10), "배송중", 107000, "여름용 린넨 셔츠", 2, 39000),
        (2, datetime(2026, 6, 10), "배송중", 107000, "코튼 반바지", 1, 29000),
        (1, datetime(2026, 6, 1), "배송완료", 59000, "스니커즈", 1, 59000),
    ]
    orders = odb._build_orders_from_rows(rows)
    assert len(orders) == 2
    # 입력 순서(최신 2번 먼저) 보존
    assert orders[0]["order_id"] == "2"
    assert len(orders[0]["items"]) == 2
    assert orders[0]["order_date"] == "2026-06-10"


def test_total_price_null_fallback():
    rows = [(3, datetime(2026, 6, 15), None, None, "상품A", 3, 10000)]
    o = odb._build_orders_from_rows(rows)[0]
    assert o["total_price"] == 30000      # 3 × 10000
    assert o["status"] == "상태 미확인"    # ORDER_STATUS NULL 폴백


def test_empty_item_order():
    rows = [(4, datetime(2026, 6, 20), "배송준비중", 0, None, None, None)]
    o = odb._build_orders_from_rows(rows)[0]
    assert o["items"] == []
    assert o["total_price"] == 0


def test_format_functions_reusable():
    rows = [
        (2, datetime(2026, 6, 10), "배송중", 68000, "셔츠", 2, 39000),
    ]
    orders = odb._build_orders_from_rows(rows)
    list_out = _format_order_list(orders)
    assert "셔츠 x2" in list_out
    assert "🚚" in list_out               # 배송중 이모지 매핑 동작
    detail_out = _format_order_detail(orders[0])
    assert "**2**" in detail_out          # 주문번호(숫자) 표시


def test_fetch_order_by_id_non_numeric_returns_none(monkeypatch):
    # 비숫자 order_id 는 DB 조회 전에 None 반환(SQL 실행 안 함)
    called = {"db": False}

    class _BoomConn:
        def __enter__(self):
            called["db"] = True
            raise AssertionError("비숫자 order_id 인데 DB 에 접근하면 안 됨")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(odb, "get_connection", lambda: _BoomConn())
    for bad in ["ORD-20260601-0001", "abc", None, ""]:
        assert odb.fetch_order_by_id(1, bad) is None
    assert called["db"] is False
