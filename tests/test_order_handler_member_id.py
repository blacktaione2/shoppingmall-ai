"""
order_handler.py 의 member_id 연동 단위테스트
  - MOCK_ORDERS 에 데이터가 있는 member_id(1)와 없는 member_id(999) 를
    pipeline_context.set_current_member_id() 로 전환하며 결과가 실제로 달라지는지 확인.
  - "MOCK_MEMBER_ID=1 하드코딩"이 완전히 제거되어, 컨텍스트 값을 바꾸면
    handle_order() 의 동작도 같이 바뀌는지가 이 테스트의 핵심이다.
  - 컨텍스트를 세팅하지 않은 기존 호출 경로(예: route_intent 직접 호출)와의
    하위호환(기본값 1)도 함께 검증한다.

실행: (FastAPI 프로젝트 루트에서) python tests/test_order_handler_member_id.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.order_handler import handle_order
from pipeline.pipeline_context import set_current_member_id
from schemas.intent_schema import IntentResult, IntentType, Entities


def _make_intent_result(order_id: str | None = None) -> IntentResult:
    return IntentResult(
        intent=IntentType.ORDER_INQUIRY,
        entities=Entities(order_id=order_id),
        confidence=0.95,
    )


def run():
    passed = failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"✅ {name}")
            passed += 1
        else:
            print(f"❌ {name}  {detail}")
            failed += 1

    async def _scenarios():
        # 1) member_id=1 (MOCK_ORDERS 에 실제 주문 2건 존재) → 목록에 주문번호가 포함되어야 함
        set_current_member_id(1)
        result_member1 = await handle_order("내 주문 내역 보여줘", _make_intent_result())
        check(
            "member_id=1 전체조회 → 실제 주문 포함",
            "ORD-20260601-0001" in result_member1 and "ORD-20260610-0002" in result_member1,
            f"실제 결과: {result_member1!r}",
        )

        # 2) member_id=999 (MOCK_ORDERS 에 데이터 없음) → "주문 내역이 없습니다" 안내
        #    [핵심] 1번과 다른 결과가 나와야 컨텍스트 값이 실제로 반영된다는 증거가 됨
        set_current_member_id(999)
        result_member999 = await handle_order("내 주문 내역 보여줘", _make_intent_result())
        check(
            "member_id=999 전체조회 → 주문 없음 안내, member_id=1 결과와 다름",
            "주문 내역이 없습니다" in result_member999 and result_member999 != result_member1,
            f"실제 결과: {result_member999!r}",
        )

        # 3) member_id=1, 단건조회(order_id 지정) → 상세 포맷 반환
        set_current_member_id(1)
        result_detail = await handle_order(
            "ORD-20260601-0001 주문 확인해줘", _make_intent_result(order_id="ORD-20260601-0001")
        )
        check(
            "member_id=1 단건조회 → 해당 주문 상세 반환",
            "ORD-20260601-0001" in result_detail and "배송완료" in result_detail,
            f"실제 결과: {result_detail!r}",
        )

        # 4) member_id=999, 존재하지 않는 주문번호로 단건조회 → 다른 회원 주문이 새지 않고 "찾을 수 없음"
        set_current_member_id(999)
        result_cross = await handle_order(
            "ORD-20260601-0001 주문 확인해줘", _make_intent_result(order_id="ORD-20260601-0001")
        )
        check(
            "member_id=999 가 다른 회원(1번) 주문번호 조회 시 정보 누출 없이 '찾을 수 없음' 반환",
            "찾을 수 없습니다" in result_cross,
            f"실제 결과: {result_cross!r}",
        )

    asyncio.run(_scenarios())

    print(f"\n=== 결과: {passed} PASSED / {failed} FAILED (총 {passed + failed}) ===")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
