"""
pipeline_context.py 의 member_id 사이드채널 단위테스트
  - get_current_member_id() 기본값(1) 확인
  - set_current_member_id() 이후 값 반영 확인
  - asyncio Task 간 격리(동시 요청에서 서로 다른 member_id 가 섞이지 않는지) 확인
    → rag_context 와 동일한 ContextVar 격리 보장이 member_id 에도 적용되는지가
      이 테스트의 핵심이다(여러 사용자가 동시에 챗봇을 쓰는 운영 상황의 핵심 안전장치).
"""
import asyncio

import pytest

from pipeline.pipeline_context import set_current_member_id, get_current_member_id


def test_default_is_one_when_never_set():
    # [설계 의도] set_current_member_id 를 호출하지 않는 경로(예: route_intent 직접 호출
    # 오프라인 테스트)는 기존과 동일하게 동작해야 하므로 기본값은 1(MOCK_MEMBER_ID)이다.
    # 단, 이 ContextVar 는 프로세스 전역이 아니라 Task 단위이므로, 다른 테스트가 같은
    # 이벤트루프 Task 안에서 먼저 set 을 호출했을 가능성을 막기 위해 새 Task 에서 검증한다.
    async def _run():
        assert get_current_member_id() == 1

    asyncio.run(_run())


def test_set_then_get_returns_same_value():
    async def _run():
        set_current_member_id(42)
        assert get_current_member_id() == 42

    asyncio.run(_run())


def test_concurrent_tasks_do_not_leak_member_id():
    # [핵심 검증] FastAPI 는 요청마다 새 asyncio Task 를 만든다. 동시에 두 사용자가
    # 요청을 보냈을 때 ContextVar 가 Task 간에 격리되지 않으면 한 사용자의 member_id 가
    # 다른 사용자의 주문조회 등에 새어 들어가는 심각한 보안 버그가 된다.
    results = {}

    async def _worker(name: str, member_id: int, delay: float):
        set_current_member_id(member_id)
        await asyncio.sleep(delay)  # 다른 Task 가 끼어들 틈을 의도적으로 만든다
        results[name] = get_current_member_id()

    async def _run():
        await asyncio.gather(
            _worker("user_a", 100, 0.02),
            _worker("user_b", 200, 0.01),
        )

    asyncio.run(_run())
    assert results == {"user_a": 100, "user_b": 200}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
