"""
services/idempotency.py 단위 테스트 — Redis 없이 오프라인 검증.

검증 항목:
1) 최초 request_id 는 통과(True), 같은 ID 재시도는 차단(False)
2) 서로 다른 ID 는 각각 통과 (의미 기반이 아닌 요청 기반 — 같은 질문 재전송 오차단 없음)
3) request_id 미전송(None/빈값)은 하위호환으로 항상 통과
4) Redis 미설정이면 항상 통과 (best-effort 폴백)
5) Redis 오류 시에도 통과 (가용성 우선)
6) SETNX 호출 계약: NX + TTL(EX=300)
(프로젝트 관례에 따라 pytest-asyncio 없이 asyncio.run 으로 실행)
"""
import asyncio

import pytest

from services import idempotency


class FakeRedis:
    """redis.asyncio 의 set(nx=..., ex=...) 만 흉내내는 대역."""

    def __init__(self):
        self.store = {}
        self.last_kwargs = None

    async def set(self, key, value, nx=False, ex=None):
        self.last_kwargs = {"nx": nx, "ex": ex}
        if nx and key in self.store:
            return None  # redis-py: NX 실패 시 None
        self.store[key] = value
        return True


class BrokenRedis:
    async def set(self, *args, **kwargs):
        raise ConnectionError("redis down")


@pytest.fixture(autouse=True)
def _reset_module_state():
    """모듈 전역(lazy 싱글톤)을 테스트마다 초기화."""
    idempotency._client = None
    idempotency._failed = False
    yield
    idempotency._client = None
    idempotency._failed = False


def test_first_acquire_passes_and_duplicate_blocked():
    idempotency._client = FakeRedis()
    assert asyncio.run(idempotency.try_acquire("req-uuid-1")) is True
    assert asyncio.run(idempotency.try_acquire("req-uuid-1")) is False  # 중복 차단


def test_different_ids_pass_independently():
    idempotency._client = FakeRedis()
    assert asyncio.run(idempotency.try_acquire("req-uuid-1")) is True
    assert asyncio.run(idempotency.try_acquire("req-uuid-2")) is True  # 새 요청은 통과


def test_missing_request_id_passes_for_backward_compat():
    idempotency._client = FakeRedis()
    assert asyncio.run(idempotency.try_acquire(None)) is True
    assert asyncio.run(idempotency.try_acquire("")) is True


def test_no_redis_configured_passes(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert asyncio.run(idempotency.try_acquire("req-uuid-1")) is True
    # 미설정은 1회 확인 후 _failed 로 고정돼 재시도 비용이 없다
    assert idempotency._failed is True


def test_redis_error_passes_best_effort():
    idempotency._client = BrokenRedis()
    assert asyncio.run(idempotency.try_acquire("req-uuid-1")) is True


def test_setnx_contract_nx_and_ttl():
    fake = FakeRedis()
    idempotency._client = fake
    asyncio.run(idempotency.try_acquire("req-uuid-1"))
    assert fake.last_kwargs == {"nx": True, "ex": idempotency.IDEM_TTL_SECONDS}
    assert "idem:req-uuid-1" in fake.store
