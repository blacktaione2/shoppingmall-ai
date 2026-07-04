"""
요청 멱등성(idempotency) 체크 — Redis SETNX 기반 중복 요청 차단.

[역할 분리 — 세션 토큰 vs 멱등키]
- chat_token (Spring 발급 UUID) : "누가" 보냈는지 — 세션 식별자라 모든 메시지에서 동일.
- request_id (클라이언트 발급 UUID): "이 요청이 처음인지" — 전송마다 crypto.randomUUID().
  chat_token 을 멱등키로 쓰면 첫 메시지 이후 모든 대화가 중복으로 차단되므로
  요청 단위 UUID 를 별도로 받는다.

[동작 — best-effort]
- SET idem:{request_id} NX EX 300 성공 → 최초 요청, 통과.
- 실패(키 존재)               → 중복 요청, 차단 (호출부가 409 처리).
- Redis 미설정/미설치/다운     → 항상 통과 — 체크포인터(graph/checkpointer.py)와
  동일한 폴백 철학: 멱등성 보장보다 챗봇 가용성이 우선이다.

[TTL 5분 근거]
응답 완료 후에도 같은 ID 재전송(네트워크 재시도, 프록시 중복)을 계속 차단하면서,
UUID 키 1개당 수십 바이트라 Redis 메모리 부담은 무시할 수준.

[적용 범위]
대표 경로 /chat/ask, /chat/stream 만 적용 (routers/chat.py 진입부 1곳).
/voice, /agent 계열은 동일 패턴으로 확장 가능하나 범위를 최소로 유지한다.
"""
import logging
import os

logger = logging.getLogger(__name__)

IDEM_TTL_SECONDS = 300
_KEY_PREFIX = "idem:"

# lazy 싱글톤 — REDIS_URL 이 없거나 초기화에 실패하면 다시 시도하지 않는다(_failed).
_client = None
_failed = False


def _get_client():
    """redis.asyncio 클라이언트 lazy 초기화. 미설정/미설치 시 None (통과 모드)."""
    global _client, _failed
    if _client is not None or _failed:
        return _client
    url = (os.getenv("REDIS_URL") or "").strip()
    if not url:
        _failed = True
        return None
    try:
        # 체크포인터와 동일하게 지연 import — redis 미설치 환경(로컬/CI)에서도
        # 모듈 로드가 실패하지 않아야 한다.
        import redis.asyncio as aioredis
        _client = aioredis.from_url(url, decode_responses=True)
    except Exception as exc:  # noqa: BLE001 — 어떤 실패든 통과 모드로 강등
        logger.warning("멱등성 Redis 초기화 실패 — 중복 차단 없이 통과 모드로 동작: %s", exc)
        _failed = True
    return _client


async def try_acquire(request_id: str | None) -> bool:
    """request_id 선점을 시도한다. True=최초(통과) / False=중복(차단).

    request_id 미전송(None/빈값)은 구버전 클라이언트·외부 호출 하위호환으로 항상 통과.
    Redis 계열 오류도 통과 처리(best-effort) — 본 응답 흐름을 절대 막지 않는다.
    """
    if not request_id:
        return True
    client = _get_client()
    if client is None:
        return True
    try:
        acquired = await client.set(
            _KEY_PREFIX + request_id, "1", nx=True, ex=IDEM_TTL_SECONDS,
        )
        return bool(acquired)
    except Exception as exc:  # noqa: BLE001
        logger.warning("멱등성 체크 실패 — 통과 처리: request_id=%s, cause=%s", request_id, exc)
        return True
