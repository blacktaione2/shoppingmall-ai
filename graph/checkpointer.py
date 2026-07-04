"""
graph/checkpointer.py
체크포인터(Redis / Memory) 생성 + 수명 관리.

[역할]
- REDIS_URL 유무에 따라 적절한 체크포인터를 만들어 builder 에 주입한다.
  · REDIS_URL 있음 → AsyncRedisSaver(연결 + 인덱스 셋업) → 영속화.
  · REDIS_URL 없음 → MemorySaver → 기존 인메모리 동작(로컬/테스트 회귀 없음).
- AsyncRedisSaver 는 내부적으로 비동기 컨텍스트 매니저(async with)로 살아있어야
  연결 풀이 유지된다. 그래서 'lifespan 이 살아있는 동안 컨텍스트를 연 채로 잡고,
  종료 시 닫는' 구조가 필요하다. 이 모듈이 그 컨텍스트의 진입/이탈을 담당한다.

[설계 메모]
- 왜 main.py 가 아니라 별도 모듈인가:
  lifespan 본문이 짧고 읽기 쉬워야 하고, Redis 의존성(import)을 'REDIS_URL 이
  있을 때만' 지연 import 하기 위해서다. langgraph-checkpoint-redis 를 설치하지
  않은 환경(텍스트 검색만 쓰는 서빙 인스턴스, CI 등)에서도 이 모듈 import 자체는
  실패하지 않아야 한다 → redis 관련 import 를 함수 안으로 넣는다.
- 폴백 정책:
  메모리 늘리는 방향(영속/안정 우선)으로 가되, REDIS_URL 자체가 비어 있으면
  '로컬 개발/테스트 의도'로 간주해 조용히 MemorySaver 를 쓴다.
  REDIS_URL 이 설정돼 있는데 연결에 실패하면 그것은 '운영 설정 오류'이므로
  예외를 그대로 올려 기동을 중단시킨다(조용한 데이터 유실 방지).
"""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _redis_url() -> str:
    """.env 의 REDIS_URL 을 읽는다(없거나 공백이면 빈 문자열)."""
    return (os.getenv("REDIS_URL") or "").strip()


def is_redis_enabled() -> bool:
    """REDIS_URL 이 설정돼 있으면 Redis 체크포인터를 쓸 의도로 본다."""
    return bool(_redis_url())


async def open_checkpointer():
    """체크포인터를 만들어 (checkpointer, aclose) 튜플로 반환한다.

    Returns:
        (checkpointer, aclose)
        · checkpointer : builder.set_checkpointer() 로 주입할 인스턴스.
        · aclose       : 앱 종료 시 호출할 비동기 정리 콜러블(없으면 no-op).

    [동작]
    - REDIS_URL 있음:
        AsyncRedisSaver 컨텍스트를 '열어둔 채' 내부 saver 를 꺼내 반환하고,
        aclose 가 그 컨텍스트를 닫는다. asetup() 으로 Redis 인덱스를 1회 생성한다.
        연결/셋업 실패 시 예외를 그대로 올린다(기동 중단 = 의도된 안전 동작).
    - REDIS_URL 없음:
        MemorySaver 를 반환하고 aclose 는 no-op.
    """
    url = _redis_url()

    if not url:
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("REDIS_URL 미설정 → MemorySaver 사용(인메모리, 재시작 시 소실)")

        async def _noop_close() -> None:
            return None

        return MemorySaver(), _noop_close

    # ── Redis 경로 ────────────────────────────────────────────────────
    # 지연 import: langgraph-checkpoint-redis 가 설치되지 않은 환경 보호.
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    logger.info("REDIS_URL 감지 → AsyncRedisSaver 사용(영속화)")

    # [개선 — 게스트 thread 누적 대비 TTL]
    #   게스트 요청은 매번 'guest-{uuid4}' 1회성 thread_id 로 invoke 되므로,
    #   Redis 영속화를 켜면 재조회되지 않는 thread 가 계속 쌓일 수 있다.
    #   langgraph-checkpoint-redis 0.5.0 의 TTL 적용은 from_conn_string 의
    #   ttl 인자 형식이 버전에 민감하므로(잘못된 키 → 기동 실패), 실제 Redis 를
    #   켜는 시점(REDIS_URL 설정)에 아래 형태로 검증 후 활성화할 것:
    #       cm = AsyncRedisSaver.from_conn_string(
    #               url, ttl={"default_ttl": 60, "refresh_on_read": True})
    #   (단위/키 이름은 설치된 패키지 버전의 docstring 으로 반드시 확인)
    #   현 시점 REDIS_URL 은 비어 있어 MemorySaver 폴백이라 이 경로는 미사용.
    # from_conn_string 은 async context manager 를 돌려준다.
    # lifespan 이 사는 동안 열어둬야 하므로 __aenter__/__aexit__ 를 수동 제어한다.
    cm = AsyncRedisSaver.from_conn_string(url)
    saver = await cm.__aenter__()

    try:
        # Redis 측 인덱스(체크포인트 저장 구조) 1회 생성. 멱등이라 재기동에도 안전.
        await saver.asetup()
    except Exception:
        # 셋업 실패 시 열어둔 컨텍스트를 정리하고 예외 전파(기동 중단).
        await cm.__aexit__(None, None, None)
        logger.exception("AsyncRedisSaver asetup() 실패 → 기동 중단")
        raise

    async def _aclose() -> None:
        try:
            await cm.__aexit__(None, None, None)
            logger.info("AsyncRedisSaver 컨텍스트 정리 완료")
        except Exception:
            logger.exception("AsyncRedisSaver 정리 중 예외(무시)")

    return saver, _aclose
