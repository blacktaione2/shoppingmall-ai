"""
graph/observability.py
LangSmith 모니터링 설정

[동작]
- LangChain/LangGraph 는 아래 환경변수가 있으면 모든 LLM/노드 실행을 자동 트레이싱한다:
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=ls__...
    LANGSMITH_PROJECT=shoppingmall-ai
- 이 모듈은:
  1) 앱 시작 시 활성화 여부를 로깅(키 노출 없이)해 운영자가 확인 가능하게 한다.
  2) 각 처리 경로(라우터/단일Agent/멀티Agent)에 붙일 metadata 태그를 제공한다.
     → LangSmith UI 에서 'route' 별 필터링/비용 비교가 가능해진다.

[안전]
- LANGSMITH_TRACING 이 false/미설정이면 트레이싱은 완전 no-op.
  본 파이프라인 동작에는 어떤 영향도 주지 않는다(키 오류 시에도 LangChain 이
  트레이싱만 silent 실패시키고 응답 생성은 계속된다).
"""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def is_tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "false").strip().lower() in (
        "true", "1", "yes",
    )


def init_observability() -> None:
    """앱 시작 시 호출. LangSmith 활성화 상태를 로깅한다(키는 노출하지 않음)."""
    if is_tracing_enabled():
        project = os.getenv("LANGSMITH_PROJECT", "default")
        has_key = bool(os.getenv("LANGSMITH_API_KEY"))
        logger.info(
            "LangSmith 트레이싱 활성화됨 (project=%s, api_key=%s)",
            project, "설정됨" if has_key else "없음(⚠️ 키 누락)",
        )
        if not has_key:
            logger.warning(
                "LANGSMITH_TRACING=true 이지만 LANGSMITH_API_KEY 가 없습니다. "
                "트레이스가 전송되지 않습니다."
            )
    else:
        logger.info("LangSmith 트레이싱 비활성화 (LANGSMITH_TRACING != true)")


def route_metadata(route: str, **extra) -> dict:
    """invoke config 에 실을 LangSmith metadata 를 만든다.

    Args:
        route: 처리 경로 식별자 ('router_pipeline'|'single_agent'|'multi_agent'|'mcp_agent').
        extra: 추가 태그(예: is_guest, intent 등).
    Returns:
        {"metadata": {...}, "tags": [...]} 형태 — invoke config 에 병합해 사용.

    [사용 예]
        config = {"configurable": {"thread_id": tid}, **route_metadata("single_agent")}
    """
    metadata = {"route": route, **extra}
    return {"metadata": metadata, "tags": [route]}
