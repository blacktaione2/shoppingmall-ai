"""
graph/mcp_tools.py
MCP(Model Context Protocol) 외부 서버 도구 로드 어댑터.

[목적]
- 외부 MCP 서버가 제공하는 도구를 langchain-mcp-adapters 로 LangChain Tool 로
  변환해, 우리 Agent 에 로컬 도구와 함께 바인딩한다.
  · 예: 날씨 MCP(계절/날씨 기반 추천 보강), 웹검색 MCP(트렌드/리뷰) 등.

[안전장치 — 가장 중요]
- MCP_ENABLED=false(기본) → MCP 자체를 건너뛰고 빈 도구 리스트 반환.
  (기존 Agent 동작과 동일, 외부 의존 0)
- 서버 연결 실패/패키지 미설치/설정 오류 → '빈 리스트' 폴백.
  MCP 도구가 없어도 Agent 는 로컬 도구만으로 정상 동작한다(전체가 죽지 않음).
- lazy import: langchain-mcp-adapters 는 MCP 활성화 시에만 import.
- 서버 설정은 mcp_config.json(JSON)에서 읽는다. 신뢰할 수 있는 서버만 등록할 것.

[로드 캐시]
- get_mcp_tools() 는 1회 로드 후 결과를 캐시한다(서버 재연결 비용 절감).
  앱 시작 시 prefetch_mcp_tools() 로 미리 로드해 두는 것을 권장.

[설정 파일 예시] mcp_config.json
{
  "weather": {
    "command": "python",
    "args": ["-m", "mcp_server_weather"],
    "transport": "stdio"
  },
  "search": {
    "url": "https://mcp.example.com/sse",
    "transport": "sse"
  }
}
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

_MCP_CONFIG_PATH = os.getenv("MCP_CONFIG_PATH", "mcp_config.json")

# 로드 결과 캐시 (None = 아직 로드 안 함)
_cached_tools: list | None = None


def is_mcp_enabled() -> bool:
    """MCP 활성화 여부 (환경변수, 기본 off)."""
    return os.getenv("MCP_ENABLED", "false").lower() == "true"


def _load_config() -> dict:
    """mcp_config.json 을 읽어 connections dict 반환. 없으면 빈 dict."""
    path = _MCP_CONFIG_PATH
    if not os.path.exists(path):
        logger.info("MCP 설정 파일 없음(%s) → MCP 도구 로드 건너뜀", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if not isinstance(config, dict):
            logger.warning("MCP 설정이 dict 형식이 아님 → 무시")
            return {}
        return config
    except Exception:
        logger.exception("MCP 설정 파일 파싱 실패 → MCP 도구 로드 건너뜀")
        return {}


async def get_mcp_tools(force_reload: bool = False) -> list:
    """MCP 서버 도구를 LangChain Tool 리스트로 반환한다(캐시).

    Returns:
        LangChain BaseTool 리스트. 비활성/실패/미설정 시 빈 리스트.

    [폴백 정책]
        - MCP_ENABLED=false        → []
        - 설정 파일 없음/파싱 실패  → []
        - 패키지 미설치             → [] (안내 로그)
        - 서버 연결 실패            → [] (경고 로그)
    """
    global _cached_tools

    if not is_mcp_enabled():
        return []

    if _cached_tools is not None and not force_reload:
        return _cached_tools

    connections = _load_config()
    if not connections:
        _cached_tools = []
        return _cached_tools

    # lazy import: MCP 활성화 시에만 어댑터 로드
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.warning(
            "langchain-mcp-adapters 미설치 → MCP 도구 건너뜀. "
            "pip install langchain-mcp-adapters"
        )
        _cached_tools = []
        return _cached_tools

    try:
        client = MultiServerMCPClient(connections=connections)
        tools = await client.get_tools()
        logger.info("MCP 도구 %d개 로드됨 (서버 %d개)", len(tools), len(connections))
        _cached_tools = tools
        return _cached_tools
    except Exception:
        logger.exception("MCP 서버 연결/도구 로드 실패 → 빈 리스트 폴백")
        _cached_tools = []
        return _cached_tools


async def prefetch_mcp_tools() -> None:
    """앱 시작 시 MCP 도구를 미리 로드해 캐시한다(요청 지연 방지)."""
    if is_mcp_enabled():
        await get_mcp_tools()


def reset_cache() -> None:
    """테스트용: 캐시 초기화."""
    global _cached_tools
    _cached_tools = None
