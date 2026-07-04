"""
tests/test_mcp_integration.py
MCP 연동 테스트.

[전략]
- 실제 MCP 서버 없이 graph.mcp_tools 의 동작을 검증한다.
- get_mcp_tools 의 폴백(비활성/미설정/실패)을 중점 검증.
- build_agent 가 extra_tools(MCP 도구)를 로컬 도구와 병합하는지 검증.

[검증]
1. MCP_ENABLED=false → 빈 도구 리스트
2. 설정 파일 없음 → 빈 리스트 (활성화돼도)
3. MCP 클라이언트 예외 → 빈 리스트 폴백
4. 정상 로드 → 도구 리스트 반환 (mock)
5. build_agent extra_tools 병합 → 로컬+MCP 도구 모두 바인딩
"""
import asyncio

import pytest

from graph import mcp_tools


def _reset(monkeypatch):
    mcp_tools.reset_cache()


# ────────────────────────────────────────────────────────────────────────
# 1) MCP 비활성 → 빈 리스트
# ────────────────────────────────────────────────────────────────────────
def test_mcp_disabled_returns_empty(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setenv("MCP_ENABLED", "false")
    out = asyncio.run(mcp_tools.get_mcp_tools())
    assert out == []


# ────────────────────────────────────────────────────────────────────────
# 2) 활성화됐지만 설정 파일 없음 → 빈 리스트
# ────────────────────────────────────────────────────────────────────────
def test_mcp_no_config_returns_empty(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setattr(mcp_tools, "_load_config", lambda: {}, raising=True)
    out = asyncio.run(mcp_tools.get_mcp_tools())
    assert out == []


# ────────────────────────────────────────────────────────────────────────
# 3) MCP 클라이언트 예외 → 빈 리스트 폴백
# ────────────────────────────────────────────────────────────────────────
def test_mcp_client_failure_falls_back(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setattr(mcp_tools, "_load_config",
                        lambda: {"weather": {"url": "x", "transport": "sse"}}, raising=True)

    # MultiServerMCPClient 를 예외 발생하도록 치환
    import langchain_mcp_adapters.client as mcp_client_mod
    class BoomClient:
        def __init__(self, *a, **k):
            raise RuntimeError("MCP 서버 연결 실패")
    monkeypatch.setattr(mcp_client_mod, "MultiServerMCPClient", BoomClient, raising=True)

    out = asyncio.run(mcp_tools.get_mcp_tools())
    assert out == []   # 폴백


# ────────────────────────────────────────────────────────────────────────
# 4) 정상 로드 → 도구 리스트 (mock)
# ────────────────────────────────────────────────────────────────────────
def test_mcp_loads_tools(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setattr(mcp_tools, "_load_config",
                        lambda: {"weather": {"url": "x", "transport": "sse"}}, raising=True)

    fake_tools = ["TOOL_A", "TOOL_B"]   # 실제로는 BaseTool, 여기선 식별만
    import langchain_mcp_adapters.client as mcp_client_mod
    class FakeClient:
        def __init__(self, *a, **k):
            pass
        async def get_tools(self):
            return fake_tools
    monkeypatch.setattr(mcp_client_mod, "MultiServerMCPClient", FakeClient, raising=True)

    out = asyncio.run(mcp_tools.get_mcp_tools())
    assert out == fake_tools


# ────────────────────────────────────────────────────────────────────────
# 5) build_agent 가 extra_tools 를 병합하는지
# ────────────────────────────────────────────────────────────────────────
def test_build_agent_merges_extra_tools(monkeypatch):
    import graph.agent_builder as ab
    from graph.tools import ALL_TOOLS
    from langchain_core.tools import tool

    @tool
    def fake_mcp_tool(x: str) -> str:
        """가짜 MCP 도구."""
        return x

    captured = {}
    # create_react_agent 를 가짜로 대체: tools 만 캡처하고 더미 그래프 노드 반환.
    # (도구 병합 여부 검증이 목적이라 실제 ReAct 그래프는 불필요)
    # **kwargs 로 pre_model_hook 등 부가 인자를 흡수해 실제 시그니처 변화에 견고하게.
    def fake_create(model, tools, state_schema, prompt, **kwargs):
        captured["tools"] = tools
        async def _dummy_node(state):
            return {}
        return _dummy_node
    monkeypatch.setattr(ab, "create_react_agent", fake_create, raising=True)

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langgraph.checkpoint.memory import MemorySaver
    model = FakeMessagesListChatModel(responses=[])

    ab.build_agent(model=model, checkpointer=MemorySaver(),
                   force_rebuild=True, extra_tools=[fake_mcp_tool])

    tool_names = [t.name for t in captured["tools"]]
    # 로컬 도구 + MCP 도구가 모두 포함
    assert "fake_mcp_tool" in tool_names
    assert len(captured["tools"]) == len(ALL_TOOLS) + 1
