"""
tests/test_slack_notification.py
[추가] Human-in-the-loop 환불 알림의 Slack MCP 채널 검증.

[검증]
1. SLACK_REFUND_CHANNEL_ID 미설정 → 발송 생략(MCP 호출 없음)
2. MCP_ENABLED=false → 발송 생략
3. MCP 도구 목록에 conversations_add_message 없음 → 생략(예외 없음)
4. 정상 로드 → 도구가 channel_id/payload 로 호출됨
5. 도구 호출 예외 → 삼켜짐(호출자에게 전파 안 됨)
"""
import asyncio

import pytest

from services import notification_service


def _reset(monkeypatch):
    from graph import mcp_tools
    mcp_tools.reset_cache()


def test_skips_without_channel_id(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setenv("SLACK_REFUND_CHANNEL_ID", "")
    monkeypatch.setenv("MCP_ENABLED", "true")

    called = {}
    async def _fake_get_tools():
        called["hit"] = True
        return []
    monkeypatch.setattr("graph.mcp_tools.get_mcp_tools", _fake_get_tools)

    asyncio.run(notification_service.send_refund_admin_slack("3", 7, "단순 변심"))
    assert "hit" not in called  # 채널ID 없으면 MCP 조회 자체를 안 함


def test_skips_when_mcp_disabled(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setenv("SLACK_REFUND_CHANNEL_ID", "C0123456789")
    monkeypatch.setenv("MCP_ENABLED", "false")

    called = {}
    async def _fake_get_tools():
        called["hit"] = True
        return []
    monkeypatch.setattr("graph.mcp_tools.get_mcp_tools", _fake_get_tools)

    asyncio.run(notification_service.send_refund_admin_slack("3", 7, None))
    assert "hit" not in called


def test_skips_when_tool_not_found(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setenv("SLACK_REFUND_CHANNEL_ID", "C0123456789")
    monkeypatch.setenv("MCP_ENABLED", "true")

    async def _fake_get_tools():
        return []  # conversations_add_message 없음
    monkeypatch.setattr("graph.mcp_tools.get_mcp_tools", _fake_get_tools)

    # 예외 없이 조용히 리턴되어야 한다.
    asyncio.run(notification_service.send_refund_admin_slack("3", 7, None))


def test_invokes_tool_with_channel_and_payload(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setenv("SLACK_REFUND_CHANNEL_ID", "C0123456789")
    monkeypatch.setenv("MCP_ENABLED", "true")

    captured = {}

    class FakeSlackTool:
        name = "conversations_add_message"
        async def ainvoke(self, args):
            captured.update(args)
            return "ok"

    async def _fake_get_tools():
        return [FakeSlackTool()]
    monkeypatch.setattr("graph.mcp_tools.get_mcp_tools", _fake_get_tools)

    asyncio.run(notification_service.send_refund_admin_slack("3", 7, "단순 변심"))

    assert captured["channel_id"] == "C0123456789"
    assert "3" in captured["payload"]
    assert "단순 변심" in captured["payload"]


def test_tool_exception_is_swallowed(monkeypatch):
    _reset(monkeypatch)
    monkeypatch.setenv("SLACK_REFUND_CHANNEL_ID", "C0123456789")
    monkeypatch.setenv("MCP_ENABLED", "true")

    class BoomTool:
        name = "conversations_add_message"
        async def ainvoke(self, args):
            raise RuntimeError("slack 서버 연결 실패")

    async def _fake_get_tools():
        return [BoomTool()]
    monkeypatch.setattr("graph.mcp_tools.get_mcp_tools", _fake_get_tools)

    # 예외가 밖으로 전파되지 않아야 한다(best-effort).
    asyncio.run(notification_service.send_refund_admin_slack("3", 7, None))
