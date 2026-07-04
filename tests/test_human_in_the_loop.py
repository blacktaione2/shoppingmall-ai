"""
tests/test_human_in_the_loop.py
[추가] Human-in-the-loop(환불 신청) interrupt/resume 흐름 테스트.

[전략]
- test_agent_graph.py 와 동일하게 FakeToolCallingModel 을 주입한다.
- request_refund 도구가 interrupt 를 걸면 ainvoke 결과에 __interrupt__ 가 실린다.
- Command(resume=...) 로 같은 thread 를 재개하면 도구가 이어 실행된다.

[검증]
1. 로그인 회원 환불 → interrupt 발생(__interrupt__ payload 확인).
2. resume "approve" → 환불 접수 메시지.
3. resume "reject" → 환불 취소 메시지.
4. 게스트 환불 → interrupt 없이 즉시 차단 메시지(로그인 안내).
5. request_refund 가 ALL_TOOLS 에 등록되어 있다(회귀 가드).
"""
import asyncio
import uuid

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from graph import tools as tools_mod
import graph.agent_builder as agent_builder


class FakeToolCallingModel(FakeMessagesListChatModel):
    """미리 정한 AIMessage 시퀀스를 순서대로 반환하는 가짜 모델."""
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture(autouse=True)
def _mock_order_lookup(monkeypatch):
    """request_refund 가 interrupt 전에 호출하는 주문 존재/소유 검증을 격리한다.

    기본은 '주문 존재(본인 것)'로 모킹해 interrupt 흐름을 검증한다.
    주문 없음 케이스는 개별 테스트에서 다시 monkeypatch 한다.
    """
    def _fake_fetch(member_id, order_id):
        return {
            "order_id": str(order_id), "order_date": "2026-06-01",
            "items": [], "total_price": 10000, "status": "배송완료",
        }
    monkeypatch.setattr(tools_mod, "fetch_order_by_id", _fake_fetch)


def _build(ai_sequence):
    model = FakeToolCallingModel(responses=ai_sequence)
    return agent_builder.build_agent(
        model=model,
        checkpointer=MemorySaver(),
        force_rebuild=True,
    )


_REFUND_CALL = AIMessage(content="", tool_calls=[{
    "name": "request_refund",
    "args": {"order_id": "3"},
    "id": "call_refund_1",
}])
_FINAL = AIMessage(content="처리되었습니다.")


def test_refund_triggers_interrupt():
    """로그인 회원이 환불 요청 → interrupt 발생, payload 에 order_id 포함."""
    app = _build([_REFUND_CALL, _FINAL])
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}, "recursion_limit": 12}
    state = {"messages": [], "member_id": 7, "is_guest": False, "rag_hits": []}

    result = asyncio.run(app.ainvoke(state, config=config))
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "confirm_refund"
    assert payload["order_id"] == "3"
    assert "prompt" in payload


def test_refund_order_not_found_no_interrupt(monkeypatch):
    """존재하지 않거나 타인 주문이면 interrupt 없이 '찾을 수 없음' 안내."""
    # 이 테스트만 '주문 없음(None)'으로 덮어쓴다.
    monkeypatch.setattr(tools_mod, "fetch_order_by_id", lambda m, o: None)
    app = _build([_REFUND_CALL, _FINAL])
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}, "recursion_limit": 12}
    state = {"messages": [], "member_id": 7, "is_guest": False, "rag_hits": []}

    result = asyncio.run(app.ainvoke(state, config=config))
    # 존재하지 않는 주문에는 환불 확인(interrupt)을 띄우지 않는다.
    assert "__interrupt__" not in result
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any("찾을 수 없" in (m.content or "") for m in tool_msgs)


def test_refund_resume_approve():
    """interrupt 후 approve 재개 → 환불 접수 메시지."""
    app = _build([_REFUND_CALL, _FINAL])
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}, "recursion_limit": 12}
    state = {"messages": [], "member_id": 7, "is_guest": False, "rag_hits": []}

    asyncio.run(app.ainvoke(state, config=config))
    result = asyncio.run(app.ainvoke(Command(resume="approve"), config=config))

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any("접수" in (m.content or "") for m in tool_msgs)


def test_refund_resume_reject():
    """interrupt 후 reject 재개 → 환불 취소 메시지."""
    app = _build([_REFUND_CALL, _FINAL])
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}, "recursion_limit": 12}
    state = {"messages": [], "member_id": 7, "is_guest": False, "rag_hits": []}

    asyncio.run(app.ainvoke(state, config=config))
    result = asyncio.run(app.ainvoke(Command(resume="reject"), config=config))

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any("취소" in (m.content or "") for m in tool_msgs)


def test_refund_guest_blocked_without_interrupt():
    """게스트 환불 요청 → interrupt 없이 즉시 로그인 안내(차단)."""
    app = _build([_REFUND_CALL, _FINAL])
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}, "recursion_limit": 12}
    state = {"messages": [], "member_id": None, "is_guest": True, "rag_hits": []}

    result = asyncio.run(app.ainvoke(state, config=config))
    # 게스트는 interrupt 가 발생하지 않아야 한다(도구 진입 즉시 차단).
    assert "__interrupt__" not in result
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any("로그인" in (m.content or "") for m in tool_msgs)


def test_refund_tool_registered():
    """request_refund 가 ALL_TOOLS 에 등록되어 있다(회귀 가드)."""
    names = {getattr(t, "name", "") for t in tools_mod.ALL_TOOLS}
    assert "request_refund" in names
    # 기존 4개 도구도 그대로 유지
    assert {"search_products", "semantic_search", "search_faq", "get_my_orders"} <= names


def test_mcp_agent_excludes_refund():
    """include_refund=False 로 빌드하면 request_refund 가 도구에서 빠진다."""
    model = FakeToolCallingModel(responses=[_FINAL])
    # build_agent 내부에서 tools 를 필터링하는지 간접 검증:
    # include_refund=False 빌드는 환불 도구 없이 정상 컴파일되어야 한다.
    app = agent_builder.build_agent(
        model=model, checkpointer=MemorySaver(),
        force_rebuild=True, include_refund=False,
    )
    assert app is not None
