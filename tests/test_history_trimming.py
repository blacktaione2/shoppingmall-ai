"""
tests/test_history_trimming.py
멀티턴 이력 트리밍 테스트 (컨텍스트 윈도우 초과 방지).

[배경]
Checkpointer 에 누적된 messages 는 대화가 길어질수록 무한정 늘어난다. 이를 그대로
LLM 에 넘기면 컨텍스트 윈도우를 초과할 수 있어, LangChain trim_messages 로 'LLM 입력만'
최근 N개로 잘라낸다(원본은 보존). graph.llm 의 트리밍 헬퍼를 직접 검증한다.

[검증]
1. Agent 경로(BaseMessage): 긴 대화가 HISTORY_MAX_MESSAGES 기준으로 잘린다.
2. tool_call 쌍 보존: AIMessage(tool_calls)와 ToolMessage 가 분리되지 않는다.
3. pre_model_hook: llm_input_messages 만 반환해 원본 messages 를 건드리지 않는다.
4. 라우터 경로(history dict): 최근 N개로 잘리고 마지막(최신) 항목이 보존된다.
5. 안전장치: 캡이 한 턴 분량보다 작아도 '이번 질문'은 절대 누락되지 않는다.
"""
import importlib

from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, SystemMessage,
)

import graph.llm as llm_mod


def _reload_with_cap(monkeypatch, cap: str):
    """HISTORY_MAX_MESSAGES 를 설정하고 graph.llm 을 재로드해 캡을 반영한다."""
    monkeypatch.setenv("HISTORY_MAX_MESSAGES", cap)
    importlib.reload(llm_mod)
    return llm_mod


def _make_tool_dialog(turns: int) -> list:
    """질문 → AI(tool_call) → ToolMessage → AI(최종답변) 패턴을 turns 번 반복."""
    msgs = [SystemMessage(content="시스템 프롬프트")]
    for i in range(turns):
        msgs.append(HumanMessage(content=f"질문{i}", id=f"h{i}"))
        msgs.append(AIMessage(
            content="", id=f"ai_c{i}",
            tool_calls=[{"name": "search_products", "args": {}, "id": f"c{i}"}],
        ))
        msgs.append(ToolMessage(content=f"결과{i}", tool_call_id=f"c{i}", id=f"tm{i}"))
        msgs.append(AIMessage(content=f"답변{i}", id=f"ai_f{i}"))
    return msgs


def test_agent_message_list_is_trimmed(monkeypatch):
    """긴 대화가 캡 기준으로 잘린다(원본보다 짧아진다)."""
    m = _reload_with_cap(monkeypatch, "8")
    msgs = _make_tool_dialog(turns=6)   # 1 + 6*4 = 25개
    trimmed = m.trim_message_list(msgs)
    assert len(trimmed) < len(msgs)
    # 시스템 프롬프트는 항상 유지
    assert any(isinstance(x, SystemMessage) for x in trimmed)


def test_tool_call_pairs_are_not_broken(monkeypatch):
    """트림 후에도 모든 tool_call 에 대응하는 ToolMessage 가 존재한다."""
    m = _reload_with_cap(monkeypatch, "8")
    msgs = _make_tool_dialog(turns=6)
    trimmed = m.trim_message_list(msgs)

    call_ids = {tc["id"] for x in trimmed if isinstance(x, AIMessage) for tc in x.tool_calls}
    result_ids = {x.tool_call_id for x in trimmed if isinstance(x, ToolMessage)}
    # tool_call 은 있는데 결과가 없는(쌍이 깨진) id 가 없어야 한다
    assert call_ids - result_ids == set()


def test_pre_model_hook_preserves_original(monkeypatch):
    """pre_model_hook 은 llm_input_messages 만 반환(원본 messages 비파괴)."""
    m = _reload_with_cap(monkeypatch, "8")
    msgs = _make_tool_dialog(turns=6)
    out = m.agent_pre_model_hook({"messages": msgs})
    assert "llm_input_messages" in out
    assert "messages" not in out   # 원본 messages 키는 갱신하지 않음
    assert len(out["llm_input_messages"]) <= len(msgs)


def test_router_history_is_trimmed(monkeypatch):
    """history dict 도 최근 N개로 잘리고, 가장 최신 항목이 보존된다."""
    m = _reload_with_cap(monkeypatch, "8")
    hist = [
        {"role": "user" if i % 2 == 0 else "bot", "text": f"메시지{i}"}
        for i in range(20)
    ]
    trimmed = m.trim_history(hist)
    assert len(trimmed) <= 8
    assert trimmed[-1] == hist[-1]   # 최신 대화 보존


def test_current_question_never_dropped(monkeypatch):
    """안전장치: 캡이 한 턴 분량보다 작아도 '이번 질문'은 누락되지 않는다."""
    m = _reload_with_cap(monkeypatch, "2")   # 한 턴(4개)보다 작은 캡
    msgs = _make_tool_dialog(turns=3)
    trimmed = m.trim_message_list(msgs)
    # 원본의 마지막 HumanMessage 가 트림 결과에 반드시 포함되어야 한다
    last_human = next(x for x in reversed(msgs) if isinstance(x, HumanMessage))
    assert last_human in trimmed


def test_empty_inputs_are_safe(monkeypatch):
    """빈 입력은 그대로 빈 결과를 돌려준다(예외 없음)."""
    m = _reload_with_cap(monkeypatch, "8")
    assert m.trim_message_list([]) == []
    assert m.trim_history([]) == []
