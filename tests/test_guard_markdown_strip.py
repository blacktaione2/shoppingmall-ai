"""
tests/test_guard_markdown_strip.py
guard_node 의 마크다운 굵게(**) 강제 제거 오프라인 단위테스트.

[배경] SYSTEM_PROMPT 로 "굵게(**) 쓰지 마라"고 지시해도 LLM 이 종종 그대로
써서, 모든 인텐트 답변이 합류하는 guard_node 에서 강제로 벗기도록 했다.
guard_answer_state(실제 환각 가드 로직)는 monkeypatch 로 격리한다(오프라인).
"""
import asyncio

from graph import nodes


def test_guard_node_strips_markdown_bold(monkeypatch):
    async def fake_guard_answer_state(question, answer, intent_result, rag_hits, history):
        return answer   # 가드는 통과시키고 원문 그대로 반환한다고 가정

    monkeypatch.setattr("graph.guard.guard_answer_state", fake_guard_answer_state, raising=True)

    state = {
        "question": "겨울 옷 추천해줘",
        "raw_answer": "**오버핏 양털 후리스**(98,000원)를 추천드려요.",
        "intent_result": {"intent": "SEMANTIC_SEARCH"},
        "rag_hits": [],
        "history": [],
    }
    result = asyncio.run(nodes.guard_node(state))

    assert result["final_answer"] == "오버핏 양털 후리스(98,000원)를 추천드려요."
    assert "**" not in result["final_answer"]


def test_guard_node_leaves_plain_text_unaffected(monkeypatch):
    """굵게 기호가 없는 일반 답변은 내용이 그대로 유지돼야 한다(회귀 방지)."""
    async def fake_guard_answer_state(question, answer, intent_result, rag_hits, history):
        return answer

    monkeypatch.setattr("graph.guard.guard_answer_state", fake_guard_answer_state, raising=True)

    state = {
        "question": "안녕",
        "raw_answer": "안녕하세요! 무엇을 도와드릴까요?",
        "intent_result": {"intent": "SMALL_TALK"},
        "rag_hits": [],
        "history": [],
    }
    result = asyncio.run(nodes.guard_node(state))

    assert result["final_answer"] == "안녕하세요! 무엇을 도와드릴까요?"