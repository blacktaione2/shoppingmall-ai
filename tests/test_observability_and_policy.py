"""
tests/test_observability_and_policy.py
LangSmith 관측성 + 동적 모델 선택 테스트.

[검증]
1. 동적 모델 선택 정책:
   - OFF(기본): SIMPLE/COMPLEX 모두 gpt-5.4
   - ON: SIMPLE → gpt-5.4-mini, COMPLEX → gpt-5.4
   - 알 수 없는 값 → gpt-5.4 폴백
2. observability:
   - route_metadata 구조
   - is_tracing_enabled 토글
3. small_talk_node 가 동적 선택을 사용(ON 시 mini, OFF 시 gpt-5.4)
4. @traceable 부착 함수가 트레이싱 OFF 에서도 정상 동작(no-op)
"""
import os

import pytest


def _model_name(llm):
    """ChatOpenAI(바인딩) 인스턴스에서 모델명을 추출."""
    if hasattr(llm, "bound") and hasattr(llm.bound, "model_name"):
        return llm.bound.model_name
    if hasattr(llm, "model_name"):
        return llm.model_name
    return None


# ────────────────────────────────────────────────────────────────────────
# 1) 동적 모델 선택 정책
# ────────────────────────────────────────────────────────────────────────
def test_policy_off_always_gpt4o(monkeypatch):
    monkeypatch.delenv("DYNAMIC_MODEL_SELECTION", raising=False)
    from graph.model_policy import select_llm, TaskComplexity
    assert _model_name(select_llm(TaskComplexity.SIMPLE)) == "gpt-5.4"
    assert _model_name(select_llm(TaskComplexity.COMPLEX)) == "gpt-5.4"


def test_policy_on_simple_is_mini(monkeypatch):
    monkeypatch.setenv("DYNAMIC_MODEL_SELECTION", "true")
    from graph.model_policy import select_llm, TaskComplexity
    assert _model_name(select_llm(TaskComplexity.SIMPLE)) == "gpt-5.4-mini"
    assert _model_name(select_llm(TaskComplexity.COMPLEX)) == "gpt-5.4"


def test_policy_toggle_runtime(monkeypatch):
    """런타임에 환경변수를 바꾸면 즉시 반영된다(매 호출 재평가)."""
    from graph.model_policy import select_llm, TaskComplexity
    monkeypatch.setenv("DYNAMIC_MODEL_SELECTION", "false")
    assert _model_name(select_llm(TaskComplexity.SIMPLE)) == "gpt-5.4"
    monkeypatch.setenv("DYNAMIC_MODEL_SELECTION", "true")
    assert _model_name(select_llm(TaskComplexity.SIMPLE)) == "gpt-5.4-mini"


# ────────────────────────────────────────────────────────────────────────
# 2) observability
# ────────────────────────────────────────────────────────────────────────
def test_route_metadata_structure():
    from graph.observability import route_metadata
    md = route_metadata("single_agent", is_guest=True)
    assert md["metadata"]["route"] == "single_agent"
    assert md["metadata"]["is_guest"] is True
    assert "single_agent" in md["tags"]


def test_tracing_toggle(monkeypatch):
    from graph.observability import is_tracing_enabled
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert is_tracing_enabled() is False
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert is_tracing_enabled() is True


# ────────────────────────────────────────────────────────────────────────
# 3) small_talk_node 가 동적 선택을 사용
# ────────────────────────────────────────────────────────────────────────
def test_small_talk_uses_dynamic_model(monkeypatch):
    """small_talk_node 가 select_llm(SIMPLE) 을 호출하는지 확인.

    select_llm 을 가짜로 치환해 호출 인자(complexity)를 캡처한다.
    """
    import asyncio
    from langchain_core.runnables import RunnableLambda
    from graph import nodes
    from graph.model_policy import TaskComplexity
    from schemas.intent_schema import IntentResult, IntentType

    captured = {}

    def fake_select_llm(complexity, temperature=0.7):
        captured["complexity"] = complexity
        # 실제 Runnable 을 반환해야 prompt | llm | parser 체인이 성립한다.
        # LLM 자리에는 "AI 메시지 역할"의 고정 문자열을 내는 Runnable 을 둔다.
        return RunnableLambda(lambda _messages: "안녕하세요!")

    monkeypatch.setattr(nodes, "select_llm", fake_select_llm, raising=True)

    state = {
        "question": "안녕",
        "history": [],
        "intent_result": IntentResult(intent=IntentType.SMALL_TALK, confidence=0.9),
    }
    out = asyncio.run(nodes.small_talk_node(state))
    assert captured["complexity"] == TaskComplexity.SIMPLE
    assert "안녕하세요" in out["raw_answer"]


# ────────────────────────────────────────────────────────────────────────
# 4) @traceable 부착 함수가 트레이싱 OFF 에서 정상 동작
# ────────────────────────────────────────────────────────────────────────
def test_traceable_noop_when_off(monkeypatch):
    """LANGSMITH_TRACING off 상태에서 traceable 함수가 그대로 실행되는지.

    실제 OpenAI 호출은 막으므로, '데코레이터가 호출을 가로채 막지 않는지'만 본다.
    clean_text_for_tts 는 외부호출이 없어 순수 검증에 적합하지만 traceable 이
    아니므로, synthesize 대신 가벼운 경로로 데코레이터 통과만 확인한다.
    """
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    from services import gpt_service
    # chat_completion 은 traceable 로 감싸였지만 호출 가능한 함수여야 한다.
    assert callable(gpt_service.chat_completion)
    assert callable(gpt_service.structured_completion)
    # 내부 OpenAI 호출 직전까지 도달하는지: get_client 를 막아 RuntimeError 유도 →
    # 데코레이터가 함수 본문을 정상 실행했다는 증거.
    import asyncio
    monkeypatch.setattr(gpt_service, "get_client",
                        lambda: (_ for _ in ()).throw(RuntimeError("client blocked")),
                        raising=True)
    with pytest.raises(RuntimeError, match="client blocked"):
        asyncio.run(gpt_service.chat_completion("sys", "user"))
