"""
tests/test_multiturn_memory.py
멀티턴 하이브리드(checkpointer) 통합 테스트.

[전략]
- classify_node + 핸들러 노드를 monkeypatch 로 고정/관찰.
- 핵심은 "노드가 받은 state['history'] 가 무엇인가"를 캡처해서,
  서버 메모리(checkpointer) 누적분이 다음 턴에 제대로 주입되는지 확인하는 것.

[검증 시나리오]
1. 로그인 2턴: 2턴째 노드의 history 에 1턴 질문/답변이 들어있다 (서버 메모리 누적).
2. 게스트 2턴: config 없이 invoke 되어 서버 메모리에 누적되지 않는다
   (2턴째 history 는 클라이언트가 보낸 것만).
3. 재시작 폴백: 서버 메모리를 비운 상태에서 클라이언트 폴백 history 가 주입된다.
4. thread 격리: 서로 다른 chat_token 의 대화가 섞이지 않는다.
"""
import asyncio
import importlib

import pytest

from schemas.chat_schema import ChatRequest, HistoryItem
from schemas.intent_schema import IntentResult, IntentType, Entities
from graph import nodes
import graph.builder as builder


# 각 테스트가 노드가 받은 history 를 관찰할 수 있도록 캡처 버퍼
_captured = {}


def _setup(monkeypatch, answer_text="응답입니다"):
    """classify=SMALL_TALK 고정 + small_talk 노드가 history 캡처하도록 치환.

    [중요] checkpointer 상태가 테스트 간 누적되지 않도록 builder 캐시 + checkpointer
           를 매번 새로 만든다.
    """
    ir = IntentResult(
        intent=IntentType.SMALL_TALK, entities=Entities(), confidence=0.9,
    )

    async def fake_classify(state):
        # 운영 classify_node 와 동일하게 dict(primitive)로 기록
        # (체크포인트 직렬화 불변식 검증)
        return {"intent_result": ir.model_dump(mode="json")}

    async def fake_small_talk(state):
        # 노드가 실제로 받은 history 를 캡처 (멀티턴 주입 검증용)
        _captured["history"] = list(state.get("history", []))
        return {"raw_answer": answer_text}

    monkeypatch.setattr(nodes, "classify_node", fake_classify, raising=True)
    monkeypatch.setattr(nodes, "small_talk_node", fake_small_talk, raising=True)

    # checkpointer/그래프 싱글톤 초기화 → 깨끗한 새 MemorySaver 로 재컴파일
    from langgraph.checkpoint.memory import MemorySaver
    builder._compiled_app = None
    builder._checkpointer = MemorySaver()

    # chat.py 가 import 시점에 build_graph()/messages_to_history 를 잡으므로 reload
    import routers.chat as chat
    importlib.reload(chat)
    # Oracle 의존 격리
    monkeypatch.setattr(chat, "resolve_chat_token", lambda t: 7, raising=True)
    monkeypatch.setattr(chat, "save_chat_history", lambda *a, **k: None, raising=True)
    return chat


# ────────────────────────────────────────────────────────────────────────
# 1) 로그인 2턴 — 서버 메모리 누적 확인
# ────────────────────────────────────────────────────────────────────────
def test_logged_in_multiturn_accumulates(monkeypatch):
    chat = _setup(monkeypatch, answer_text="네 셔츠 추천드려요")

    # 1턴
    req1 = ChatRequest(chat_token="tok-A", question="셔츠 추천해줘", history=[])
    asyncio.run(chat.process_chat_pipeline(req1))
    # 1턴 노드는 history 가 비어있어야 함 (첫 대화)
    assert _captured["history"] == []

    # 2턴 (클라이언트는 history 를 안 보냄 → 서버 메모리에 의존)
    req2 = ChatRequest(chat_token="tok-A", question="그럼 그 색은?", history=[])
    asyncio.run(chat.process_chat_pipeline(req2))

    # 2턴 노드의 history 에 1턴 질문/답변이 서버 메모리에서 복원되어야 함
    assert _captured["history"] == [
        {"role": "user", "text": "셔츠 추천해줘"},
        {"role": "bot", "text": "네 셔츠 추천드려요"},
    ]


# ────────────────────────────────────────────────────────────────────────
# 2) 게스트 2턴 — 서버 메모리에 누적되지 않음 (클라이언트 history 만)
# ────────────────────────────────────────────────────────────────────────
def test_guest_no_server_memory(monkeypatch):
    chat = _setup(monkeypatch, answer_text="안녕하세요")

    # 게스트 1턴
    req1 = ChatRequest(chat_token=None, question="안녕", history=[])
    asyncio.run(chat.process_chat_pipeline(req1))
    assert _captured["history"] == []

    # 게스트 2턴 — 클라이언트가 직접 history 를 실어보냄
    req2 = ChatRequest(
        chat_token=None,
        question="뭐 추천해?",
        history=[HistoryItem(role="user", text="안녕"),
                 HistoryItem(role="bot", text="안녕하세요")],
    )
    asyncio.run(chat.process_chat_pipeline(req2))
    # 게스트는 클라이언트가 보낸 history 그대로 (서버 누적 아님)
    assert _captured["history"] == [
        {"role": "user", "text": "안녕"},
        {"role": "bot", "text": "안녕하세요"},
    ]


# ────────────────────────────────────────────────────────────────────────
# 3) 서버 재시작 폴백 — 메모리 비운 뒤 클라이언트 폴백 history 주입
# ────────────────────────────────────────────────────────────────────────
def test_restart_fallback_to_client_history(monkeypatch):
    chat = _setup(monkeypatch, answer_text="첫 답변")

    # 1턴 (서버 메모리에 누적됨)
    req1 = ChatRequest(chat_token="tok-B", question="첫 질문", history=[])
    asyncio.run(chat.process_chat_pipeline(req1))

    # === uvicorn 재시작 시뮬레이션: checkpointer 를 새것으로 교체 + 그래프 재컴파일 ===
    from langgraph.checkpoint.memory import MemorySaver
    builder._compiled_app = None
    builder._checkpointer = MemorySaver()
    importlib.reload(chat)
    monkeypatch.setattr(chat, "resolve_chat_token", lambda t: 7, raising=True)
    monkeypatch.setattr(chat, "save_chat_history", lambda *a, **k: None, raising=True)

    # 재시작 후, 같은 thread 로 2턴 — 서버 메모리엔 없지만 클라이언트가 폴백 history 보냄
    req2 = ChatRequest(
        chat_token="tok-B",
        question="이어서 질문",
        history=[HistoryItem(role="user", text="첫 질문"),
                 HistoryItem(role="bot", text="첫 답변")],
    )
    asyncio.run(chat.process_chat_pipeline(req2))

    # 서버 메모리가 비었으므로 클라이언트 폴백 history 가 주입되어야 함
    assert _captured["history"] == [
        {"role": "user", "text": "첫 질문"},
        {"role": "bot", "text": "첫 답변"},
    ]


# ────────────────────────────────────────────────────────────────────────
# 4) thread 격리 — 다른 chat_token 의 대화가 섞이지 않음
# ────────────────────────────────────────────────────────────────────────
def test_thread_isolation(monkeypatch):
    chat = _setup(monkeypatch, answer_text="답변X")

    # thread A 1턴
    asyncio.run(chat.process_chat_pipeline(
        ChatRequest(chat_token="tok-A", question="A의 질문", history=[])
    ))
    # thread B 1턴
    asyncio.run(chat.process_chat_pipeline(
        ChatRequest(chat_token="tok-B", question="B의 질문", history=[])
    ))
    # thread A 2턴 — A 의 이력만 보여야 함 (B 섞임 없음)
    asyncio.run(chat.process_chat_pipeline(
        ChatRequest(chat_token="tok-A", question="A의 두번째", history=[])
    ))
    assert _captured["history"] == [
        {"role": "user", "text": "A의 질문"},
        {"role": "bot", "text": "답변X"},
    ]
    # B 의 질문이 섞이지 않았는지
    texts = [h["text"] for h in _captured["history"]]
    assert "B의 질문" not in texts
