"""
tests/test_multiturn_memory.py
멀티턴 하이브리드(checkpointer) 통합 테스트.

[전략]
- classify_node + 핸들러 노드를 monkeypatch 로 고정/관찰.
- 핵심은 "노드가 받은 state['history'] 가 무엇인가"를 캡처해서,
  서버 메모리(checkpointer) 누적분이 다음 턴에 제대로 주입되는지 확인하는 것.

[다중 세션 전환]
- LangGraph thread_id 는 이제 chat_token 이 아니라 session_id 다(회원당
  chat_token 은 1개뿐이라 여러 대화방을 구분할 수 없기 때문).
  CHAT_SESSION 관련 Oracle 함수(create_chat_session/get_chat_session_owner/
  touch_chat_session)를 인메모리 딕셔너리로 monkeypatch 해 대체한다.

[검증 시나리오]
1. 로그인 2턴: 2턴째 노드의 history 에 1턴 질문/답변이 들어있다 (서버 메모리 누적,
   1턴 응답의 session_id 를 2턴 요청에 그대로 실어보낸다).
2. 게스트 2턴: config 없이 invoke 되어 서버 메모리에 누적되지 않는다
   (2턴째 history 는 클라이언트가 보낸 것만).
3. 재시작 폴백: 서버 메모리를 비운 상태에서 클라이언트 폴백 history 가 주입된다.
4. thread 격리: 서로 다른 session_id 의 대화가 섞이지 않는다.
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

    # [다중 세션] CHAT_SESSION 을 인메모리 딕셔너리로 대체 (session_id -> member_id)
    sessions: dict[str, int] = {}
    counter = {"n": 0}

    def fake_create_chat_session(member_id):
        counter["n"] += 1
        sid = f"sess-{counter['n']}"
        sessions[sid] = member_id
        return {"session_id": sid, "title": "새 대화", "updated_at": "2024-01-01T00:00:00"}

    def fake_get_chat_session_owner(session_id):
        return sessions.get(session_id)

    monkeypatch.setattr(chat, "create_chat_session", fake_create_chat_session, raising=True)
    monkeypatch.setattr(chat, "get_chat_session_owner", fake_get_chat_session_owner, raising=True)
    monkeypatch.setattr(chat, "touch_chat_session", lambda *a, **k: None, raising=True)
    return chat


# ────────────────────────────────────────────────────────────────────────
# 1) 로그인 2턴 — 서버 메모리 누적 확인
# ────────────────────────────────────────────────────────────────────────
def test_logged_in_multiturn_accumulates(monkeypatch):
    chat = _setup(monkeypatch, answer_text="네 셔츠 추천드려요")

    # 1턴 (session_id 미전송 → 서버가 새 대화방 생성)
    req1 = ChatRequest(chat_token="tok-A", question="셔츠 추천해줘", history=[])
    resp1 = asyncio.run(chat.process_chat_pipeline(req1))
    # 1턴 노드는 history 가 비어있어야 함 (첫 대화)
    assert _captured["history"] == []
    assert resp1.session_id is not None

    # 2턴 (같은 session_id 를 실어보내 같은 대화방으로 이어감)
    req2 = ChatRequest(
        chat_token="tok-A", question="그럼 그 색은?", history=[],
        session_id=resp1.session_id,
    )
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
    resp1 = asyncio.run(chat.process_chat_pipeline(req1))
    session_id = resp1.session_id

    # === uvicorn 재시작 시뮬레이션: checkpointer 를 새것으로 교체 + 그래프 재컴파일 ===
    from langgraph.checkpoint.memory import MemorySaver
    builder._compiled_app = None
    builder._checkpointer = MemorySaver()
    importlib.reload(chat)
    monkeypatch.setattr(chat, "resolve_chat_token", lambda t: 7, raising=True)
    monkeypatch.setattr(chat, "save_chat_history", lambda *a, **k: None, raising=True)
    # CHAT_SESSION 자체(Oracle 행)는 재시작해도 남아있으므로, 소유권 검증만
    # 통과하면 된다(이 session_id 는 이 회원 소유라고 가정).
    monkeypatch.setattr(chat, "get_chat_session_owner", lambda sid: 7, raising=True)
    monkeypatch.setattr(chat, "touch_chat_session", lambda *a, **k: None, raising=True)

    # 재시작 후, 같은 thread 로 2턴 — 서버 메모리엔 없지만 클라이언트가 폴백 history 보냄
    req2 = ChatRequest(
        chat_token="tok-B",
        question="이어서 질문",
        history=[HistoryItem(role="user", text="첫 질문"),
                 HistoryItem(role="bot", text="첫 답변")],
        session_id=session_id,
    )
    asyncio.run(chat.process_chat_pipeline(req2))

    # 서버 메모리가 비었으므로 클라이언트 폴백 history 가 주입되어야 함
    assert _captured["history"] == [
        {"role": "user", "text": "첫 질문"},
        {"role": "bot", "text": "첫 답변"},
    ]


# ────────────────────────────────────────────────────────────────────────
# 4) thread 격리 — 서로 다른 session_id 의 대화가 섞이지 않음
# ────────────────────────────────────────────────────────────────────────
def test_thread_isolation(monkeypatch):
    chat = _setup(monkeypatch, answer_text="답변X")

    # 대화방 A 1턴 (session_id 미전송 → 새 대화방 A 생성)
    resp_a1 = asyncio.run(chat.process_chat_pipeline(
        ChatRequest(chat_token="tok-A", question="A의 질문", history=[])
    ))
    session_a = resp_a1.session_id
    # 대화방 B 1턴 (session_id 미전송 → 새 대화방 B 생성, A 와 다른 thread)
    asyncio.run(chat.process_chat_pipeline(
        ChatRequest(chat_token="tok-B", question="B의 질문", history=[])
    ))
    # 대화방 A 2턴 — A 의 이력만 보여야 함 (B 섞임 없음)
    asyncio.run(chat.process_chat_pipeline(
        ChatRequest(chat_token="tok-A", question="A의 두번째", history=[], session_id=session_a)
    ))
    assert _captured["history"] == [
        {"role": "user", "text": "A의 질문"},
        {"role": "bot", "text": "답변X"},
    ]
    # B 의 질문이 섞이지 않았는지
    texts = [h["text"] for h in _captured["history"]]
    assert "B의 질문" not in texts

