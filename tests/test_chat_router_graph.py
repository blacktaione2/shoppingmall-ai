"""
tests/test_chat_router_graph.py
chat.py ↔ LangGraph end-to-end 통합 테스트.

[전략]
- resolve_chat_token / save_chat_history(Oracle) 를 monkeypatch 로 격리.
- classify_node + 핸들러 서비스도 monkeypatch.
- 검증:
  1) 로그인 사용자: 토큰 → member_id 해석, 응답 생성, 이력 저장 호출됨
  2) 게스트: 토큰 없음 → is_guest=True 로 그래프 진입, 이력 저장 안 됨
  3) ChatResponse(answer/intent/confidence) 형식 정상
"""
import asyncio

import pytest

from schemas.chat_schema import ChatRequest, HistoryItem
from schemas.intent_schema import IntentResult, IntentType, Entities
from graph import nodes
import graph.builder as builder


def _patch_classify(monkeypatch, intent: IntentType, **ent):
    emotion = ent.pop("emotion", None)
    ir = IntentResult(
        intent=intent, entities=Entities(**ent), emotion=emotion, confidence=0.9,
    )
    async def fake_classify(state):
        # 운영 classify_node 와 동일하게 dict(primitive)로 기록
        # (체크포인트 직렬화 불변식 검증)
        return {"intent_result": ir.model_dump(mode="json")}
    monkeypatch.setattr(nodes, "classify_node", fake_classify, raising=True)
    builder._compiled_app = None  # 재컴파일 강제


def test_logged_in_user_flow(monkeypatch):
    _patch_classify(monkeypatch, IntentType.SMALL_TALK)

    async def fake_small_talk(state):
        return {"raw_answer": "안녕하세요! 무엇을 도와드릴까요?"}
    monkeypatch.setattr(nodes, "small_talk_node", fake_small_talk, raising=True)
    builder._compiled_app = None

    # chat 모듈은 import 시점에 build_graph() 를 호출하므로, 패치 후 import
    import importlib
    import routers.chat as chat
    importlib.reload(chat)

    # Oracle 의존 격리
    monkeypatch.setattr(chat, "resolve_chat_token", lambda t: 42, raising=True)
    saved = {}
    def fake_save(member_id, q, a, intent):
        saved.update(member_id=member_id, q=q, a=a, intent=intent)
    monkeypatch.setattr(chat, "save_chat_history", fake_save, raising=True)
    # [다중 세션] session_id 미전송 → 서버가 새 대화방을 만든다(Oracle 격리)
    monkeypatch.setattr(
        chat, "create_chat_session",
        lambda member_id: {"session_id": "sess-1", "title": "새 대화", "updated_at": "2024-01-01T00:00:00"},
        raising=True,
    )
    monkeypatch.setattr(chat, "touch_chat_session", lambda *a, **k: None, raising=True)

    req = ChatRequest(chat_token="tok-123", question="안녕", history=[])
    resp = asyncio.run(chat.process_chat_pipeline(req))

    assert resp.intent == "SMALL_TALK"
    assert "안녕하세요" in resp.answer
    # 이력 저장이 호출됐는지
    assert saved["member_id"] == 42
    assert saved["intent"] == "SMALL_TALK"
    # 새 대화방이 생성되어 응답에 실려왔는지
    assert resp.session_id == "sess-1"


def test_guest_flow_no_history_save(monkeypatch):
    _patch_classify(monkeypatch, IntentType.SMALL_TALK)

    async def fake_small_talk(state):
        return {"raw_answer": "안녕하세요!"}
    monkeypatch.setattr(nodes, "small_talk_node", fake_small_talk, raising=True)
    builder._compiled_app = None

    import importlib
    import routers.chat as chat
    importlib.reload(chat)

    # 게스트는 resolve_chat_token 이 호출되면 안 됨
    def fake_resolve(t):
        raise AssertionError("게스트는 토큰 해석 안 함")
    monkeypatch.setattr(chat, "resolve_chat_token", fake_resolve, raising=True)

    save_called = {"n": 0}
    def fake_save(*a, **k):
        save_called["n"] += 1
    monkeypatch.setattr(chat, "save_chat_history", fake_save, raising=True)

    req = ChatRequest(chat_token=None, question="안녕", history=[])
    resp = asyncio.run(chat.process_chat_pipeline(req))

    assert resp.intent == "SMALL_TALK"
    assert save_called["n"] == 0  # 게스트는 저장 안 함


def test_history_passed_to_graph(monkeypatch):
    """멀티턴 이력이 State(history)로 노드에 전달되는지 확인."""
    _patch_classify(monkeypatch, IntentType.SMALL_TALK)

    seen_history = {}
    async def fake_small_talk(state):
        seen_history["h"] = state.get("history")
        return {"raw_answer": "ok"}
    monkeypatch.setattr(nodes, "small_talk_node", fake_small_talk, raising=True)
    builder._compiled_app = None

    import importlib
    import routers.chat as chat
    importlib.reload(chat)
    monkeypatch.setattr(chat, "resolve_chat_token", lambda t: 1, raising=True)
    monkeypatch.setattr(chat, "save_chat_history", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(
        chat, "create_chat_session",
        lambda member_id: {"session_id": "sess-1", "title": "새 대화", "updated_at": "2024-01-01T00:00:00"},
        raising=True,
    )
    monkeypatch.setattr(chat, "touch_chat_session", lambda *a, **k: None, raising=True)

    req = ChatRequest(
        chat_token="tok",
        question="그럼 그건?",
        history=[HistoryItem(role="user", text="셔츠 추천해줘"),
                 HistoryItem(role="bot", text="흰 셔츠 어때요")],
    )
    asyncio.run(chat.process_chat_pipeline(req))

    assert seen_history["h"] == [
        {"role": "user", "text": "셔츠 추천해줘"},
        {"role": "bot", "text": "흰 셔츠 어때요"},
    ]
