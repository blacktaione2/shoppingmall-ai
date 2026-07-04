"""
tests/test_agent_endpoint.py
/chat/agent 엔드포인트 통합 테스트.

[전략]
- agent_chat 라우터가 쓰는 _agent_app 을 '가짜 모델 기반 Agent'로 교체.
- Oracle(resolve_chat_token/save_chat_history) 격리.
- 도구 내부 외부 의존도 monkeypatch.

[검증]
1. 메트릭 집계: tool_calls / tools_used / total_tokens 가 응답에 채워진다.
2. 기존 ChatResponse 하위호환: answer/intent/confidence 정상.
3. 게스트도 처리되고 이력 저장은 안 된다.
"""
import importlib
import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langgraph.checkpoint.memory import MemorySaver

from graph import tools as tools_mod
import graph.agent_builder as agent_builder


class FakeToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _make_client(monkeypatch, ai_sequence):
    """가짜 모델로 Agent 를 빌드해 agent_chat._agent_app 에 주입한 TestClient 반환."""
    # 도구 외부 의존 격리 (search_products)
    def fake_search(**kwargs):
        return [{"product_name": "패딩", "category": "상의", "price": 99000, "stock": 3}]
    monkeypatch.setattr(tools_mod, "search_products_structured", fake_search, raising=True)

    model = FakeToolCallingModel(responses=ai_sequence)
    app_graph = agent_builder.build_agent(
        model=model, checkpointer=MemorySaver(), force_rebuild=True,
    )

    import routers.agent_chat as ac
    importlib.reload(ac)
    # reload 후 _agent_app 을 가짜로 교체
    monkeypatch.setattr(ac, "_agent_app", app_graph, raising=True)
    monkeypatch.setattr(ac, "resolve_chat_token", lambda t: 7, raising=True)
    monkeypatch.setattr(ac, "save_chat_history", lambda *a, **k: None, raising=True)

    # main 앱에 reload 된 라우터를 다시 물리기보다, ac.router 를 직접 마운트
    from fastapi import FastAPI
    test_app = FastAPI()
    test_app.include_router(ac.router)
    return TestClient(test_app), ac


def test_agent_metrics_populated(monkeypatch):
    # 토큰 메타데이터가 담긴 AIMessage 로 메트릭 집계 검증
    seq = [
        AIMessage(
            content="",
            tool_calls=[{"name": "search_products", "args": {"category": "상의"}, "id": "c1"}],
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        ),
        AIMessage(
            content="패딩을 추천드려요. 99,000원입니다.",
            usage_metadata={"input_tokens": 150, "output_tokens": 30, "total_tokens": 180},
        ),
    ]
    client, ac = _make_client(monkeypatch, seq)

    resp = client.post("/chat/agent", json={
        "chat_token": "tok-1", "question": "상의 추천", "history": [],
    })
    assert resp.status_code == 200
    body = resp.json()
    # 기본 필드
    assert body["intent"] == "AGENT"
    assert "패딩" in body["answer"]
    # 메트릭 (결정 ②)
    assert body["tool_calls"] == 1
    assert body["tools_used"] == ["search_products"]
    assert body["total_tokens"] == 300   # 120 + 180


def test_agent_guest_no_save(monkeypatch):
    seq = [
        AIMessage(content="", tool_calls=[{
            "name": "search_products", "args": {}, "id": "c1"}]),
        AIMessage(content="패딩 추천드려요."),
    ]
    client, ac = _make_client(monkeypatch, seq)

    save_calls = {"n": 0}
    monkeypatch.setattr(ac, "save_chat_history",
                        lambda *a, **k: save_calls.__setitem__("n", save_calls["n"] + 1),
                        raising=True)
    # 게스트는 resolve_chat_token 호출 안 함
    monkeypatch.setattr(ac, "resolve_chat_token",
                        lambda t: (_ for _ in ()).throw(AssertionError("게스트는 토큰해석 안함")),
                        raising=True)

    resp = client.post("/chat/agent", json={
        "chat_token": None, "question": "추천해줘", "history": [],
    })
    assert resp.status_code == 200
    assert resp.json()["answer"] == "패딩 추천드려요."
    assert save_calls["n"] == 0   # 게스트는 저장 안 함
