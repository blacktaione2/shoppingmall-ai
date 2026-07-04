"""
tests/test_multi_agent_endpoint.py
/chat/multi-agent 엔드포인트 통합 테스트.

[전략]
- multi_agent_chat 의 _multi_app 을 '가짜 모델 + 치환 supervisor' Agent로 교체.
- Oracle/도구 외부 의존 격리.

[검증]
1. 메트릭 집계: tool_calls / tools_used / total_tokens 채워짐
2. intent='MULTI_AGENT', answer 정상
3. 게스트 이력 미저장
"""
import importlib
import uuid

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from langchain_core.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langgraph.checkpoint.memory import MemorySaver

from graph import tools as tools_mod
import graph.multi_agent_builder as mab


class FakeToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _patch_supervisor(monkeypatch, decisions):
    seq = list(decisions)
    idx = {"i": 0}
    async def fake_supervisor(state):
        i = idx["i"]; idx["i"] += 1
        nxt, final = seq[i]
        rc = state.get("routing_count", 0)
        upd = {"next_agent": nxt, "routing_count": rc + 1}
        if nxt == "FINISH":
            upd["final_answer"] = final
        return upd
    monkeypatch.setattr(mab, "_supervisor_node", fake_supervisor, raising=True)


def _make_client(monkeypatch, sub_seq, decisions):
    def fake_search(**kwargs):
        return [{"product_name": "패딩", "category": "상의", "price": 99000, "stock": 3}]
    monkeypatch.setattr(tools_mod, "search_products_structured", fake_search, raising=True)

    _patch_supervisor(monkeypatch, decisions)
    model = FakeToolCallingModel(responses=sub_seq)
    app_graph = mab.build_multi_agent(
        model=model, checkpointer=MemorySaver(), force_rebuild=True,
    )

    import routers.multi_agent_chat as mac
    importlib.reload(mac)
    monkeypatch.setattr(mac, "_multi_app", app_graph, raising=True)
    monkeypatch.setattr(mac, "resolve_chat_token", lambda t: 7, raising=True)
    monkeypatch.setattr(mac, "save_chat_history", lambda *a, **k: None, raising=True)

    test_app = FastAPI()
    test_app.include_router(mac.router)
    return TestClient(test_app), mac


def test_multi_agent_metrics(monkeypatch):
    sub_seq = [
        AIMessage(
            content="",
            tool_calls=[{"name": "search_products", "args": {}, "id": "c1"}],
            usage_metadata={"input_tokens": 80, "output_tokens": 10, "total_tokens": 90},
        ),
        AIMessage(content="찾았습니다.",
                  usage_metadata={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}),
    ]
    decisions = [("product_agent", ""), ("FINISH", "패딩을 추천드려요.")]
    client, mac = _make_client(monkeypatch, sub_seq, decisions)

    resp = client.post("/chat/multi-agent", json={
        "chat_token": "tok-1", "question": "패딩 추천", "history": [],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "MULTI_AGENT"
    assert "패딩" in body["answer"]
    assert body["tool_calls"] == 1
    assert body["tools_used"] == ["search_products"]
    assert body["total_tokens"] == 150   # 90 + 60


def test_multi_agent_guest_no_save(monkeypatch):
    sub_seq = [
        AIMessage(content="", tool_calls=[{"name": "search_products", "args": {}, "id": "c1"}]),
        AIMessage(content="ok"),
    ]
    decisions = [("product_agent", ""), ("FINISH", "추천드려요.")]
    client, mac = _make_client(monkeypatch, sub_seq, decisions)

    save_calls = {"n": 0}
    monkeypatch.setattr(mac, "save_chat_history",
                        lambda *a, **k: save_calls.__setitem__("n", save_calls["n"] + 1),
                        raising=True)
    monkeypatch.setattr(mac, "resolve_chat_token",
                        lambda t: (_ for _ in ()).throw(AssertionError("게스트는 토큰해석 안함")),
                        raising=True)

    resp = client.post("/chat/multi-agent", json={
        "chat_token": None, "question": "추천", "history": [],
    })
    assert resp.status_code == 200
    assert resp.json()["answer"] == "추천드려요."
    assert save_calls["n"] == 0
