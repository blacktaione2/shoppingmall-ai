"""
tests/test_metrics_dashboard.py
성능/비용 대시보드 테스트 (metrics.jsonl 집계 + admin 엔드포인트).

[검증]
1. summarize_metrics: 경로별 집계, 순서(라우터→단일→멀티), 깨진 줄 스킵, 평균 계산.
2. 파일 없음/빈 파일 → available=False (예외 없음).
3. GET /admin/metrics/summary: 키 없음/틀린 키 403, 정상 키 200.
4. GET /admin/metrics: HTML 페이지가 인증 없이 열리고 Chart.js 를 포함.
"""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import graph.metrics as metrics_mod
from routers import admin


_ADMIN_KEY = "test-admin-key"


@pytest.fixture(autouse=True)
def _admin_key_env(monkeypatch):
    monkeypatch.setenv("ADMIN_KEY", _ADMIN_KEY)
    monkeypatch.setenv("METRICS_ENABLED", "true")


def _write_metrics(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write((r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)) + "\n")


_SAMPLE = [
    {"route": "router_pipeline", "provider": "openai", "latency_ms": 800.0,
     "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
     "tool_calls": 0, "cost_usd": 0.002},
    {"route": "router_pipeline", "provider": "openai", "latency_ms": 400.0,
     "prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280,
     "tool_calls": 0, "cost_usd": 0.0004},
    {"route": "single_agent", "provider": "openai", "latency_ms": 2000.0,
     "prompt_tokens": 3000, "completion_tokens": 500, "total_tokens": 3500,
     "tool_calls": 2, "cost_usd": 0.006},
    {"route": "multi_agent", "provider": "gemini", "latency_ms": 3000.0,
     "prompt_tokens": 5000, "completion_tokens": 700, "total_tokens": 5700,
     "tool_calls": 4, "cost_usd": 0.001},
    "{invalid json line",   # 깨진 줄 — 집계에서 제외되어야 함
]


def test_summarize_groups_and_order(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, _SAMPLE)
    monkeypatch.setenv("METRICS_LOG_PATH", str(path))

    s = metrics_mod.summarize_metrics()
    assert s["available"] is True
    # 깨진 줄 1개 제외 → 4건
    assert s["total_requests"] == 4
    # 경로 순서: 라우터 → 단일 → 멀티
    labels = [r["label"] for r in s["by_route"]]
    assert labels == ["라우터 파이프라인", "단일 Agent", "멀티 Agent"]
    # 라우터 평균 레이턴시 = (800+400)/2 = 600
    router = s["by_route"][0]
    assert router["count"] == 2
    assert router["avg_latency_ms"] == 600.0
    # 누적 비용 합
    assert round(s["cost_total_usd"], 4) == round(0.002 + 0.0004 + 0.006 + 0.001, 4)


def test_summarize_missing_file_is_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("METRICS_LOG_PATH", str(tmp_path / "nope.jsonl"))
    s = metrics_mod.summarize_metrics()
    assert s["available"] is False
    assert s["total_requests"] == 0
    assert s["by_route"] == []


def test_summarize_limit(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, _SAMPLE)
    monkeypatch.setenv("METRICS_LOG_PATH", str(path))
    # 최근 1건만(마지막 유효 줄은 multi_agent) — 깨진 줄 포함 슬라이스라도 파싱만 성공한 것 집계
    s = metrics_mod.summarize_metrics(limit=1)
    # 마지막 줄은 깨진 줄이라 파싱 실패 → 유효 0건 → available False
    assert s["available"] is False


def _client():
    app = FastAPI()
    app.include_router(admin.router)
    return TestClient(app)


def test_summary_endpoint_auth(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    _write_metrics(path, _SAMPLE)
    monkeypatch.setenv("METRICS_LOG_PATH", str(path))
    c = _client()

    assert c.get("/admin/metrics/summary").status_code == 403           # 키 없음
    assert c.get("/admin/metrics/summary",
                 headers={"X-ADMIN-KEY": "wrong"}).status_code == 403    # 틀린 키
    ok = c.get("/admin/metrics/summary", headers={"X-ADMIN-KEY": _ADMIN_KEY})
    assert ok.status_code == 200
    assert ok.json()["available"] is True
    assert ok.json()["total_requests"] == 4


def test_dashboard_html_opens_without_auth():
    c = _client()
    r = c.get("/admin/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "chart.js" in r.text.lower()
