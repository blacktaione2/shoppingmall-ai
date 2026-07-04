"""
tests/test_voice_endpoints.py
음성 STT/TTS/voice 엔드포인트 통합 테스트.

[전략]
- OpenAI 호출(transcribe/synthesize)을 monkeypatch 로 격리.
- /chat/voice 의 파이프라인(process_chat_pipeline)도 monkeypatch 로 고정.
- FastAPI TestClient 로 실제 HTTP 요청(multipart/JSON)을 보내 라우터 동작을 검증.

[검증 시나리오]
1. /chat/transcribe : 오디오 업로드 → STT 텍스트 반환
2. /chat/transcribe : 크기 초과 → 400
3. /chat/transcribe : 잘못된 MIME → 400
4. /chat/tts        : 텍스트 → audio/mpeg 바이너리 반환
5. /chat/voice      : 오디오 → STT → 파이프라인 → TTS → base64 JSON
6. /chat/voice      : 빈 STT 결과 → 422 (안내)
7. /chat/voice      : history Form(JSON) 이 파이프라인에 전달되는지
"""
import base64
import json

import pytest
from fastapi.testclient import TestClient

from schemas.chat_schema import ChatResponse
from services import voice_service
import routers.voice as voice_router


@pytest.fixture
def client(monkeypatch):
    # OpenAI STT/TTS 격리
    async def fake_transcribe(file_bytes, filename):
        return "안녕 추천해줘"
    async def fake_synthesize(text):
        return b"FAKE_MP3_BYTES"
    monkeypatch.setattr(voice_service, "transcribe", fake_transcribe, raising=True)
    monkeypatch.setattr(voice_service, "synthesize", fake_synthesize, raising=True)

    from main import app
    return TestClient(app)


def _audio_file(content=b"x" * 1000, mime="audio/webm", name="voice.webm"):
    return {"file": (name, content, mime)}


# ────────────────────────────────────────────────────────────────────────
# 1) /chat/transcribe 정상
# ────────────────────────────────────────────────────────────────────────
def test_transcribe_ok(client):
    resp = client.post("/chat/transcribe", files=_audio_file())
    assert resp.status_code == 200
    assert resp.json() == {"text": "안녕 추천해줘"}


# ────────────────────────────────────────────────────────────────────────
# 2) /chat/transcribe 크기 초과 → 400
# ────────────────────────────────────────────────────────────────────────
def test_transcribe_too_large(client):
    big = b"x" * (voice_service.MAX_AUDIO_BYTES + 1)
    resp = client.post("/chat/transcribe", files=_audio_file(content=big))
    assert resp.status_code == 400
    assert "큽니다" in resp.json()["detail"]


# ────────────────────────────────────────────────────────────────────────
# 3) /chat/transcribe 잘못된 MIME → 400
# ────────────────────────────────────────────────────────────────────────
def test_transcribe_bad_mime(client):
    resp = client.post(
        "/chat/transcribe",
        files=_audio_file(mime="application/pdf", name="x.pdf"),
    )
    assert resp.status_code == 400
    assert "지원하지 않는" in resp.json()["detail"]


# ────────────────────────────────────────────────────────────────────────
# 4) /chat/tts 정상 → audio/mpeg
# ────────────────────────────────────────────────────────────────────────
def test_tts_ok(client):
    resp = client.post("/chat/tts", json={"text": "안녕하세요 반갑습니다"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content == b"FAKE_MP3_BYTES"


# ────────────────────────────────────────────────────────────────────────
# 4-1) /chat/tts 빈 텍스트 → 422 (스키마 min_length)
# ────────────────────────────────────────────────────────────────────────
def test_tts_empty_text(client):
    resp = client.post("/chat/tts", json={"text": ""})
    assert resp.status_code == 422


# ────────────────────────────────────────────────────────────────────────
# 5) /chat/voice 정상 → base64 JSON
# ────────────────────────────────────────────────────────────────────────
def test_voice_ok(client, monkeypatch):
    async def fake_pipeline(request):
        # STT 결과가 question 으로 잘 들어왔는지 확인
        assert request.question == "안녕 추천해줘"
        return ChatResponse(answer="흰 셔츠 어때요?", intent="SEMANTIC_SEARCH", confidence=0.9)
    monkeypatch.setattr(voice_router, "process_chat_pipeline", fake_pipeline, raising=True)

    resp = client.post("/chat/voice", files=_audio_file())
    assert resp.status_code == 200
    body = resp.json()
    assert body["question"] == "안녕 추천해줘"
    assert body["answer"] == "흰 셔츠 어때요?"
    assert body["intent"] == "SEMANTIC_SEARCH"
    # base64 디코딩 시 가짜 mp3 바이트와 일치
    assert base64.b64decode(body["audio_base64"]) == b"FAKE_MP3_BYTES"


# ────────────────────────────────────────────────────────────────────────
# 6) /chat/voice 빈 STT 결과 → 422
# ────────────────────────────────────────────────────────────────────────
def test_voice_empty_stt(client, monkeypatch):
    async def empty_transcribe(file_bytes, filename):
        return ""   # 무음/잡음
    monkeypatch.setattr(voice_service, "transcribe", empty_transcribe, raising=True)

    def should_not_run(request):
        raise AssertionError("빈 STT 면 파이프라인을 호출하면 안 됨")
    monkeypatch.setattr(voice_router, "process_chat_pipeline", should_not_run, raising=True)

    resp = client.post("/chat/voice", files=_audio_file())
    assert resp.status_code == 422
    assert "인식하지 못" in resp.json()["detail"]


# ────────────────────────────────────────────────────────────────────────
# 7) /chat/voice history Form(JSON) 전달
# ────────────────────────────────────────────────────────────────────────
def test_voice_history_form(client, monkeypatch):
    captured = {}
    async def fake_pipeline(request):
        captured["history"] = [h.model_dump() for h in request.history]
        captured["chat_token"] = request.chat_token
        return ChatResponse(answer="네", intent="SMALL_TALK", confidence=0.8)
    monkeypatch.setattr(voice_router, "process_chat_pipeline", fake_pipeline, raising=True)

    history_json = json.dumps([
        {"role": "user", "text": "이전 질문"},
        {"role": "bot", "text": "이전 답변"},
    ])
    resp = client.post(
        "/chat/voice",
        files=_audio_file(),
        data={"chat_token": "tok-123", "history": history_json},
    )
    assert resp.status_code == 200
    assert captured["chat_token"] == "tok-123"
    assert captured["history"] == [
        {"role": "user", "text": "이전 질문"},
        {"role": "bot", "text": "이전 답변"},
    ]
