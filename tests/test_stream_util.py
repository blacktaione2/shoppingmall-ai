"""
stream_util.py 단위테스트
  - split_into_chunks  : 글자수 경계 / 빈 문자열 / 한영 혼용 / 재조립 무결성
  - sse_event          : 포맷 / 빈 줄 종료 / 한글 비이스케이프 / 개행 안전성
  - event_stream       : 성공 시퀀스 / 에러 이벤트 / 빈 답변
"""
import asyncio
import json
from dataclasses import dataclass

import pytest
from fastapi import HTTPException

from pipeline.stream_util import split_into_chunks, sse_event, event_stream


# ---------- split_into_chunks ----------
class TestSplitIntoChunks:
    def test_basic_korean(self):
        assert split_into_chunks("안녕하세요", 2) == ["안녕", "하세", "요"]

    def test_exact_multiple(self):
        assert split_into_chunks("123456", 3) == ["123", "456"]

    def test_chunk_larger_than_text(self):
        assert split_into_chunks("hi", 5) == ["hi"]

    def test_empty_string(self):
        assert split_into_chunks("", 3) == []

    def test_chunk_size_one(self):
        assert split_into_chunks("abc", 1) == ["a", "b", "c"]

    def test_chunk_size_zero_is_corrected_to_one(self):
        assert split_into_chunks("abc", 0) == ["a", "b", "c"]

    def test_negative_chunk_size_corrected(self):
        assert split_into_chunks("abc", -5) == ["a", "b", "c"]

    def test_mixed_korean_english(self):
        assert split_into_chunks("A가B나", 2) == ["A가", "B나"]

    def test_default_chunk_size_is_three(self):
        assert split_into_chunks("123456789") == ["123", "456", "789"]

    def test_reassembly_preserves_text(self):
        text = "스트리밍 테스트 문장입니다 with English 123 !?"
        assert "".join(split_into_chunks(text, 3)) == text


# ---------- sse_event ----------
class TestSseEvent:
    def test_chunk_exact_format(self):
        out = sse_event("chunk", {"text": "안녕"})
        assert out == 'event: chunk\ndata: {"text": "안녕"}\n\n'

    def test_ends_with_blank_line(self):
        out = sse_event("done", {"chat_id": 1})
        assert out.endswith("\n\n")

    def test_korean_not_unicode_escaped(self):
        out = sse_event("chunk", {"text": "한글테스트"})
        assert "한글테스트" in out
        assert "\\u" not in out

    def test_newline_in_text_does_not_break_data_line(self):
        # 답변에 줄바꿈이 들어가도 raw 개행은 SSE 구분 3개뿐이어야 한다
        # (event 뒤 1 + data 뒤 1 + 종료 빈 줄 1). 텍스트 내 \n 은 \\n 으로 이스케이프됨.
        out = sse_event("chunk", {"text": "줄1\n줄2"})
        assert out.count("\n") == 3
        assert "\\n" in out  # JSON 이스케이프된 개행 존재


# ---------- event_stream ----------
@dataclass
class FakeResult:
    answer: str
    intent: str
    confidence: float


async def _fake_pipeline_ok(request):
    return FakeResult(answer="안녕하세요", intent="SMALL_TALK", confidence=0.91)


async def _fake_pipeline_fail(request):
    raise RuntimeError("pipeline boom")


# chat_token 인증 실패를 흉내내는 fake pipeline (routers/chat.py 의 401 분기와 동일 패턴)
async def _fake_pipeline_auth_fail(request):
    raise HTTPException(status_code=401, detail="로그인이 만료되었거나 유효하지 않습니다. 다시 로그인해 주세요.")


async def _fake_pipeline_empty(request):
    return FakeResult(answer="", intent="FAQ", confidence=1.0)


def _collect(agen):
    async def _run():
        return [e async for e in agen]
    return asyncio.run(_run())


def _parse(block):
    lines = block.strip().split("\n")
    event = lines[0].replace("event: ", "", 1)
    data = json.loads(lines[1].replace("data: ", "", 1))
    return event, data


class TestEventStream:
    def test_success_sequence_and_metadata(self):
        events = _collect(event_stream(_fake_pipeline_ok, None, chunk_size=2, delay=0))
        parsed = [_parse(e) for e in events]
        kinds = [p[0] for p in parsed]

        # 마지막은 done, 그 앞은 모두 chunk
        assert kinds[-1] == "done"
        assert all(k == "chunk" for k in kinds[:-1])
        # "안녕하세요"(5글자) / 2 = 3청크
        assert kinds.count("chunk") == 3
        # 청크 재조립 = 원문
        text = "".join(p[1]["text"] for p in parsed if p[0] == "chunk")
        assert text == "안녕하세요"
        # done 메타데이터
        done = parsed[-1][1]
        assert done == {"intent": "SMALL_TALK", "confidence": 0.91, "sources": None}

    def test_error_event_on_pipeline_failure(self):
        events = _collect(event_stream(_fake_pipeline_fail, None, delay=0))
        assert len(events) == 1
        event, data = _parse(events[0])
        assert event == "error"
        assert "응답을 생성할 수 없습니다" in data["message"]

    def test_error_event_uses_http_exception_detail(self):
        # HTTPException(401 등)은 일반 예외와 달리 detail 메시지가 그대로 노출되어야 함
        # (사용자가 재로그인해야 하는 상황을 "일시적 오류"로 뭉개면 안 됨)
        events = _collect(event_stream(_fake_pipeline_auth_fail, None, delay=0))
        assert len(events) == 1
        event, data = _parse(events[0])
        assert event == "error"
        assert data["message"] == "로그인이 만료되었거나 유효하지 않습니다. 다시 로그인해 주세요."
        assert "일시적으로" not in data["message"]  # 일반 폴백 문구로 덮이지 않았는지 확인

    def test_empty_answer_emits_only_done(self):
        events = _collect(event_stream(_fake_pipeline_empty, None, delay=0))
        parsed = [_parse(e) for e in events]
        assert len(parsed) == 1
        assert parsed[0][0] == "done"
        assert parsed[0][1]["intent"] == "FAQ"
