"""
chatbot-stream.js 의 SSE 파싱 알고리즘 회귀 테스트
==================================================

브라우저 코드 자체는 이 환경에서 실행할 수 없으므로, JS 의 핵심 로직
(버퍼 누적 + '\\n\\n' 기준 블록 분리 + 라인 파싱)을 Python 으로 1:1 포팅해
'네트워크 청크가 SSE 블록 경계와 전혀 무관하게 쪼개져 들어와도'
이벤트가 정확히 복원되는지 검증한다.

이것이 fetch+ReadableStream 수동 파싱에서 가장 흔한 버그(블록이 두 read 에
걸쳐 쪼개질 때 유실/중복)를 막는지 보는 테스트다.

주의: 여기서는 이미 디코딩된 str 을 글자 단위로 쪼개므로 멀티바이트 분할은
다루지 않는다. JS 쪽 바이트 경계 분할은 TextDecoder({stream:true}) 가 처리하며,
이 테스트는 '블록 경계' 견고성만 책임진다.
"""
import json

from pipeline.stream_util import sse_event


# ----- JS parseSseBlock 의 Python 포팅 -----
def parse_sse_block(block: str):
    event = None
    data_raw = None
    for line in block.split("\n"):
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_raw = line[len("data:"):].strip()
    if event is None or data_raw is None:
        return None
    try:
        return {"event": event, "data": json.loads(data_raw)}
    except Exception:
        return None


# ----- JS streamChat while-loop 버퍼 분리 로직의 Python 포팅 -----
def feed_stream(network_chunks):
    buffer = ""
    events = []
    for nc in network_chunks:
        buffer += nc
        while True:
            idx = buffer.find("\n\n")
            if idx == -1:
                break
            block = buffer[:idx]
            buffer = buffer[idx + 2:]
            if not block.strip():
                continue
            parsed = parse_sse_block(block)
            if parsed:
                events.append(parsed)
    return events


def build_stream():
    """서버가 보낼 법한 전체 SSE 스트림 한 덩어리."""
    parts = [
        sse_event("chunk", {"text": "안"}),
        sse_event("chunk", {"text": "녕하"}),
        sse_event("chunk", {"text": "세요"}),
        sse_event("done", {"intent": "SMALL_TALK", "confidence": 0.9}),
    ]
    return "".join(parts)


EXPECTED_EVENTS = ["chunk", "chunk", "chunk", "done"]
EXPECTED_TEXT = "안녕하세요"


def _assert_stream(events):
    assert [e["event"] for e in events] == EXPECTED_EVENTS
    text = "".join(e["data"]["text"] for e in events if e["event"] == "chunk")
    assert text == EXPECTED_TEXT
    done = [e for e in events if e["event"] == "done"][0]
    assert done["data"]["intent"] == "SMALL_TALK"


class TestSseParseAlgorithm:
    def test_single_delivery(self):
        _assert_stream(feed_stream([build_stream()]))

    def test_char_by_char_delivery(self):
        # 1글자씩 도착 (가장 잘게 쪼개진 최악 케이스)
        _assert_stream(feed_stream(list(build_stream())))

    def test_arbitrary_boundaries(self):
        stream = build_stream()
        for size in [1, 2, 3, 5, 7, 11, 13, 17, 50, 9999]:
            chunks = [stream[i:i + size] for i in range(0, len(stream), size)]
            events = feed_stream(chunks)
            assert [e["event"] for e in events] == EXPECTED_EVENTS, f"size={size}"
            text = "".join(e["data"]["text"] for e in events if e["event"] == "chunk")
            assert text == EXPECTED_TEXT, f"size={size}"

    def test_error_event_parsed(self):
        stream = sse_event("error", {"message": "일시적으로 응답을 생성할 수 없습니다."})
        events = feed_stream(list(stream))  # 1글자씩
        assert len(events) == 1
        assert events[0]["event"] == "error"
        assert "생성할 수 없습니다" in events[0]["data"]["message"]

    def test_incomplete_trailing_block_not_emitted(self):
        # done 블록의 종료 빈 줄이 아직 안 온 상태 → 이벤트로 방출되면 안 됨
        stream = build_stream()
        truncated = stream[:-1]  # 마지막 '\n' 제거 → done 블록 미완성
        events = feed_stream([truncated])
        # chunk 3개는 온전, done 은 아직 미완성이라 빠져야 함
        assert [e["event"] for e in events] == ["chunk", "chunk", "chunk"]
