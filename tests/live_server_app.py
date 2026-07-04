"""
실서버 SSE 검증용 standalone 앱 (테스트 전용, 배포 파일 아님)
=============================================================
GPT/Oracle/Chroma 없이 process_chat_pipeline 만 stub 으로 대체하고,
실제 pipeline.stream_util.event_stream + StreamingResponse 를 그대로 사용한다.
→ SSE 가 HTTP 소켓 위에서 진짜로 청크 단위로 흘러나오는지 curl -N 으로 확인하기 위함.
"""
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from pipeline.stream_util import event_stream

app = FastAPI()


@dataclass
class StubResult:
    answer: str
    intent: str
    confidence: float


# 인텐트별 stub 응답 (실제 핸들러 대신)
_STUB = {
    "안녕": StubResult("안녕하세요! 무엇을 도와드릴까요?", "SMALL_TALK", 0.95),
    "에러": None,  # 파이프라인 실패 시뮬레이션
}


async def stub_pipeline(request: dict):
    """process_chat_pipeline 의 stub. delay 로 GPT 지연도 흉내."""
    msg = request.get("question", "")
    if msg == "에러":
        raise RuntimeError("stub failure")
    result = _STUB.get(msg)
    if result is None:
        result = StubResult(
            answer=f"'{msg}' 에 대한 검색 결과입니다. 추천 상품을 정리해드릴게요.",
            intent="SEMANTIC_SEARCH",
            confidence=0.88,
        )
    return result


@app.post("/chat/stream")
async def chat_stream(request: dict):
    # 실제 chat_stream 과 동일한 헤더/구성
    return StreamingResponse(
        event_stream(stub_pipeline, request, chunk_size=3, delay=0.03),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
