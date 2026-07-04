"""
pipeline/stream_util.py — SSE 스트리밍 유틸리티
================================

설계(확정):
  - 가짜 스트리밍(A안): GPT 토큰을 실시간 중계하지 않는다.
    process_chat_pipeline 이 guard 통과 + 프리픽스 제거까지 끝낸
    '검증 완료 answer'를 서버에서 글자 수 기준으로 잘라 흘려보낸다.
  - 6개 인텐트 모두 동일 경로(고정 응답도 청크로 쪼개져 나감).
  - hallucination_guard / strip_known_prefix 는 전부 청크 분할 '이전'에 완료된 상태.

이 모듈은 GPT/DB/Chroma 에 의존하지 않는 순수 로직만 담아 단위테스트가 가능하다.
실제 파이프라인은 routers/chat.py 의 process_chat_pipeline 이 주입(inject)한다.
"""

import asyncio
import json
from typing import AsyncIterator, Awaitable, Callable, List

from fastapi import HTTPException


def split_into_chunks(text: str, chunk_size: int = 3) -> List[str]:
    """
    검증된 answer 를 글자 수 기준으로 분할한다.

    한국어는 음절 자체가 완결된 글자라 영어처럼 단어 경계로 끊을 필요가 없어
    고정 글자 수로 분할한다(영문/숫자 혼용도 동일 규칙으로 처리).

    :param text:       분할 대상(이미 guard/프리픽스 처리 완료 상태여야 함)
    :param chunk_size: 청크당 글자 수(기본 3). 1 미만이 들어오면 1로 보정한다.
    :return:           청크 문자열 리스트. 빈 입력이면 빈 리스트.
    """
    if not text:
        return []
    if chunk_size < 1:
        chunk_size = 1
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def sse_event(event: str, data: dict) -> str:
    """
    SSE 이벤트 1블록을 와이어 포맷 문자열로 직렬화한다.

    형식:
        event: <event>\\n
        data: <json>\\n
        \\n                  (블록 종료 빈 줄)

    data 는 json.dumps(ensure_ascii=False) 로 직렬화한다.
      - ensure_ascii=False : 한글이 \\uXXXX 로 깨지지 않도록 원문 보존.
      - JSON 직렬화 덕분에 answer 안에 줄바꿈(\\n)이 있어도 \\\\n 으로 이스케이프되어
        data 라인이 여러 줄로 쪼개지지 않는다(SSE 포맷 안전성 보장).

    :param event: 이벤트 이름 (chunk / done / error)
    :param data:  data 라인에 실을 dict
    :return:      SSE 텍스트 1블록
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def event_stream(
    pipeline_fn: Callable[..., Awaitable],
    request,
    chunk_size: int = 3,
    delay: float = 0.03,
) -> AsyncIterator[str]:
    """
    /chat/stream 의 응답 제너레이터.
    StreamingResponse 에 그대로 전달되어 SSE 스트림을 만든다.

    동작 순서:
      1) await pipeline_fn(request)
         → 검증 완료된 ChatResponse 확보. (이 시점에 CHAT_HISTORY 저장도 끝남)
         → 파이프라인이 예외를 던지면 스트리밍을 시작하지 않고 error 이벤트 1개만 보냄.
           HTTPException(예: chat_token 인증 실패로 401)은 일반 예외와
           분리해 detail 메시지를 그대로 노출한다 — "일시적 오류"가 아니라
           사용자가 재로그인 등 행동을 취해야 하는 상황이라 구분이 필요하다.
           그 외 모든 예외는 기존과 동일하게 일반 메시지로 감춘다.
      2) result.answer 를 청크로 분할 → event: chunk 로 순차 전송 (청크당 delay 만큼 대기)
      3) 마지막에 event: done 으로 메타데이터(intent/confidence/chat_id) 전송

    :param pipeline_fn: async callable. await pipeline_fn(request) 가
                        .answer / .intent / .confidence 속성을 가진 객체를 반환해야 함.
                        (실제 ChatResponse 에는 chat_id 가 없어 done 이벤트에도 포함하지 않음)
    :param request:     pipeline_fn 에 그대로 넘길 요청 객체(ChatRequestDto)
    :param chunk_size:  청크당 글자 수(기본 3)
    :param delay:       청크 간 대기 초(기본 0.03 = 30ms). 타이핑 속도 튜닝 지점.
    """
    try:
        result = await pipeline_fn(request)
    except HTTPException as e:
        # 인증/입력 검증성 오류 - detail 메시지를 그대로 노출(내부 구현 정보 누출 없는 안내문)
        yield sse_event("error", {"message": e.detail})
        return
    except Exception:
        # 파이프라인 자체 실패: 그 외 어떤 내부 오류든 사용자에겐 일반 메시지만 노출.
        yield sse_event("error", {"message": "일시적으로 응답을 생성할 수 없습니다."})
        return

    for chunk in split_into_chunks(result.answer, chunk_size):
        yield sse_event("chunk", {"text": chunk})
        if delay > 0:
            await asyncio.sleep(delay)

    yield sse_event(
        "done",
        {
            "intent": result.intent,
            "confidence": result.confidence,
        },
    )
