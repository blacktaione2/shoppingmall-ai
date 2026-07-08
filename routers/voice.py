"""
routers/voice.py
음성(STT/TTS) 라우터

[엔드포인트]
  1) POST /chat/transcribe : 오디오 → 텍스트 (STT 단독)
  2) POST /chat/tts        : 텍스트 → mp3 스트리밍 (TTS 단독)
  3) POST /chat/voice      : 오디오 → STT → 기존 파이프라인 → TTS
                             → JSON{question, answer, intent, confidence, audio_base64}

[설계]
- 음성 처리 핵심(STT/TTS/정제/검증)은 services/voice_service.py 가 담당.
  라우터는 '입출력 변환 + 검증 + 파이프라인 연결'만 책임진다.
- /chat/voice 는 STT 결과 텍스트를 ChatRequest 로 감싸 기존
  routers.chat.process_chat_pipeline() 을 그대로 재호출한다.
  → 멀티턴(checkpointer)/환각가드/인텐트 분기가 전부 자동으로 동작.
- 오디오는 multipart/form-data 로 받는다(파일 업로드).
  · /chat/voice 는 JSON body 를 동시에 받을 수 없으므로 chat_token / history 를
    Form 필드로 받는다. history 는 JSON 문자열로 받아 파싱한다.
- base64 인코딩은 라우터에서 처리(서비스는 순수 bytes 반환 유지):
  · /chat/voice → base64 문자열(JSON 에 실어야 하므로)
  · /chat/tts   → StreamingResponse(바이너리 그대로)
"""
import base64
import json
import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from schemas.chat_schema import (
    ChatRequest,
    HistoryItem,
    TranscribeResponse,
    TtsRequest,
    VoiceChatResponse,
)
from services import voice_service
from services.voice_service import AudioValidationError
from routers.chat import process_chat_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["voice"])

_STT_FAILED_MESSAGE = "음성을 인식하지 못했어요. 조용한 곳에서 다시 말씀해 주세요."
_VOICE_ERROR_MESSAGE = (
    "죄송합니다. 음성 처리 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요."
)


def _parse_history_form(history_json: str | None) -> list[HistoryItem]:
    """Form 으로 전달된 history JSON 문자열 → List[HistoryItem].

    - None/빈 문자열이면 빈 리스트.
    - 파싱 실패/형식 오류는 멀티턴 이력 누락일 뿐 치명적이지 않으므로,
      경고 로그만 남기고 빈 리스트로 폴백한다(요청 전체를 실패시키지 않음).
    """
    if not history_json:
        return []
    try:
        raw = json.loads(history_json)
        if not isinstance(raw, list):
            return []
        return [HistoryItem(**item) for item in raw]
    except Exception:
        logger.warning("history Form 파싱 실패 → 빈 이력으로 폴백: %r", history_json)
        return []


# ════════════════════════════════════════════════════════════════════════
# 1) STT 단독
# ════════════════════════════════════════════════════════════════════════
@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)) -> TranscribeResponse:
    """오디오 업로드 → 인식 텍스트 반환."""
    file_bytes = await file.read()
    try:
        voice_service.validate_audio_upload(file_bytes, file.content_type)
    except AudioValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        text = await voice_service.transcribe(file_bytes, file.filename or "voice.webm")
    except Exception:
        logger.exception("STT 처리 실패")
        raise HTTPException(status_code=502, detail=_VOICE_ERROR_MESSAGE)

    return TranscribeResponse(text=text)


# ════════════════════════════════════════════════════════════════════════
# 2) TTS 단독
# ════════════════════════════════════════════════════════════════════════
@router.post("/tts")
async def text_to_speech(request: TtsRequest):
    """텍스트 → mp3 스트리밍 응답."""
    try:
        audio_bytes = await voice_service.synthesize(request.text)
    except Exception:
        logger.exception("TTS 처리 실패")
        raise HTTPException(status_code=502, detail=_VOICE_ERROR_MESSAGE)

    return StreamingResponse(
        iter([audio_bytes]),
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'inline; filename="tts.mp3"'},
    )


# ════════════════════════════════════════════════════════════════════════
# 3) 음성 대화 (STT → 파이프라인 → TTS)
# ════════════════════════════════════════════════════════════════════════
@router.post("/voice", response_model=VoiceChatResponse)
async def voice_chat(
    file: UploadFile = File(...),
    chat_token: str | None = Form(None),
    history: str | None = Form(None),
    session_id: str | None = Form(None),
) -> VoiceChatResponse:
    """오디오 → STT → 기존 파이프라인 → TTS → JSON(텍스트+음성 base64).

    chat_token / history / session_id 는 Form 필드(파일과 동시 전송).
    history 는 JSON 문자열(예: '[{"role":"user","text":"..."}]').
    session_id 는 로그인 회원이 현재 열어둔 대화방 ID(다중 세션과 동일하게 이어짐).
    """
    file_bytes = await file.read()
    try:
        voice_service.validate_audio_upload(file_bytes, file.content_type)
    except AudioValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 1) STT
    try:
        question = await voice_service.transcribe(
            file_bytes, file.filename or "voice.webm"
        )
    except Exception:
        logger.exception("voice STT 처리 실패")
        raise HTTPException(status_code=502, detail=_VOICE_ERROR_MESSAGE)

    # 1-1) 빈 인식 결과 차단 (min_length=1 검증 우회 방지 + 명확한 안내)
    if not question:
        raise HTTPException(status_code=422, detail=_STT_FAILED_MESSAGE)

    # ChatRequest.question 의 max_length=1000 제약 대응: 긴 음성의 STT 결과가
    # 이를 넘으면 ValidationError(500)가 나므로 사전 절단한다
    # (save_chat_history 의 QUESTION 절단과 동일한 방어 정책).
    if len(question) > 1000:
        logger.warning("STT 인식 결과가 1000자를 초과해 절단합니다(원본 %d자)", len(question))
        question = question[:1000]

    # 2) 기존 파이프라인 재사용 (멀티턴/환각가드/인텐트 전부 자동)
    chat_request = ChatRequest(
        chat_token=chat_token or None,
        question=question,
        history=_parse_history_form(history),
        session_id=session_id,
    )
    chat_response = await process_chat_pipeline(chat_request)

    # 3) 답변 → TTS → base64
    try:
        audio_bytes = await voice_service.synthesize(chat_response.answer)
    except Exception:
        logger.exception("voice TTS 처리 실패")
        raise HTTPException(status_code=502, detail=_VOICE_ERROR_MESSAGE)

    audio_base64 = base64.b64encode(audio_bytes).decode("ascii")

    return VoiceChatResponse(
        question=question,
        answer=chat_response.answer,
        intent=chat_response.intent,
        confidence=chat_response.confidence,
        audio_base64=audio_base64,
        session_id=chat_response.session_id,
    )
