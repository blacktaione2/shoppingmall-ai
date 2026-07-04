"""
services/voice_service.py
음성 처리(STT/TTS) 전담 모듈.

[구성]
- STT(Speech-To-Text): OpenAI Whisper(whisper-1) 로 음성 → 텍스트
- TTS(Text-To-Speech): OpenAI TTS(tts-1) 로 텍스트 → mp3 bytes
- clean_text_for_tts: 합성 전 이모지/마크다운 기호 제거 (자연스러운 음성)

[설계 메모]
- AsyncOpenAI 네이티브 비동기 사용 (네트워크 I/O → to_thread 불필요).
- 모델/보이스명은 상수로 분리 → 추후 OpenAI 의 gpt-4o-transcribe / gpt-4o-mini-tts 로
  교체 시 이 상수만 수정 (다른 모듈과 동일 패턴).
- gpt_service 의 get_client() 와 별도 클라이언트를 쓰지 않고, 동일한 lazy
  싱글톤 패턴으로 음성 전용 클라이언트를 둔다(역할 분리 + 단일 책임).
"""
import os
import re

from dotenv import load_dotenv
from openai import AsyncOpenAI
from langsmith import traceable

load_dotenv()

# ── 모델/보이스 상수 (교체 지점) ──────────────────────────────────────
STT_MODEL = "whisper-1"          # 음성 인식 모델 (안정 버전)
TTS_MODEL = "tts-1"              # 음성 합성 모델 (저지연 버전; 고품질은 tts-1-hd)
TTS_VOICE = "nova"               # 합성 보이스 (alloy/echo/fable/onyx/nova/shimmer)
TTS_FORMAT = "mp3"               # 출력 오디오 포맷

# ── 입력 제약 (OpenAI 한계 기반) ──────────────────────────────────────
MAX_AUDIO_BYTES = 25 * 1024 * 1024   # Whisper 업로드 상한 25MB
MAX_TTS_CHARS = 4096                  # TTS 입력 텍스트 상한 4096자
# 허용 오디오 MIME (브라우저 MediaRecorder webm 포함)
ALLOWED_AUDIO_MIME = {
    "audio/webm", "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/m4a", "audio/x-m4a", "audio/mp4", "audio/ogg",
}

_voice_client: AsyncOpenAI | None = None


class AudioValidationError(ValueError):
    """오디오 업로드 검증 실패 (라우터가 400 으로 변환)."""


def validate_audio_upload(file_bytes: bytes, content_type: str | None) -> None:
    """업로드된 오디오의 크기/MIME 을 검증한다. 실패 시 AudioValidationError.

    [검증 항목]
      1) 빈 파일(0바이트) 차단.
      2) MAX_AUDIO_BYTES(25MB) 초과 차단 — Whisper 업로드 상한.
      3) content_type 이 주어졌고 ALLOWED_AUDIO_MIME 에 없으면 차단.
         · 브라우저 MediaRecorder 는 'audio/webm;codecs=opus' 처럼 파라미터를
           붙여 보내므로, ';' 앞부분만 떼어 비교한다.
         · content_type 이 None/빈 값이면 MIME 검사는 건너뛴다(확장자 기반 처리에 위임).
    """
    if not file_bytes:
        raise AudioValidationError("빈 오디오 파일입니다. 다시 녹음해 주세요.")
    if len(file_bytes) > MAX_AUDIO_BYTES:
        raise AudioValidationError(
            f"오디오 파일이 너무 큽니다(최대 {MAX_AUDIO_BYTES // (1024 * 1024)}MB)."
        )
    if content_type:
        base_mime = content_type.split(";", 1)[0].strip().lower()
        if base_mime and base_mime not in ALLOWED_AUDIO_MIME:
            raise AudioValidationError(f"지원하지 않는 오디오 형식입니다: {base_mime}")


def _openai_timeout() -> float | None:
    """LLM_TIMEOUT(초)을 읽어 OpenAI SDK timeout 으로 사용(0 이하/오류면 None=무제한).

    gpt_service 와 동일 규칙(순환 import 회피용 로컬 정의).
    STT/TTS 는 오디오 업로드/합성이 포함돼 채팅보다 느릴 수 있으니, 타임아웃을
    더 늘려야 하면 .env 의 LLM_TIMEOUT 을 올리면 된다(0/빈 값이면 무제한).
    """
    raw = os.getenv("LLM_TIMEOUT", "30")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 30.0
    return val if val > 0 else None


def _get_client() -> AsyncOpenAI:
    global _voice_client
    if _voice_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 가 .env 에 설정되어 있지 않습니다.")
        # HTTP 요청 타임아웃(.env LLM_TIMEOUT, 기본 30초). STT/TTS 무한 대기 방지.
        _voice_client = AsyncOpenAI(api_key=api_key, timeout=_openai_timeout())
    return _voice_client


# ── 합성 전 텍스트 정제 ────────────────────────────────────────────────
# 이모지/픽토그램 유니코드 영역 (TTS 가 어색하게 읽거나 무시하는 기호 제거용)
_EMOJI_PATTERN = re.compile(
    "[" 
    "\U0001F300-\U0001FAFF"   # 기타 픽토그램/이모지 확장
    "\U00002600-\U000027BF"   # 기타 기호/딩벳
    "\U0001F000-\U0001F0FF"   # 마작/도미노 등
    "\U00002190-\U000021FF"   # 화살표
    "\U0000FE00-\U0000FE0F"   # 변이 선택자
    "]+",
    flags=re.UNICODE,
)


def clean_text_for_tts(text: str) -> str:
    """TTS 합성 전 텍스트 정제.

    - 마크다운 강조 기호(**, __, ` ) 제거 → 'aaa' 가 '별표별표aaa' 로 안 읽히게.
    - 이모지/픽토그램 제거.
    - 연속 공백/개행 정리.
    빈 결과면 안내 문구로 대체(빈 입력으로 TTS 호출 시 400 방지).
    """
    if not text:
        return "답변을 음성으로 변환할 내용이 없습니다."
    cleaned = text
    # 마크다운 강조/코드 기호 제거
    cleaned = cleaned.replace("**", "").replace("__", "").replace("`", "")
    # 이모지/픽토그램 제거
    cleaned = _EMOJI_PATTERN.sub("", cleaned)
    # 개행 → 공백, 연속 공백 1칸으로
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "답변을 음성으로 변환할 내용이 없습니다."
    # TTS 입력 상한 초과 시 절단 (앞부분 우선)
    if len(cleaned) > MAX_TTS_CHARS:
        cleaned = cleaned[:MAX_TTS_CHARS]
    return cleaned


# ── STT ────────────────────────────────────────────────────────────────
@traceable(run_type="tool", name="voice_service.transcribe")
async def transcribe(file_bytes: bytes, filename: str) -> str:
    """음성 바이트 → 인식 텍스트.

    Args:
        file_bytes : 업로드된 오디오 원본 바이트
        filename   : 원본 파일명(확장자 포함). OpenAI SDK 가 포맷 판별에 사용하므로
                     반드시 확장자가 있는 이름을 넘긴다(예: "voice.webm").
    Returns:
        인식된 텍스트(앞뒤 공백 제거). 무음/인식 실패 시 빈 문자열 가능.
    """
    client = _get_client()
    # OpenAI SDK 는 (filename, bytes) 튜플 형태의 file 인자를 받는다.
    resp = await client.audio.transcriptions.create(
        model=STT_MODEL,
        file=(filename, file_bytes),
        language="ko",   # 한국어 고정(쇼핑몰 사용자) → 인식 정확도/속도 향상
    )
    return (resp.text or "").strip()


# ── TTS ────────────────────────────────────────────────────────────────
@traceable(run_type="tool", name="voice_service.synthesize")
async def synthesize(text: str) -> bytes:
    """텍스트 → mp3 오디오 바이트.

    내부에서 clean_text_for_tts 로 정제 후 합성한다.
    """
    cleaned = clean_text_for_tts(text)
    client = _get_client()
    # streaming response 로 받아 전체 바이트를 모은다(엔드포인트가 base64/스트림으로 재가공).
    resp = await client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=cleaned,
        response_format=TTS_FORMAT,
    )
    # resp.read() 로 전체 바이트 확보 (httpx Response 래퍼)
    return resp.read()
