"""
채팅 API 요청/응답 스키마
"""
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class HistoryItem(BaseModel):
    """대화 이력 단건 (role: user | bot)"""
    role: str = Field(..., description="'user' 또는 'bot'")
    text: str = Field(..., max_length=2000)
    # [상품 카드 유지] bot 메시지에 딸린 검색 근거 상품(썸네일/상세링크용).
    # GET /chat/sessions/{id}/messages 로 과거 대화를 불러올 때만 채워짐
    # (클라이언트가 보내는 요청 history 에는 보통 없음 → 기본 None).
    sources: Optional[List["SourceItem"]] = Field(None, description="검색 근거 상품 목록(있는 경우만)")


class ChatRequest(BaseModel):
    chat_token: Optional[str] = Field(
        None, max_length=36,
        description="로그인 시 발급된 CHAT_TOKEN(UUID). 게스트는 생략(None)",
    )
    question: str = Field(..., min_length=1, max_length=1000)
    # [멱등성] 요청 단위 UUID — 전송마다 클라이언트가 crypto.randomUUID() 로 발급.
    # 세션 식별자인 chat_token 과 역할이 다르다(services/idempotency.py 참고).
    # 옵션 필드라 미전송(구버전 클라이언트/외부 호출) 시 기존과 동일하게 동작한다.
    request_id: Optional[str] = Field(
        None, max_length=36,
        description="요청 멱등키(UUID). 같은 값의 재전송은 5분간 중복으로 차단.",
    )
    # 클라이언트가 sessionStorage 에서 읽어 전달하는 직전 대화 이력 (최대 10턴)
    history: List[HistoryItem] = Field(
        default_factory=list,
        max_length=10,
        description="직전 대화 이력. 오래된 순서대로 전달.",
    )
    # [다중 세션] 로그인 회원의 대화방(Thread) 식별자. LangGraph thread_id 로 쓰인다.
    # 생략하면 서버가 새 대화방을 만들어 사용한다(응답의 session_id 로 확인).
    session_id: Optional[str] = Field(
        None, max_length=36,
        description="로그인 회원의 대화방 ID(UUID). 생략 시 서버가 새로 생성.",
    )

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        """[개선] 공백/개행만으로 이뤄진 질문 차단.

        min_length=1 은 공백 1글자(" ")를 통과시킨다. 공백만 들어오면
        인텐트 분류 LLM 을 헛호출(토큰/비용 낭비)하므로 여기서 막는다.
        앞뒤 공백은 제거해 다운스트림(임베딩/분류)에 깔끔한 텍스트를 넘긴다.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("질문이 비어 있습니다.")
        return stripped


class ChatResponse(BaseModel):
    answer: str
    intent: str
    confidence: float = 0.0
    # ── [단일 Agent] 비교 측정용 메타데이터 (Agent 경로에서만 채움) ──
    # 기존 /chat/ask·/chat/stream 응답은 None 으로 남아 하위호환을 유지한다.
    # '단일 파이프라인 vs 멀티 Agent 수치 비교' 문서화의 입력 데이터가 된다.
    tool_calls: Optional[int] = Field(
        None, description="Agent 가 호출한 도구 횟수 (Agent 경로 전용)")
    total_tokens: Optional[int] = Field(
        None, description="해당 요청의 누적 토큰 사용량 (Agent 경로 전용)")
    tools_used: Optional[List[str]] = Field(
        None, description="호출된 도구 이름 목록 (Agent 경로 전용)")
    # ── [RAG 고도화] 출처 표시 (SEMANTIC 검색 시에만 채움) ──────────────
    # 프론트가 상품 카드/링크를 렌더할 수 있도록 검색 근거 상품을 구조화해 전달.
    # SEMANTIC 이 아닌 인텐트는 None → 하위호환 유지.
    sources: Optional[List["SourceItem"]] = Field(
        None, description="SEMANTIC 검색 근거 상품 목록 (출처 표시)")
    # ── [Human-in-the-loop] 민감 작업(환불) 확인 대기 상태 ────────────────
    # Agent 가 request_refund 도구에서 interrupt 를 걸면 채워진다.
    # 평상시 응답은 None → 하위호환 유지.
    #   · interrupt_pending=True 면 클라이언트는 사용자에게 승인/거부를 물어
    #     POST /chat/agent/resume 로 thread_id + decision 을 보낸다.
    interrupt_pending: Optional[bool] = Field(
        None, description="민감 작업 사용자 확인 대기 여부(Human-in-the-loop)")
    interrupt_payload: Optional[dict] = Field(
        None, description="확인 요청 상세(order_id, prompt 등). interrupt_pending=True 일 때만.")
    resume_thread_id: Optional[str] = Field(
        None, description="재개에 사용할 thread_id. /chat/agent/resume 호출 시 그대로 전달.")
    # [다중 세션] 이번 응답이 사용된 대화방 ID. 로그인 사용자만 채워짐(게스트는 None).
    # 클라이언트가 session_id 없이 요청했다면 서버가 새로 만든 값이 여기로 온다.
    session_id: Optional[str] = Field(
        None, description="이번 요청이 사용된 대화방 ID(로그인 사용자만, 신규 생성분 포함).")


class SourceItem(BaseModel):
    """RAG 출처(검색 근거 상품) 1건.

    score 는 '높을수록 관련'으로 통일 (재랭킹 시 relevance, 미적용 시 distance 정규화).
    """
    product_id: Optional[int] = Field(None, description="상품 PK")
    product_name: str = Field("", description="상품명")
    category: str = Field("", description="카테고리")
    price: Optional[float] = Field(None, description="가격(원)")
    # 프론트 상품 카드 썸네일용 이미지 URL(없으면 빈 문자열)
    image_url: str = Field("", description="상품 이미지 URL")
    score: float = Field(0.0, description="관련도 점수(높을수록 관련)")


class ChatSessionItem(BaseModel):
    """대화방 목록 1건 (GET /chat/sessions, POST /chat/sessions 응답)"""
    session_id: str
    title: str
    updated_at: str


class ChatSessionListResponse(BaseModel):
    sessions: List[ChatSessionItem]


class ChatSessionMessagesResponse(BaseModel):
    messages: List[HistoryItem]


class FaqItem(BaseModel):
    faq_id: int
    question: str
    answer: str
    category: Optional[str] = None


class ResumeRequest(BaseModel):
    """[Human-in-the-loop] 민감 작업(환불) 확인 응답 (POST /chat/agent/resume).

    interrupt 로 멈춘 Agent 그래프를 같은 thread 에서 재개한다.
    """
    chat_token: str = Field(
        ..., max_length=36,
        description="환불 확인을 요청한 로그인 회원의 CHAT_TOKEN(=재개할 thread_id).",
    )
    approved: bool = Field(..., description="True=환불 승인, False=취소")

# ── 음성(STT/TTS) 응답 스키마 ──────────────────────────────
class TranscribeResponse(BaseModel):
    """STT 단독 결과 (POST /chat/transcribe)"""
    text: str = Field(..., description="음성에서 인식된 텍스트")


class TtsRequest(BaseModel):
    """TTS 단독 요청 (POST /chat/tts)"""
    text: str = Field(..., min_length=1, max_length=4096,
                      description="음성으로 합성할 텍스트")


class VoiceChatResponse(BaseModel):
    """음성 대화 결과 (POST /chat/voice)

    음성 인식 텍스트 + 파이프라인 답변 + 합성 음성(base64)을 한 번에 제공.
    프론트는 question/answer 를 화면에 표시하고 audio_base64 를 재생한다.
    """
    question: str = Field(..., description="음성에서 인식된 사용자 질문(STT 결과)")
    answer: str = Field(..., description="파이프라인이 생성한 텍스트 답변")
    intent: str = Field(..., description="분류된 인텐트")
    confidence: float = Field(0.0, description="인텐트 분류 확신도")
    audio_base64: str = Field(..., description="답변 음성(mp3) base64 인코딩 문자열")
    session_id: Optional[str] = Field(
        None, description="이번 요청이 사용된 대화방 ID(로그인 사용자만).")
    session_id: Optional[str] = Field(
        None, description="이번 요청이 사용된 대화방 ID(로그인 사용자만).")
