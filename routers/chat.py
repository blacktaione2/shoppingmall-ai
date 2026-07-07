"""
채팅 라우터: POST /chat/ask, POST /chat/stream

[LangGraph 전환]
- 기존 파이프라인을 graph.builder 의 컴파일된 StateGraph ainvoke 로 대체.

[멀티턴 하이브리드]
- 게스트: thread_id 없음 → 클라이언트가 history 전송 → config 없이 invoke (A안, 비영속)
- 로그인: thread_id = chat_token → 체크포인터(MemorySaver 또는 Redis)에서
    messages 조회 (B안)
    · 서버 메모리에 messages 있음        → messages → history 변환분 주입
    · 없음(첫 대화 / uvicorn 재시작 소실) → 클라이언트 폴백 history 로 seed
  invoke 시 config={"configurable":{"thread_id": chat_token}} 를 넘겨
  append_message_node 가 누적한 messages 가 thread 별로 보존된다.
- chat.py 책임:
  1) 게스트/로그인 판정 (chat_token → member_id)
  2) (로그인) 체크포인터 조회 + 폴백 seed
  3) 그래프 ainvoke (게스트: config 없음 / 로그인: config 있음)
  4) CHAT_HISTORY Oracle 저장 (로그인만, 기존 유지)
  5) ChatResponse 변환
"""
import logging
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from schemas.chat_schema import ChatRequest, ChatResponse
from schemas.intent_schema import IntentType, coerce_intent_result
from database.oracle_db import save_chat_history, resolve_chat_token
from pipeline.stream_util import event_stream
from graph.builder import build_graph
from graph.llm import messages_to_history, trim_history
from graph.observability import route_metadata
from graph.metrics import record_metrics, RequestMetrics, LatencyTimer
from graph.model_factory import get_provider, resolve_model_name, ModelRole
from services import idempotency

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

_PIPELINE_ERROR_MESSAGE = (
    "죄송합니다. 일시적인 오류로 답변을 생성하지 못했어요. 잠시 후 다시 시도해 주세요."
)
_INVALID_TOKEN_MESSAGE = "로그인이 만료되었거나 유효하지 않습니다. 다시 로그인해 주세요."
_DUPLICATE_REQUEST_MESSAGE = "이미 처리 중인 요청입니다. 잠시만 기다려 주세요."

# 컴파일된 그래프를 모듈 로드 시점에 만들지 않는다.
# 이유: main.py 가 'from routers import chat' 하는 순간(=lifespan 이전)
#       build_graph() 가 실행되면, 아직 체크포인터가 주입되지 않아(None)
#       안전망 MemorySaver 로 굳어버린다 → Redis 주입이 무효가 된다.
#       따라서 lifespan 에서 set_checkpointer() → build_graph() 가 끝난 뒤
#       첫 사용 시점에 가져오도록 lazy getter 로 바꾼다.
#       (정상 기동이면 lifespan 이 이미 _compiled_app 싱글톤을 만들어두므로
#        여기서는 그 싱글톤을 그대로 돌려받는다.)
_graph_app = None


def _get_graph():
    """컴파일된 그래프 싱글톤을 가져온다(없으면 이 시점에 컴파일).

    정상 경로에서는 lifespan 이 미리 build_graph() 를 호출해 _compiled_app 을
    만들어두므로, 여기서는 동일 싱글톤을 즉시 반환한다.
    """
    global _graph_app
    if _graph_app is None:
        _graph_app = build_graph()
    return _graph_app


async def _resolve_history(chat_token: str, client_history: list[dict]) -> list[dict]:
    """로그인 사용자의 대화 이력을 결정한다.

    우선순위:
      1) checkpointer(thread_id=chat_token)에 누적된 messages 가 있으면 그것을
         history dict 로 변환해 사용 (서버가 기억하는 정식 이력).
      2) 없으면(첫 대화 / uvicorn 재시작으로 메모리 소실) 클라이언트가 보낸
         폴백 history 를 그대로 사용 → 이후 append_message_node 가 다시 누적 시작.

    aget_state 는 해당 thread 가 한 번도 실행된 적 없으면 values 가 빈 dict 인
    StateSnapshot 을 반환한다(예외 아님). 따라서 .values.get("messages", []) 로
    안전하게 조회 가능하다.
    """
    config = {"configurable": {"thread_id": chat_token}}
    try:
        snapshot = await _get_graph().aget_state(config)
        server_messages = (snapshot.values or {}).get("messages", []) if snapshot else []
    except Exception:
        logger.exception("checkpointer 상태 조회 실패 → 클라이언트 폴백 사용")
        server_messages = []

    if server_messages:
        # 트리밍: 누적 messages 가 길어져도 LLM 입력은 최근 N개로 제한(윈도우 초과 방지).
        return trim_history(messages_to_history(server_messages))
    # 서버 메모리에 이력 없음 → 클라이언트 폴백(이미 max 10개로 제한되지만 일관성 위해 트림)
    return trim_history(client_history)


async def process_chat_pipeline(request: ChatRequest) -> ChatResponse:
    # -1) [멱등성] 요청 단위 UUID 중복 차단 (Redis SETNX, best-effort).
    #     /ask 는 409 JSON, /stream 은 event_stream 의 HTTPException 분기를 타고
    #     detail 이 그대로 error 이벤트로 노출된다 — 별도 처리 코드가 필요 없다.
    if not await idempotency.try_acquire(request.request_id):
        raise HTTPException(status_code=409, detail=_DUPLICATE_REQUEST_MESSAGE)

    # 0) 게스트/로그인 판정
    chat_token = request.chat_token
    is_guest = not chat_token
    member_id = None
    if not is_guest:
        member_id = await run_in_threadpool(resolve_chat_token, chat_token)
        if member_id is None:
            raise HTTPException(status_code=401, detail=_INVALID_TOKEN_MESSAGE)

    # 1) 그래프 입력 State + invoke 설정 구성
    client_history = [item.model_dump() for item in request.history]

    if is_guest:
        # [게스트 - A안] 클라이언트 history 직접 주입.
        # checkpointer 가 달린 그래프는 thread_id 없이 invoke 할 수 없으므로,
        # 매 요청마다 '1회성 UUID' thread_id 를 부여한다. 다음 요청은 다른 UUID 라
        # 서로 연결되지 않아 사실상 비영속(누적 없음)으로 동작한다.
        # (인메모리에 1회성 thread 가 잠깐 생기지만 재조회되지 않고, MemorySaver 라면
        #  프로세스 메모리라 재시작 시 정리된다. Redis 체크포인터라면 TTL 정책에 따라
        #  정리된다 — graph/checkpointer.py 참고.)
        init_state = {
            "question": request.question,
            "member_id": None,
            "is_guest": True,
            "history": client_history,
            "rag_hits": [],   # 체크포인트 잔존값 차단(스테일 sources 방지)
        }
        invoke_config = {
            "configurable": {"thread_id": f"guest-{uuid.uuid4()}"},
            **route_metadata("router_pipeline", is_guest=True),
        }
    else:
        # [로그인 - B안] checkpointer 조회 + 폴백 seed
        history = await _resolve_history(chat_token, client_history)
        init_state = {
            "question": request.question,
            "member_id": member_id,
            "is_guest": False,
            "history": history,
            # [스테일 방지] checkpointer 는 rag_hits 채널을 thread 단위로 영속하므로,
            # SEMANTIC 이 아닌 턴에는 직전 SEMANTIC 턴의 hits 가 그대로 복원된다.
            # 매 턴 빈 리스트로 리셋해 이번 턴 검색 결과만 남긴다(agent 라우터와 동일 정책).
            "rag_hits": [],
        }
        invoke_config = {
            "configurable": {"thread_id": chat_token},
            **route_metadata("router_pipeline", is_guest=False),
        }

    # 2) 그래프 실행 (분류 → 핸들러 → 가드 → 메시지 누적)
    #    게스트/로그인 모두 invoke_config 를 가진다(게스트는 1회성 UUID).
    try:
        with LatencyTimer() as _timer:
            result = await _get_graph().ainvoke(init_state, config=invoke_config)
        final_answer = result["final_answer"]
        # 체크포인트에는 dict 로 저장되므로 경계선에서 IntentResult 로 복원한다.
        intent_result = coerce_intent_result(result["intent_result"])
        intent_value = intent_result.intent.value
        confidence = intent_result.confidence
        rag_hits = result.get("rag_hits", [])
    except Exception:
        logger.exception("그래프 파이프라인 처리 실패: question=%s", request.question)
        return ChatResponse(
            answer=_PIPELINE_ERROR_MESSAGE,
            intent="SMALL_TALK",
            confidence=0.0,
        )

    # 라우터 그래프 경로 측정 기록.
    # 라우터 경로의 GPT 사용 노드(semantic/complaint/small_talk)는 모두 get_main_llm()/
    # get_intent_llm()/select_llm() 을 경유하므로 provider 가 일관되게 적용된다. 다만
    # 노드가 LCEL 체인 결과 텍스트만 raw_answer 로 넘기고 usage_metadata 를 State 밖으로
    # 올리지 않아, 토큰/비용은 여기서 0 으로 두고 LangSmith 트레이스로 보완한다
    # (PHASE3_BENCHMARK.md 방법론 참고).
    try:
        provider = get_provider()
        model_main = resolve_model_name(provider, ModelRole.MAIN)
        record_metrics(RequestMetrics(
            route="router_pipeline",
            provider=provider,
            model_main=model_main,
            intent=intent_value,
            is_guest=is_guest,
            latency_ms=_timer.elapsed_ms,
            # 토큰/비용은 라우터 경로에서 미측정(LangSmith 로 보완) → 0
        ))
    except Exception:
        logger.exception("router_pipeline 측정 기록 실패(무시)")

    # 3) 대화 이력 저장 (로그인 사용자만)
    if not is_guest:
        try:
            await run_in_threadpool(
                save_chat_history,
                member_id,
                request.question,
                final_answer,
                intent_value,
            )
        except Exception:
            logger.exception("CHAT_HISTORY 저장 실패")

    # SEMANTIC/STRUCTURED 검색 시에만 출처(sources) 부착 (그 외 인텐트는 None → 하위호환)
    # rag_hits 존재 여부만으로 판단하지 않고 인텐트를 함께 확인한다
    # (init_state 리셋과 함께 스테일 sources 에 대한 2중 방어).
    sources = None
    if rag_hits and intent_result.intent in (
        IntentType.SEMANTIC_SEARCH, IntentType.STRUCTURED_QUERY,
    ):
        from graph.rag_pipeline import hits_to_sources
        sources = hits_to_sources(rag_hits)

    return ChatResponse(
        answer=final_answer,
        intent=intent_value,
        confidence=confidence,
        sources=sources,
    )


@router.post("/ask", response_model=ChatResponse)
async def ask(request: ChatRequest) -> ChatResponse:
    return await process_chat_pipeline(request)


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        event_stream(process_chat_pipeline, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
