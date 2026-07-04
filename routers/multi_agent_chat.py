"""
routers/multi_agent_chat.py
멀티 Agent 엔드포인트 — POST /chat/multi-agent

[역할]
- 단일 Agent(/chat/agent)와 동일한 입력(ChatRequest)/멀티턴 정책을 쓰되,
  supervisor 오케스트레이터 그래프(graph.multi_agent_builder)로 처리.
- 비교용 메트릭을 ChatResponse 에 채운다:
    · tool_calls   : sub-agent 들이 호출한 도구 횟수 합
    · tools_used   : 호출된 도구 이름 목록
    · total_tokens : 누적 토큰(서브에이전트 + supervisor)
  (intent 필드는 경로 식별자 'MULTI_AGENT' 고정)

[멀티턴]
- 로그인: thread_id=chat_token, checkpointer 공유로 맥락 유지.
- 게스트: 1회성 UUID thread_id + 클라이언트 history 시드(비영속).
"""
import logging
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import HumanMessage

from schemas.chat_schema import ChatRequest, ChatResponse
from database.oracle_db import save_chat_history, resolve_chat_token
from graph.multi_agent_builder import build_multi_agent
from graph.llm import history_to_messages
from graph.observability import route_metadata
from graph.metrics import (
    collect_message_metrics,
    collect_token_breakdown,
    record_metrics,
    RequestMetrics,
    LatencyTimer,
    snapshot_prior_message_ids,
    filter_new_messages,
)
from graph.model_factory import get_provider, resolve_model_name, ModelRole
from services.pricing import estimate_cost

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["multi-agent"])

_ERROR_MESSAGE = (
    "죄송합니다. 일시적인 오류로 답변을 생성하지 못했어요. 잠시 후 다시 시도해 주세요."
)
_INVALID_TOKEN_MESSAGE = "로그인이 만료되었거나 유효하지 않습니다. 다시 로그인해 주세요."
_RECURSION_LIMIT = 20   # supervisor↔agent 왕복 + 도구호출 여유 상한

# 컴파일된 멀티 Agent 그래프를 모듈 로드 시점에 만들지 않는다.
# 이유: main.py 가 'from routers import multi_agent_chat' 하는 순간(=lifespan 이전)
#       build_multi_agent() 가 실행되면, 아직 체크포인터가 주입되지 않아(None)
#       안전망 MemorySaver 로 굳어버린다 → lifespan 이 주입하는 정식 체크포인터가
#       무효가 된다. 라우터 그래프(chat.py)와 동일하게 첫 사용 시점까지 컴파일을
#       미뤄, set_checkpointer() → build_graph() 가 끝난 뒤의 체크포인터를 공유받는다.
_multi_app = None


def _get_multi_app():
    """컴파일된 멀티 Agent 그래프 싱글톤을 가져온다(없으면 이 시점에 컴파일).

    정상 경로에서는 lifespan 이 라우터 그래프를 먼저 컴파일하며 체크포인터를
    주입해두므로, 여기서 build_multi_agent() 가 그 체크포인터를 공유받아
    라우터/단일 Agent 와 같은 thread_id 로 멀티턴 맥락을 공유한다.
    """
    global _multi_app
    if _multi_app is None:
        _multi_app = build_multi_agent()
    return _multi_app


def _collect_metrics(messages: list) -> tuple[int, int, list[str]]:
    """messages 에서 도구호출/토큰/도구이름 집계.

    중복 제거: graph.metrics.collect_message_metrics 로 위임.
    반환 계약(tool_calls, total_tokens, tools_used) 동일 → 동작 불변.
    """
    return collect_message_metrics(messages)


@router.post("/multi-agent", response_model=ChatResponse)
async def multi_agent_chat(request: ChatRequest) -> ChatResponse:
    # 0) 게스트/로그인 판정
    chat_token = request.chat_token
    is_guest = not chat_token
    member_id = None
    if not is_guest:
        member_id = await run_in_threadpool(resolve_chat_token, chat_token)
        if member_id is None:
            raise HTTPException(status_code=401, detail=_INVALID_TOKEN_MESSAGE)

    # 1) 입력 구성
    new_message = HumanMessage(content=request.question)
    if is_guest:
        client_history = [item.model_dump() for item in request.history]
        seed = history_to_messages(client_history)
        init_state = {
            "messages": seed + [new_message],
            "member_id": None, "is_guest": True,
            "rag_hits": [], "routing_count": 0,
        }
        config = {
            "configurable": {"thread_id": f"multi-guest-{uuid.uuid4()}"},
            "recursion_limit": _RECURSION_LIMIT,
            **route_metadata("multi_agent", is_guest=True),
        }
    else:
        init_state = {
            "messages": [new_message],
            "member_id": member_id, "is_guest": False,
            "rag_hits": [], "routing_count": 0,
        }
        config = {
            "configurable": {"thread_id": chat_token},
            "recursion_limit": _RECURSION_LIMIT,
            **route_metadata("multi_agent", is_guest=False),
        }

    # 2) 멀티 Agent 실행
    try:
        app = _get_multi_app()
        # [집계 기준선] 로그인 멀티턴 누적 이중 집계 방지 — 이번 턴 신규 메시지만 집계.
        prior_ids = await snapshot_prior_message_ids(app, config)
        with LatencyTimer() as _timer:
            result = await app.ainvoke(init_state, config=config)
        messages = result.get("messages", [])
        new_messages = filter_new_messages(messages, prior_ids)
        final_answer = result.get("final_answer") or _ERROR_MESSAGE
        tool_calls, total_tokens, tools_used = _collect_metrics(new_messages)
    except Exception:
        logger.exception("멀티 Agent 처리 실패: question=%s", request.question)
        return ChatResponse(answer=_ERROR_MESSAGE, intent="MULTI_AGENT", confidence=0.0)

    # 측정 기록 (입·출력 토큰 구분 → 비용 환산 → JSONL append)
    try:
        prompt_tokens, completion_tokens = collect_token_breakdown(new_messages)
        provider = get_provider()
        model_main = resolve_model_name(provider, ModelRole.MAIN)
        cost_usd = estimate_cost(
            provider, model_main, prompt_tokens, completion_tokens,
        )
        record_metrics(RequestMetrics(
            route="multi_agent",
            provider=provider,
            model_main=model_main,
            intent="MULTI_AGENT",
            is_guest=is_guest,
            latency_ms=_timer.elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls=tool_calls,
            tools_used=tools_used,
            cost_usd=cost_usd,
        ))
    except Exception:
        logger.exception("multi_agent 측정 기록 실패(무시)")

    # 3) 이력 저장 (로그인만)
    if not is_guest:
        try:
            await run_in_threadpool(
                save_chat_history, member_id, request.question, final_answer, "MULTI_AGENT",
            )
        except Exception:
            logger.exception("CHAT_HISTORY 저장 실패")

    return ChatResponse(
        answer=final_answer,
        intent="MULTI_AGENT",
        confidence=1.0,
        tool_calls=tool_calls,
        total_tokens=total_tokens,
        tools_used=tools_used,
    )
