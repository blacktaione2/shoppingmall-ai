"""
routers/agent_chat.py
단일 Agent 엔드포인트 — POST /chat/agent

[역할]
- 기존 라우터 그래프(/chat/ask)와 '같은 입력(ChatRequest)·같은 멀티턴 정책'을 쓰되,
  처리는 Agent 그래프(graph.agent_builder)로 한다.
- '단일 파이프라인 vs 멀티 Agent 수치 비교'를 위해 응답에
  tool_calls / total_tokens / tools_used 메타데이터를 채운다(결정 ②).

[멀티턴]
- 로그인: thread_id = chat_token → checkpointer(라우터 그래프와 공유)로 맥락 유지.
- 게스트: 1회성 UUID thread_id + 클라이언트 history 를 첫 메시지로 주입(비영속).
  (Agent 는 messages 기반이라, 게스트 history 를 HumanMessage/AIMessage 로 시드한다.)

[비교 측정]
- Agent 실행 결과 messages 를 순회하며:
    · AIMessage.tool_calls 개수 합 → tool_calls
    · ToolMessage(name) 수집      → tools_used
    · AIMessage.usage_metadata.total_tokens 합 → total_tokens
"""
import logging
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from schemas.chat_schema import ChatRequest, ChatResponse, ResumeRequest
from database.oracle_db import save_chat_history, resolve_chat_token
from graph.agent_builder import build_agent
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

router = APIRouter(prefix="/chat", tags=["agent"])

_AGENT_ERROR_MESSAGE = (
    "죄송합니다. 일시적인 오류로 답변을 생성하지 못했어요. 잠시 후 다시 시도해 주세요."
)
_INVALID_TOKEN_MESSAGE = "로그인이 만료되었거나 유효하지 않습니다. 다시 로그인해 주세요."
_RECURSION_LIMIT = 12   # 무한 도구호출 방지 (agent⇄tools 왕복 상한)

# 컴파일된 Agent 그래프를 모듈 로드 시점에 만들지 않는다.
# 이유: main.py 가 'from routers import agent_chat' 하는 순간(=lifespan 이전)
#       build_agent() 가 실행되면, 아직 체크포인터가 주입되지 않아(None)
#       안전망 MemorySaver 로 굳어버린다 → lifespan 이 주입하는 정식 체크포인터
#       (Redis 또는 라우터 그래프와 공유하는 MemorySaver)가 무효가 된다.
#       라우터 그래프(chat.py)와 동일하게 첫 사용 시점까지 컴파일을 미뤄,
#       set_checkpointer() → build_graph() 가 끝난 뒤의 체크포인터를 공유받는다.
_agent_app = None


def _get_agent_app():
    """컴파일된 단일 Agent 그래프 싱글톤을 가져온다(없으면 이 시점에 컴파일).

    정상 경로에서는 lifespan 이 라우터 그래프를 먼저 컴파일하며 체크포인터를
    주입해두므로, 여기서 build_agent() 가 그 체크포인터를 공유받아
    /chat/ask 와 /chat/agent 가 같은 thread_id 로 멀티턴 맥락을 공유한다.
    """
    global _agent_app
    if _agent_app is None:
        _agent_app = build_agent()
    return _agent_app


def _collect_metrics(messages: list) -> tuple[int, int, list[str]]:
    """Agent 실행 messages 에서 비교 측정 메타데이터를 집계한다.

    중복 제거: 실제 집계는 graph.metrics.collect_message_metrics 로
    위임한다. 반환 계약(tool_calls, total_tokens, tools_used)은 그대로라 동작 불변.
    """
    return collect_message_metrics(messages)


def _extract_final_answer(messages: list) -> str:
    """마지막 '도구호출 없는 AIMessage'(최종 답변)를 추출."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return _AGENT_ERROR_MESSAGE


def _extract_interrupt(result: dict) -> dict | None:
    """ainvoke 결과에서 interrupt 정보를 표준 dict 로 추출(없으면 None).

    LangGraph 는 interrupt 발생 시 result["__interrupt__"] 에 Interrupt 객체
    튜플을 싣는다. 라우터는 객체 내부 표현에 의존하지 않도록 value/id 만 뽑아
    {"value": <payload>, "id": <interrupt_id>} 형태로 표준화한다.
    """
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if not interrupts:
        return None
    first = interrupts[0]
    return {
        "value": getattr(first, "value", None),
        "id": getattr(first, "id", None),
    }


@router.post("/agent", response_model=ChatResponse)
async def agent_chat(request: ChatRequest) -> ChatResponse:
    # 0) 게스트/로그인 판정
    chat_token = request.chat_token
    is_guest = not chat_token
    member_id = None
    if not is_guest:
        member_id = await run_in_threadpool(resolve_chat_token, chat_token)
        if member_id is None:
            raise HTTPException(status_code=401, detail=_INVALID_TOKEN_MESSAGE)

    # 1) 입력 messages + config 구성
    new_message = HumanMessage(content=request.question)
    if is_guest:
        # 게스트: 클라이언트 history 를 메시지로 시드 + 1회성 thread_id (비영속)
        client_history = [item.model_dump() for item in request.history]
        seed = history_to_messages(client_history)
        init_state = {
            "messages": seed + [new_message],
            "member_id": None,
            "is_guest": True,
            "rag_hits": [],
        }
        config = {
            "configurable": {"thread_id": f"agent-guest-{uuid.uuid4()}"},
            "recursion_limit": _RECURSION_LIMIT,
            **route_metadata("single_agent", is_guest=True),
        }
    else:
        # 로그인: checkpointer 가 messages 를 누적/복원 → 새 질문만 추가하면 됨
        init_state = {
            "messages": [new_message],
            "member_id": member_id,
            "is_guest": False,
            "rag_hits": [],
        }
        config = {
            "configurable": {"thread_id": chat_token},
            "recursion_limit": _RECURSION_LIMIT,
            **route_metadata("single_agent", is_guest=False),
        }

    # 2) Agent 실행
    try:
        app = _get_agent_app()
        # [집계 기준선] 로그인 멀티턴은 result 에 과거 턴 messages 가 함께 복원되므로,
        # invoke 전에 기존 메시지 id 를 떠두고 '이번 턴 신규분'만 집계한다
        # (없으면 토큰/도구호출/비용이 턴마다 누적 재합산되는 이중 집계 발생).
        prior_ids = await snapshot_prior_message_ids(app, config)
        with LatencyTimer() as _timer:
            result = await app.ainvoke(init_state, config=config)
        messages = result.get("messages", [])
        new_messages = filter_new_messages(messages, prior_ids)

        # ── [Human-in-the-loop] 환불 확인 대기 감지 ──────────────────────
        # request_refund 도구가 interrupt 를 걸면 result 에 __interrupt__ 가 실린다.
        # 이 경우 최종 답변 대신 '확인 대기' 응답을 내려보낸다(로그인 전용이라 is_guest=False).
        interrupt_obj = _extract_interrupt(result)
        if interrupt_obj is not None:
            payload = interrupt_obj.get("value") if isinstance(interrupt_obj, dict) else None
            return ChatResponse(
                answer=(payload or {}).get(
                    "prompt", "환불을 진행할까요? 승인 여부를 알려주세요."
                ),
                intent="AGENT",
                confidence=1.0,
                interrupt_pending=True,
                interrupt_payload=payload,
                resume_thread_id=chat_token,   # 재개 thread = chat_token (로그인 회원)
            )

        final_answer = _extract_final_answer(messages)
        tool_calls, total_tokens, tools_used = _collect_metrics(new_messages)
    except Exception:
        logger.exception("Agent 처리 실패: question=%s", request.question)
        return ChatResponse(
            answer=_AGENT_ERROR_MESSAGE, intent="AGENT", confidence=0.0,
        )

    # 측정 기록 (입·출력 토큰 구분 → 비용 환산 → JSONL append)
    # 측정 실패가 응답을 깨지 않도록 record_metrics 가 예외를 모두 삼킨다.
    try:
        prompt_tokens, completion_tokens = collect_token_breakdown(new_messages)
        provider = get_provider()
        model_main = resolve_model_name(provider, ModelRole.MAIN)
        cost_usd = estimate_cost(
            provider, model_main, prompt_tokens, completion_tokens,
        )
        record_metrics(RequestMetrics(
            route="single_agent",
            provider=provider,
            model_main=model_main,
            intent="AGENT",
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
        logger.exception("single_agent 측정 기록 실패(무시)")

    # 3) 이력 저장 (로그인만) — intent 는 Agent 경로 식별자로 'AGENT' 고정
    if not is_guest:
        try:
            await run_in_threadpool(
                save_chat_history, member_id, request.question, final_answer, "AGENT",
            )
        except Exception:
            logger.exception("CHAT_HISTORY 저장 실패")

    return ChatResponse(
        answer=final_answer,
        intent="AGENT",
        confidence=1.0,
        tool_calls=tool_calls,
        total_tokens=total_tokens,
        tools_used=tools_used,
    )


@router.post("/agent/resume", response_model=ChatResponse)
async def agent_resume(request: ResumeRequest) -> ChatResponse:
    """[Human-in-the-loop] 환불 확인(interrupt) 이후 Agent 그래프를 재개한다.

    /chat/agent 가 interrupt_pending=True 로 응답하면, 클라이언트는 사용자에게
    승인/거부를 물어 이 엔드포인트로 chat_token(=thread_id) + approved 를 보낸다.
    같은 thread 를 Command(resume=...) 로 재개해 도구가 이어서 실행되도록 한다.

    [재개 불가/만료 처리]
      · 토큰이 유효하지 않으면 401.
      · 재개할 interrupt 가 없으면(이미 처리됨/만료) 그래프가 즉시 끝나
        최종 답변이 비므로, 안내 문구로 폴백한다.
    """
    chat_token = request.chat_token
    member_id = await run_in_threadpool(resolve_chat_token, chat_token)
    if member_id is None:
        raise HTTPException(status_code=401, detail=_INVALID_TOKEN_MESSAGE)

    # 승인/거부를 도구의 interrupt() 반환값으로 전달 → request_refund 가 해석한다.
    resume_value = "approve" if request.approved else "reject"
    config = {
        "configurable": {"thread_id": chat_token},
        "recursion_limit": _RECURSION_LIMIT,
        **route_metadata("single_agent", is_guest=False),
    }

    try:
        app = _get_agent_app()
        # [집계 기준선] 재개 시점까지의 메시지(과거 턴 + 이번 턴 interrupt 이전분)를
        # 제외하고 재개 이후 신규분만 집계한다. interrupt 이전분은 본 호출이
        # interrupt_pending 으로 조기 반환해 원래도 미집계였으므로 회귀가 아니며,
        # 전체 이력을 재합산하던 이중 집계보다 정확하다.
        prior_ids = await snapshot_prior_message_ids(app, config)
        with LatencyTimer() as _timer:
            result = await app.ainvoke(Command(resume=resume_value), config=config)
        messages = result.get("messages", [])
        new_messages = filter_new_messages(messages, prior_ids)
        final_answer = _extract_final_answer(messages)
        tool_calls, total_tokens, tools_used = _collect_metrics(new_messages)
    except Exception:
        logger.exception("Agent 재개 실패: thread_id=%s", chat_token)
        return ChatResponse(answer=_AGENT_ERROR_MESSAGE, intent="AGENT", confidence=0.0)

    # 측정 기록 (본 호출과 동일 정책 — 실패해도 응답 불변)
    try:
        prompt_tokens, completion_tokens = collect_token_breakdown(new_messages)
        provider = get_provider()
        model_main = resolve_model_name(provider, ModelRole.MAIN)
        cost_usd = estimate_cost(provider, model_main, prompt_tokens, completion_tokens)
        record_metrics(RequestMetrics(
            route="single_agent",
            provider=provider,
            model_main=model_main,
            intent="AGENT",
            is_guest=False,
            latency_ms=_timer.elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls=tool_calls,
            tools_used=tools_used,
            cost_usd=cost_usd,
        ))
    except Exception:
        logger.exception("single_agent(resume) 측정 기록 실패(무시)")

    # 이력 저장 — 질문 자리에는 사용자 결정을 기록(추후 분석용)
    decision_text = "[환불 승인]" if request.approved else "[환불 취소]"
    try:
        await run_in_threadpool(
            save_chat_history, member_id, decision_text, final_answer, "AGENT",
        )
    except Exception:
        logger.exception("CHAT_HISTORY 저장 실패")

    return ChatResponse(
        answer=final_answer,
        intent="AGENT",
        confidence=1.0,
        tool_calls=tool_calls,
        total_tokens=total_tokens,
        tools_used=tools_used,
    )
