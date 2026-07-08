"""
routers/mcp_agent_chat.py
MCP 연동 Agent 엔드포인트 — POST /chat/mcp-agent

[역할]
- 단일 Agent 구조(graph.agent_builder)를 재사용하되, 로컬 도구에 더해
  MCP 외부 서버 도구를 함께 바인딩한 Agent 로 처리한다.
- 별도 경로라 기존 /chat/agent(단일), /chat/multi-agent(멀티)는 무영향.
  비교 시 'MCP 유무' 변수를 격리할 수 있다.

[안전장치]
- MCP 비활성/로드 실패 시 MCP 도구는 빈 리스트 → 로컬 도구만으로 정상 동작.
- MCP Agent 인스턴스는 첫 요청 시 1회 빌드(도구 로드가 async 라 lazy 빌드).

[멀티턴/메트릭]
- 멀티턴: 단일 Agent 와 동일 정책(thread_id, checkpointer 공유).
- 메트릭: ChatResponse 에 tool_calls/total_tokens/tools_used 는 채우되,
  record_metrics() 는 호출하지 않는다 → metrics.jsonl 미적재
  (PHASE 3 경로 비교는 라우터/단일/멀티 3개 경로로 한정 — MCP 유무 변수 격리).
  집계는 graph.metrics.collect_message_metrics 로 위임한다(단일 출처).
"""
import logging
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import HumanMessage, AIMessage

from schemas.chat_schema import ChatRequest, ChatResponse
from database.oracle_db import save_chat_history, resolve_chat_token
from graph.agent_builder import build_agent
from graph.mcp_tools import get_mcp_tools
from graph.metrics import (
    collect_message_metrics,
    snapshot_prior_message_ids,
    filter_new_messages,
)
from graph.llm import history_to_messages
from graph.observability import route_metadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["mcp-agent"])

_ERROR_MESSAGE = (
    "죄송합니다. 일시적인 오류로 답변을 생성하지 못했어요. 잠시 후 다시 시도해 주세요."
)
_INVALID_TOKEN_MESSAGE = "로그인이 만료되었거나 유효하지 않습니다. 다시 로그인해 주세요."
_RECURSION_LIMIT = 15

# MCP Agent 인스턴스 (lazy: 첫 요청 시 MCP 도구 로드 후 1회 빌드)
_mcp_agent_app = None


async def _get_mcp_agent():
    """MCP 도구를 합친 Agent 인스턴스를 1회 빌드해 재사용."""
    global _mcp_agent_app
    if _mcp_agent_app is None:
        mcp_tools = await get_mcp_tools()   # 비활성/실패 시 []
        logger.info("MCP Agent 빌드: MCP 도구 %d개 합류", len(mcp_tools))
        # force_rebuild=True 로 단일 Agent 싱글톤을 덮지 않는 독립 인스턴스 생성
        _mcp_agent_app = build_agent(extra_tools=mcp_tools, force_rebuild=True,
                                     include_refund=False)
    return _mcp_agent_app


def _collect_metrics(messages: list) -> tuple[int, int, list[str]]:
    """집계를 graph.metrics.collect_message_metrics 로 위임(중복 제거, 반환 계약 동일)."""
    return collect_message_metrics(messages)


def _extract_final_answer(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return _ERROR_MESSAGE


@router.post("/mcp-agent", response_model=ChatResponse)
async def mcp_agent_chat(request: ChatRequest) -> ChatResponse:
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
        client_history = [item.model_dump(exclude_none=True) for item in request.history]
        seed = history_to_messages(client_history)
        init_state = {
            "messages": seed + [new_message],
            "member_id": None, "is_guest": True, "rag_hits": [],
        }
        config = {
            "configurable": {"thread_id": f"mcp-guest-{uuid.uuid4()}"},
            "recursion_limit": _RECURSION_LIMIT,
            **route_metadata("mcp_agent", is_guest=True),
        }
    else:
        init_state = {
            "messages": [new_message],
            "member_id": member_id, "is_guest": False, "rag_hits": [],
        }
        config = {
            "configurable": {"thread_id": chat_token},
            "recursion_limit": _RECURSION_LIMIT,
            **route_metadata("mcp_agent", is_guest=False),
        }

    # 2) MCP Agent 실행 (MCP 도구 없으면 로컬 도구만으로 동작)
    try:
        app = await _get_mcp_agent()
        # [집계 기준선] 로그인 멀티턴 누적 이중 집계 방지 — 이번 턴 신규 메시지만 집계.
        prior_ids = await snapshot_prior_message_ids(app, config)
        result = await app.ainvoke(init_state, config=config)
        messages = result.get("messages", [])
        new_messages = filter_new_messages(messages, prior_ids)
        final_answer = _extract_final_answer(messages)
        tool_calls, total_tokens, tools_used = _collect_metrics(new_messages)
    except Exception:
        logger.exception("MCP Agent 처리 실패: question=%s", request.question)
        return ChatResponse(answer=_ERROR_MESSAGE, intent="MCP_AGENT", confidence=0.0)

    # 3) 이력 저장 (로그인만)
    if not is_guest:
        try:
            await run_in_threadpool(
                save_chat_history, member_id, request.question, final_answer, "MCP_AGENT",
            )
        except Exception:
            logger.exception("CHAT_HISTORY 저장 실패")

    return ChatResponse(
        answer=final_answer,
        intent="MCP_AGENT",
        confidence=1.0,
        tool_calls=tool_calls,
        total_tokens=total_tokens,
        tools_used=tools_used,
    )
