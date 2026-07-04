"""
graph/multi_agent_builder.py
멀티 Agent(Supervisor 오케스트레이터) 그래프.

[구조]
    START → supervisor ─(라우팅)→ product_agent ─┐
                       ─(라우팅)→ support_agent ─┤
                       ─(FINISH)→ guard → END    │
              ▲                                   │
              └───────────────────────────────────┘
              (sub-agent 실행 후 supervisor 로 복귀해 재평가)

[역할]
- supervisor (get_intent_llm(), 구조화 출력 RouteDecision):
    · .env LLM_PROVIDER 에 따라 gpt-5.4-mini/gemini-3.1-flash/claude-.../
      deepseek-v4-flash 등으로 매핑되는 저비용 역할 모델(model_factory.py 공용).
    · 질문/누적 맥락을 보고 product_agent / support_agent / FINISH 중 선택.
    · FINISH 시 final_answer 를 직접 작성(인사/잡담/종합답변 모두 여기서).
- product_agent (create_react_agent): search_products + semantic_search
- support_agent (create_react_agent): search_faq + get_my_orders
- guard: rag_hits 가 있으면 SEMANTIC 환각검증, 없으면 pass-through (단일 Agent 와 동일).

[핑퐁 방지]
- routing_count 가 MAX_ROUTING 을 넘으면 supervisor 가 강제 FINISH.
- invoke 시 recursion_limit 도 함께 건다(이중 안전장치).

[checkpointer]
- 라우터/단일 Agent 그래프와 동일 인스턴스 공유 → thread_id 기반 멀티턴 맥락 공유.
"""
import logging

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from graph.multi_agent_state import MultiAgentState, RouteDecision
from graph.tools import search_products, semantic_search, search_faq, get_my_orders
from graph.llm import (
    get_main_llm, get_intent_llm, date_system_prefix,
    trim_message_list, agent_pre_model_hook,
)
from graph.builder import get_checkpointer
from graph.agent_state import AgentState

logger = logging.getLogger(__name__)

_compiled_multi = None

MAX_ROUTING = 4   # supervisor 라우팅 횟수 상한 (핑퐁 방지)

_SUPERVISOR_SYSTEM_PROMPT = (
    "당신은 한국어 쇼핑몰 고객지원 팀의 총괄 supervisor 입니다. "
    "사용자 요청을 분석해 적절한 전문 에이전트로 작업을 배분하세요.\n"
    "- product_agent: 상품 검색·추천(가격/카테고리 조건 검색, 의미 기반 추천)\n"
    "- support_agent: FAQ(배송/교환/환불 정책)와 주문 조회\n"
    "- FINISH: 더 이상 도구가 필요 없을 때. 인사·잡담이거나, 에이전트들이 "
    "이미 필요한 정보를 모았다면 FINISH 를 선택하고 final_answer 에 "
    "사용자에게 전달할 최종 답변을 한국어로 작성하세요.\n"
    "복합 요청이면 한 에이전트를 먼저 호출하고, 결과를 본 뒤 다음 에이전트를 "
    "호출할 수 있습니다. 같은 에이전트를 불필요하게 반복 호출하지 마세요."
)


# ════════════════════════════════════════════════════════════════════════
# supervisor 노드
# ════════════════════════════════════════════════════════════════════════
async def _supervisor_node(state: MultiAgentState) -> dict:
    """라우팅 결정 + (FINISH 시) 최종 답변 작성."""
    routing_count = state.get("routing_count", 0)

    # 핑퐁 방지: 상한 도달 시 강제 종료(현재까지 맥락으로 답변 생성)
    if routing_count >= MAX_ROUTING:
        logger.warning("supervisor 라우팅 상한(%d) 도달 → 강제 FINISH", MAX_ROUTING)
        forced = await _force_finish_answer(state)
        return {"next_agent": "FINISH", "final_answer": forced}

    system = f"{date_system_prefix()}\n\n{_SUPERVISOR_SYSTEM_PROMPT}"
    llm = get_intent_llm(temperature=0.0).with_structured_output(RouteDecision)
    # 트리밍: supervisor 도 누적 messages 를 최근 N개로 제한(윈도우 초과 방지).
    messages = [SystemMessage(content=system)] + trim_message_list(state["messages"])

    try:
        decision: RouteDecision = await llm.ainvoke(messages)
    except Exception:
        logger.exception("supervisor 라우팅 실패 → FINISH 폴백")
        return {
            "next_agent": "FINISH",
            "final_answer": "죄송합니다. 요청을 처리하지 못했어요. 다시 말씀해 주세요.",
        }

    update = {
        "next_agent": decision.next,
        "routing_count": routing_count + 1,
    }
    if decision.next == "FINISH":
        update["final_answer"] = decision.final_answer or "무엇을 도와드릴까요?"
    return update


async def _force_finish_answer(state: MultiAgentState) -> str:
    """라우팅 상한 도달 시, 누적 맥락으로 최종 답변만 생성."""
    system = (
        f"{date_system_prefix()}\n\n"
        "지금까지 수집한 정보를 바탕으로 사용자 질문에 한국어로 간결히 답변하세요."
    )
    llm = get_main_llm(temperature=0.3)
    messages = [SystemMessage(content=system)] + trim_message_list(state["messages"])
    try:
        resp = await llm.ainvoke(messages)
        return (resp.content or "").strip() or "요청을 처리했어요."
    except Exception:
        logger.exception("강제 FINISH 답변 생성 실패")
        return "죄송합니다. 요청 처리 중 문제가 발생했어요."


def _route_from_supervisor(state: MultiAgentState) -> str:
    """supervisor 결정에 따라 다음 노드로 분기."""
    nxt = state.get("next_agent", "FINISH")
    if nxt == "product_agent":
        return "product_agent"
    if nxt == "support_agent":
        return "support_agent"
    return "guard"   # FINISH → 환각가드 후 종료


# ════════════════════════════════════════════════════════════════════════
# sub-agent 래퍼 (create_react_agent → supervisor 복귀)
# ════════════════════════════════════════════════════════════════════════
def _make_sub_agent(model, tools, prompt_text: str):
    """도구 subset 을 가진 ReAct sub-agent 생성."""
    def _prompt(state):
        system = f"{date_system_prefix()}\n\n{prompt_text}"
        return [SystemMessage(content=system)] + state["messages"]

    return create_react_agent(
        model=model,
        tools=tools,
        state_schema=AgentState,   # sub-agent 는 messages/member_id/is_guest/rag_hits 사용
        prompt=_prompt,
        pre_model_hook=agent_pre_model_hook,   # LLM 호출 직전 트리밍(원본 비파괴)
    )


_PRODUCT_AGENT_PROMPT = (
    "당신은 상품 탐색 전문 에이전트입니다. 도구를 사용해 사용자에게 맞는 상품을 "
    "찾아 제시하세요. 도구 결과에 없는 가격·재고·상품명을 지어내지 마세요."
)
_SUPPORT_AGENT_PROMPT = (
    "당신은 고객지원 전문 에이전트입니다. FAQ(배송/교환/환불)와 주문 조회 도구를 "
    "사용해 정확히 안내하세요. 회원 식별은 시스템이 처리하므로 회원번호를 묻지 마세요."
)


# ════════════════════════════════════════════════════════════════════════
# guard 노드 (단일 Agent 의 가드 로직 재사용)
# ════════════════════════════════════════════════════════════════════════
async def _guard_node(state: MultiAgentState) -> dict:
    """supervisor 의 final_answer 에 환각 가드 적용 (rag_hits 있을 때만)."""
    from schemas.intent_schema import IntentResult, IntentType
    from graph.guard import guard_answer_state

    answer = state.get("final_answer", "")
    if not answer:
        return {}
    rag_hits = state.get("rag_hits", [])
    intent = IntentType.SEMANTIC_SEARCH if rag_hits else IntentType.SMALL_TALK
    dummy = IntentResult(intent=intent, confidence=1.0)

    final = await guard_answer_state(
        question="", answer=answer, intent_result=dummy,
        rag_hits=rag_hits, history=[],
    )
    return {"final_answer": final}


# ════════════════════════════════════════════════════════════════════════
# 그래프 조립
# ════════════════════════════════════════════════════════════════════════
def build_multi_agent(model=None, supervisor_model=None,
                      checkpointer=None, force_rebuild: bool = False):
    """멀티 Agent 그래프를 조립해 컴파일된 실행 객체를 반환(싱글톤).

    Args:
        model: sub-agent 용 ChatModel (테스트 주입용). None 이면 gpt-5.4.
        supervisor_model: supervisor 라우팅용. (현재 노드가 get_intent_llm 을
            직접 쓰므로 이 파라미터는 sub-agent 와의 대칭을 위한 예약 슬롯)
        checkpointer: 테스트용 독립 체크포인터.
        force_rebuild: 싱글톤 무시하고 재컴파일(테스트용).
    """
    global _compiled_multi
    if _compiled_multi is not None and not force_rebuild:
        return _compiled_multi

    sub_model = model if model is not None else get_main_llm(temperature=0.3)

    product_agent = _make_sub_agent(
        sub_model, [search_products, semantic_search], _PRODUCT_AGENT_PROMPT,
    )
    support_agent = _make_sub_agent(
        sub_model, [search_faq, get_my_orders], _SUPPORT_AGENT_PROMPT,
    )

    builder = StateGraph(MultiAgentState)
    builder.add_node("supervisor", _supervisor_node)
    builder.add_node("product_agent", product_agent)
    builder.add_node("support_agent", support_agent)
    builder.add_node("guard", _guard_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "product_agent": "product_agent",
            "support_agent": "support_agent",
            "guard": "guard",
        },
    )
    # sub-agent 실행 후 supervisor 로 복귀(재평가)
    builder.add_edge("product_agent", "supervisor")
    builder.add_edge("support_agent", "supervisor")
    builder.add_edge("guard", END)

    cp = checkpointer if checkpointer is not None else get_checkpointer()
    compiled = builder.compile(checkpointer=cp)
    if not force_rebuild:
        _compiled_multi = compiled
    return compiled
