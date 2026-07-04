"""
graph/agent_builder.py
단일 Agent 그래프 조립.

[구조]
    START
      ↓
    agent (create_react_agent: get_main_llm() + 도구 바인딩, 도구 호출 루프 내장)
      ↓
    guard (semantic 도구가 기록한 rag_hits 기반 환각검증)
      ↓
    END

[설계]
- create_react_agent 가 'agent ⇄ tools' 루프를 내부적으로 처리한다.
  · LLM 은 get_main_llm() 이 반환하는 인스턴스로, .env 의 LLM_PROVIDER 에 따라
    gpt-5.4 / gemini-3.1-flash / claude-sonnet-4-6 / deepseek-v4-flash 로 매핑된다
    (provider 고정 아님 — model_factory.py 멀티모델 팩토리 재사용).
  · LLM 이 도구를 자율 선택/연쇄 호출하고, 더 호출할 도구가 없으면 최종 답변 생성.
- 그 뒤에 우리 guard 노드를 직렬로 붙여, semantic 도구가 호출된 경우에만
  rag_hits 기반 환각검증을 적용한다(결정 ①: semantic 호출 시에만 선택 적용).
- checkpointer 는 라우터 그래프와 '같은 인스턴스'를 공유한다.
  → /chat/ask 와 /chat/agent 가 같은 thread_id(chat_token)면 멀티턴 맥락을 공유.
- recursion_limit 은 invoke 시 config 로 전달한다(무한 도구호출 방지).
"""
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from graph.agent_state import AgentState
from graph.tools import ALL_TOOLS
from graph.llm import get_main_llm, date_system_prefix, agent_pre_model_hook
from graph.builder import get_checkpointer

logger = logging.getLogger(__name__)

_compiled_agent = None

# Agent 시스템 프롬프트 (한국어 쇼핑몰 도우미 + 도구 사용 규칙)
_AGENT_SYSTEM_PROMPT = (
    "당신은 한국어 온라인 쇼핑몰의 AI 도우미입니다. "
    "사용자 질문을 해결하기 위해 제공된 도구를 적절히 선택해 사용하세요. "
    "복합 질문이면 여러 도구를 순차적으로 사용해도 됩니다. "
    "도구 결과에 없는 가격·재고·상품명을 지어내지 마세요. "
    "주문 조회는 get_my_orders 도구를 쓰고, 회원 식별은 시스템이 처리하므로 "
    "사용자에게 회원번호를 묻지 마세요. "
    "환불 요청은 request_refund 도구를 쓰세요. 이 도구는 시스템이 사용자에게 "
    "최종 확인을 받으므로, 당신이 임의로 환불을 확정하거나 확인 질문을 만들지 마세요. "
    "답변은 친절하고 간결한 한국어로 작성하세요."
)


def _agent_prompt(state: AgentState):
    """create_react_agent 의 prompt 콜백.

    매 호출마다 날짜 컨텍스트를 시스템 메시지로 앞에 붙이고, 누적 messages 를 잇는다.
    (date_system_prefix 로 '오늘 날짜/계절'을 주입 → 계절 상품 추천 정확도 향상)
    """
    from langchain_core.messages import SystemMessage
    system = f"{date_system_prefix()}\n\n{_AGENT_SYSTEM_PROMPT}"
    return [SystemMessage(content=system)] + state["messages"]


async def _guard_node(state: AgentState) -> dict:
    """Agent 최종 답변에 환각 가드 적용 (semantic 호출 시에만 rag_hits 검증).

    - 마지막 AIMessage(도구 호출이 끝난 최종 답변)를 꺼내 검증한다.
    - rag_hits 가 있으면(=semantic 도구가 호출됨) SEMANTIC 가드 로직 적용.
      없으면 pass-through (결정 ①).
    - 검증 후 답변이 바뀌면 마지막 메시지를 교체한다.
    """
    from langchain_core.messages import AIMessage
    from schemas.intent_schema import IntentResult, IntentType
    from graph.guard import guard_answer_state

    messages = state.get("messages", [])
    # 마지막 AIMessage(최종 답변) 탐색
    last_ai = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            last_ai = msg
            break
    if last_ai is None:
        return {}

    answer = last_ai.content if isinstance(last_ai.content, str) else str(last_ai.content)
    rag_hits = state.get("rag_hits", [])

    # rag_hits 유무로 인텐트를 결정: 있으면 SEMANTIC 가드, 없으면 pass-through.
    intent = IntentType.SEMANTIC_SEARCH if rag_hits else IntentType.SMALL_TALK
    dummy_intent = IntentResult(intent=intent, confidence=1.0)

    final = await guard_answer_state(
        question="",            # Agent 가드는 답변+hits 기반이라 질문 텍스트 불필요
        answer=answer,
        intent_result=dummy_intent,
        rag_hits=rag_hits,
        history=[],
    )
    if final == answer:
        return {}
    # 가드가 답변을 바꾼 경우: 마지막 메시지 id 를 유지한 채 내용 교체
    return {"messages": [AIMessage(content=final, id=last_ai.id)]}


def build_agent(model=None, checkpointer=None, force_rebuild: bool = False,
                extra_tools=None, include_refund: bool = True):
    """단일 Agent 그래프를 조립해 컴파일된 실행 객체를 반환(싱글톤).

    Args:
        model: 테스트에서 가짜 ChatModel 을 주입하기 위한 파라미터.
               None 이면 운영 기본값(get_main_llm() → .env LLM_PROVIDER 에 따라
               gpt-5.4/gemini-3.1-flash/claude-sonnet-4-6/deepseek-v4-flash 중 하나)을 사용한다.
        checkpointer: 테스트에서 독립 checkpointer 를 주입하기 위한 파라미터.
               None 이면 라우터 그래프와 공유하는 기본 checkpointer 를 사용한다.
        force_rebuild: True 면 싱글톤 캐시를 무시하고 새로 컴파일(테스트용).
        extra_tools: 로컬 도구(ALL_TOOLS)에 추가로 합칠 도구 리스트.
               MCP 도구를 바인딩할 때 사용한다(None 이면 로컬 도구만).
        include_refund: request_refund(Human-in-the-loop) 도구 포함 여부.
               기본 True(=/chat/agent 단일 Agent). MCP 경로는 False 로 호출해
               환불 interrupt 를 단일 Agent 경로로만 한정한다.
               · 이유: interrupt 재개(/chat/agent/resume)는 단일 Agent 그래프
                 (_agent_app)에서 처리한다. MCP 그래프는 별도 컴파일 객체라
                 재개 경로가 없으므로, 확인 대기가 갈 곳 없는 상태를 막는다.
    """
    global _compiled_agent
    if _compiled_agent is not None and not force_rebuild:
        return _compiled_agent

    agent_model = model if model is not None else get_main_llm(temperature=0.3)

    # 로컬 도구 + (선택)MCP 등 외부 도구
    # include_refund=False 면 request_refund 를 제외(환불 interrupt 비활성).
    base_tools = [t for t in ALL_TOOLS
                  if include_refund or getattr(t, "name", "") != "request_refund"]
    tools = list(base_tools)
    if extra_tools:
        tools = tools + list(extra_tools)

    # get_main_llm() 이 반환한 모델 + 도구 바인딩된 ReAct Agent (내부에 agent⇄tools 루프 포함)
    # pre_model_hook: LLM 호출 직전 누적 messages 를 최근 N개로 트리밍(원본 비파괴).
    react_agent = create_react_agent(
        model=agent_model,
        tools=tools,
        state_schema=AgentState,
        prompt=_agent_prompt,
        pre_model_hook=agent_pre_model_hook,
    )

    # react_agent 를 하나의 노드로 감싸고, 뒤에 guard 를 직렬 연결
    builder = StateGraph(AgentState)
    builder.add_node("agent", react_agent)
    builder.add_node("guard", _guard_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "guard")
    builder.add_edge("guard", END)

    # 라우터 그래프와 동일 checkpointer 공유 → thread_id 기반 멀티턴 맥락 공유
    cp = checkpointer if checkpointer is not None else get_checkpointer()
    compiled = builder.compile(checkpointer=cp)
    if not force_rebuild:
        _compiled_agent = compiled
    return compiled
