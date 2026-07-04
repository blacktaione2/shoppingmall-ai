"""
graph/multi_agent_state.py
멀티 Agent(Supervisor) 그래프의 State 및 라우팅 스키마.

[설계]
- 단일 Agent 의 AgentState(messages/member_id/is_guest/rag_hits)를 확장한다.
- supervisor 가 다음 행선지를 '구조화 출력'으로 결정하므로 RouteDecision 스키마를 둔다.
  · 자유 텍스트 파싱은 깨지기 쉬워 with_structured_output 으로 안정화.
- routing_count: supervisor↔sub-agent 핑퐁(무한 라우팅) 방지용 카운터.
"""
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# supervisor 가 보낼 수 있는 행선지
AgentRoute = Literal["product_agent", "support_agent", "FINISH"]


class RouteDecision(BaseModel):
    """supervisor 의 라우팅 결정 (구조화 출력).

    next 가 FINISH 이면 더 이상 sub-agent 를 호출하지 않고, final_answer 로
    사용자에게 답한다(인사/잡담/종합답변 모두 supervisor 가 직접 작성).
    """
    next: AgentRoute = Field(
        ...,
        description=(
            "다음 행선지. 상품 검색/추천이 필요하면 'product_agent', "
            "FAQ/주문조회가 필요하면 'support_agent', "
            "더 이상 도구가 필요 없으면 'FINISH'."
        ),
    )
    final_answer: str = Field(
        "",
        description=(
            "next='FINISH' 일 때 사용자에게 전달할 최종 답변. "
            "그 외(sub-agent 라우팅)에는 빈 문자열."
        ),
    )


class MultiAgentState(TypedDict, total=False):
    # 누적 대화/도구 메시지 (sub-agent 들이 add_messages 로 누적)
    messages: Annotated[list[BaseMessage], add_messages]

    # 서버 주입 값 (LLM 이 채우지 않음)
    member_id: Optional[int]
    is_guest: bool

    # product_agent 의 semantic_search 가 기록 → guard 가 읽음
    rag_hits: list[dict]

    # supervisor 라우팅 제어
    next_agent: AgentRoute     # supervisor 가 결정한 다음 행선지
    routing_count: int         # 라우팅 횟수(핑퐁 방지 상한 체크용)
    final_answer: str          # supervisor 가 FINISH 시 작성한 최종 답변
