"""
graph/agent_state.py
단일 Agent 그래프의 확장 State.

[설계]
- create_react_agent 의 기본 AgentState 는 messages / remaining_steps 만 갖는다.
- 우리 Agent 는 추가로 다음이 필요하다:
    · member_id  : order 도구가 InjectedState 로 읽어 본인 주문만 조회 (LLM 추측 금지)
    · is_guest   : 게스트의 주문조회 도구 호출 차단
    · rag_hits   : semantic 도구가 Command 로 기록 → guard 노드가 환각검증에 사용
- messages 는 prebuilt 가 add_messages reducer 로 관리하므로 여기서도 동일하게 둔다.
"""
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # prebuilt create_react_agent 가 관리하는 대화 메시지 (도구 호출/결과 포함)
    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: int           # prebuilt 의 재귀 제한 카운터

    # ── 서버 주입 값 (LLM 이 채우지 않음) ─────────────────────────────
    member_id: Optional[int]       # 로그인 회원 PK (order 도구가 InjectedState 로 사용)
    is_guest: bool                 # 게스트 여부 (order 도구 차단 판정)

    # ── 도구가 기록하는 사이드 정보 ───────────────────────────────────
    rag_hits: list[dict]           # semantic 도구가 Command 로 기록 → guard 가 읽음
