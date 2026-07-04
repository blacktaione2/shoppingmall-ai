"""
graph/state.py
LangGraph 파이프라인의 공유 상태(State) 정의.

[설계 의도]
- 기존 ContextVar 사이드채널(pipeline_context.py)을 State 로 대체한다.
  · rag_context(hits)  → state["rag_hits"]
  · member_id          → state["member_id"]
  · chat_history       → state["history"]
  LangGraph 는 노드 간 State 를 명시적으로 전달하므로, "한 노드가 쓰고 다른
  노드가 읽는" 사이드채널이 State 필드 하나로 자연스럽게 해결된다.

- 노드는 State '전체'를 반환할 필요 없이 '갱신할 키만' 담은 dict 를 반환하면
  LangGraph 가 자동으로 병합(merge)한다. (부분 업데이트)

[필드가 Optional/기본값을 갖는 이유]
- 그래프 진입 시 chat.py 가 question/member_id/is_guest/history 만 채운다.
- intent_result, rag_hits, raw_answer, final_answer 는 노드들이 진행하며 채운다.
  TypedDict 는 런타임 강제는 없지만, total=False 로 "처음엔 비어 있을 수 있음"을
  타입상으로도 명시한다.
"""
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ShoppingState(TypedDict, total=False):
    # ── 입력 (chat.py 가 그래프 진입 시 채움) ──────────────────────────
    question: str                 # 사용자 원본 질문
    member_id: Optional[int]      # 로그인 회원 PK (게스트는 None)
    is_guest: bool                # 게스트 여부 (True=비로그인)
    history: list[dict]           # 직전 대화 이력 [{"role","text"}, ...]
                                  #   · 게스트: 클라이언트가 매 요청에 실어 보냄 (A안)
                                  #   · 로그인: checkpointer messages → history 변환분 주입 (B안)
                                  #     또는 서버 재시작 시 클라이언트 폴백분

    # ── checkpointer 누적 메시지 ──────────────────────────
    # add_messages reducer: 노드가 반환한 메시지를 기존 목록 뒤에 누적(append)한다.
    # · append_message_node 가 매 턴 HumanMessage(질문) + AIMessage(최종답변)를 추가.
    # · 로그인 사용자만 thread_id(=chat_token)로 체크포인터(MemorySaver 또는
    #   AsyncRedisSaver)에 보존된다. 실체는 graph/checkpointer.py 가 결정.
    # · 게스트는 config(thread_id) 없이 invoke 하므로 이 필드가 영속되지 않는다.
    messages: Annotated[list[BaseMessage], add_messages]

    # ── 분류 단계 산출물 ──────────────────────────────────────────────
    # classify_node 결과. 체크포인트(MemorySaver/Redis) 직렬화 안전성을 위해
    # Pydantic 인스턴스가 아니라 dict(primitive)로 저장한다(클래스 경로/버전
    # 비의존 — 저장소에는 데이터만, 형태는 코드가 결정). 소비 지점(edges/nodes/
    # guard/chat.py)은 coerce_intent_result() 로 복원해 타입 안정성을 유지한다.
    intent_result: dict

    # ── 핸들러 단계 산출물 ────────────────────────────────────────────
    rag_hits: list[dict]          # SEMANTIC 검색 결과 (guard_node 가 읽음)
                                  #   ← 기존 ContextVar set_rag_context 대체
    raw_answer: str               # 핸들러가 만든 1차 답변(프리픽스 없음)

    # ── 최종 산출물 ───────────────────────────────────────────────────
    final_answer: str             # guard 통과 후 사용자에게 나갈 최종 답변
