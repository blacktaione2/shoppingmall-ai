"""
[LEGACY] pipeline/pipeline_context.py
LangGraph 전환 이후 ContextVar 사이드채널은 State 기반으로 대체됐다.
  - set_rag_context / get_rag_context: semantic_node 가 state["rag_hits"] 로 대체.
  - set_chat_history / get_chat_history: routers/chat.py 가 state["history"] 로 대체.
  - get_member_id: state["member_id"] 로 대체.
services/rag_service.py 가 get_chat_history() 폴백으로 아직 참조하므로 유지.
삭제 타이밍: rag_service 의 폴백 코드 제거 시 함께 삭제 가능.
----------------------------------------------------------------------
pipeline/pipeline_context.py
요청(Task) 단위로 격리되는 파이프라인 사이드채널 컨텍스트
"""
import contextvars
from typing import List

# ── RAG 컨텍스트 (SEMANTIC 핸들러 → hallucination_guard) ──
_rag_context_var: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "rag_context", default=None
)

def set_rag_context(hits: list[dict]) -> None:
    _rag_context_var.set(hits)

def get_rag_context() -> list[dict] | None:
    return _rag_context_var.get()

def reset_rag_context() -> None:
    _rag_context_var.set(None)


# ── member_id 사이드채널 ──
_member_id_var: contextvars.ContextVar[int] = contextvars.ContextVar(
    "member_id", default=1
)

def set_current_member_id(member_id: int) -> None:
    _member_id_var.set(member_id)

def get_current_member_id() -> int:
    return _member_id_var.get()


# ── [멀티턴] 대화 이력 사이드채널 ──
# gpt_service / rag_service 가 핸들러 시그니처 변경 없이 이력을 읽을 수 있도록
# ContextVar 로 주입한다.
_history_var: contextvars.ContextVar[List[dict]] = contextvars.ContextVar(
    "chat_history", default=[]
)

def set_chat_history(history: List[dict]) -> None:
    """routers/chat.py: 요청 진입 시 이력 저장"""
    _history_var.set(history)

def get_chat_history() -> List[dict]:
    """gpt_service / rag_service: GPT 호출 시 이력 조회"""
    return _history_var.get()