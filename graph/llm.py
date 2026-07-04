"""
graph/llm.py
LangChain ChatModel 인스턴스 및 공통 헬퍼.

[설계 의도]
- 멀티 모델 추상화: 실제 모델 생성은 graph.model_factory(create_chat_model)에 위임.
  · get_main_llm/get_intent_llm 의 '시그니처는 유지'하되, 내부에서 팩토리를 호출한다.
  · provider(LLM_PROVIDER)가 바뀌면 캐시를 무효화하고 새 provider 로 재생성한다.
- 인스턴스는 (provider, role, temperature) 조합별로 1개씩 캐시해 재사용한다.
  (.bind(temperature=) 방식은 with_structured_output 이 __getattr__ 위임으로
   '언바운드 원본 모델'에 걸려 바인딩 온도가 무시되는 함정이 있어, 생성자
   temperature 를 캐시 키에 포함하는 방식으로 확정)
- 날짜/계절 컨텍스트 주입은 기존 gpt_service._date_context 와 동일 규칙을 재현한다.
- 멀티턴 이력(history)을 LangChain 메시지로 변환하는 헬퍼를 제공한다.
  · role == "user" → HumanMessage,  그 외("bot"/"error") → AIMessage
"""
import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage, AIMessage, BaseMessage, trim_messages,
)

# 날짜/계절 컨텍스트 (기존 규칙 재사용)
from services.gpt_service import _date_context
from graph.model_factory import create_chat_model, get_provider, ModelRole

load_dotenv()


# ── ChatModel 캐시: (provider, role, temperature) → 인스턴스 ─────────
# provider 가 바뀌면(LLM_PROVIDER 변경) 키가 달라져 자동으로 새 인스턴스가 생성된다.
# temperature 를 키에 포함하는 이유: .bind(temperature=) 로 덮어쓰는 방식은
# with_structured_output() 호출이 RunnableBinding.__getattr__ 위임을 타고
# '언바운드 원본 모델'에 걸려 바인딩 온도가 조용히 무시된다(Structured Output
# 경로 한정 함정). 생성자 temperature 로 고정하면 모든 경로에서 온도가 보장되고,
# 사용 온도 조합이 소수(0.0/0.3/0.7/0.8)라 캐시 항목 수 부담도 없다.
_llm_cache: dict[tuple[str, ModelRole, float], BaseChatModel] = {}


def _get_cached(role: ModelRole, temperature: float) -> BaseChatModel:
    provider = get_provider()
    key = (provider, role, temperature)
    if key not in _llm_cache:
        _llm_cache[key] = create_chat_model(role, temperature=temperature)
    return _llm_cache[key]


def get_main_llm(temperature: float = 0.7) -> BaseChatModel:
    """MAIN(고품질) 역할 LLM 인스턴스.

    provider 에 따라 gpt-5.4 / gemini-3.1-flash / claude-sonnet-4-6 등으로 매핑된다.
    인스턴스는 (provider, role, temperature) 별로 캐시·재사용된다.
    """
    return _get_cached(ModelRole.MAIN, temperature)


def get_intent_llm(temperature: float = 0.0) -> BaseChatModel:
    """INTENT(저비용) 역할 LLM 인스턴스 (인텐트 분류 / 라우팅 / 환각 재시도 공용)."""
    return _get_cached(ModelRole.INTENT, temperature)


def date_system_prefix() -> str:
    """시스템 프롬프트 앞에 붙일 날짜/계절 컨텍스트 (기존 규칙 재사용)."""
    return _date_context()


def history_to_messages(history: list[dict]) -> list[BaseMessage]:
    """멀티턴 이력 dict 리스트 → LangChain 메시지 리스트.

    기존 gpt_service._build_history_messages 와 동일 매핑:
      role == "user" → HumanMessage,  그 외 → AIMessage
    빈 text 는 건너뛴다.
    """
    messages: list[BaseMessage] = []
    for item in history or []:
        role = item.get("role", "user")
        text = item.get("text", "")
        if not text:
            continue
        if role == "user":
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))
    return messages


def messages_to_history(messages: list[BaseMessage]) -> list[dict]:
    """LangChain 메시지 리스트 → 멀티턴 이력 dict 리스트.

    checkpointer(MemorySaver/Redis)에 누적된 state["messages"] 를 노드가 사용하는
    history dict 포맷({"role","text"})으로 역변환한다.
      · HumanMessage → role="user"
      · 그 외(AIMessage 등) → role="bot"
    빈 content 는 건너뛴다.

    [주의] 노드(complaint/small_talk/semantic)는 history dict 를 받아
           내부에서 다시 history_to_messages 로 변환하는 기존 인터페이스를
           유지한다. messages→history→messages 왕복이 생기지만,
           노드 시그니처를 바꾸지 않아 코드/테스트를 보존하는 것이
           더 큰 이득이라 이 방식을 택한다.
    """
    history: list[dict] = []
    for msg in messages or []:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if not content:
            continue
        role = "user" if isinstance(msg, HumanMessage) else "bot"
        history.append({"role": role, "text": content})
    return history


# ════════════════════════════════════════════════════════════════════════
# 멀티턴 이력 트리밍 (컨텍스트 윈도우 초과 방지)
# ════════════════════════════════════════════════════════════════════════
# Checkpointer 에 누적된 messages 는 대화가 길어질수록 무한정 늘어난다. 이를 그대로
# LLM 에 넘기면 모델의 컨텍스트 윈도우(예: 128K)를 초과해 호출이 실패할 수 있다.
# 해결책은 LangChain 내장 trim_messages 로 'LLM 에 들어가는 입력만' 슬라이딩 윈도우로
# 잘라내는 것이다. Checkpointer 의 원본 messages 는 보존하고, 호출 시점에만 트리밍한다.
def _history_max_messages() -> int:
    """LLM 에 넘길 최대 메시지 개수(.env HISTORY_MAX_MESSAGES, 기본 20).

    토큰이 아니라 '메시지 개수' 기준으로 자른다. 토큰 카운팅은 모델/Provider 마다
    토크나이저가 달라 부정확하고 비용이 들지만, 개수 기반은 Provider 중립적이고
    예측 가능하다. 기본 20개(대략 질문/답변 10턴)면 멀티턴 맥락 유지에 충분하다.
    0 이하/오류면 20 으로 폴백.
    """
    raw = os.getenv("HISTORY_MAX_MESSAGES", "20")
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return 20
    return val if val > 0 else 20


def trim_message_list(messages: list[BaseMessage]) -> list[BaseMessage]:
    """BaseMessage 리스트를 최근 N개로 트리밍한다(Agent 경로용).

    - strategy="last": 가장 최근 메시지부터 보존.
    - start_on="human": 트림 결과가 항상 HumanMessage 경계에서 시작하도록 강제.
      → AIMessage(tool_calls) 와 그 결과 ToolMessage 가 한 턴 중간에서 분리되는
        것을 방지한다(분리되면 대부분의 LLM Provider 가 요청을 거부함).
    - include_system=True: SystemMessage 가 있으면 항상 유지.

    [안전장치] 트림 결과가 비거나, 마지막 HumanMessage(=이번 턴 질문)가 사라지면
    트림을 적용하지 않고 원본을 그대로 반환한다. (캡이 한 턴 분량보다 작은
    극단적 경우에 '이번 질문조차 누락'되는 사고를 막는다.)
    누락 검사는 `is`(identity) 비교로 한다 — BaseMessage 는 내용이 같으면 == 로
    동일 취급되어, "네" 같은 반복 질문에서 안전장치가 오판할 수 있기 때문.
    """
    if not messages:
        return messages
    try:
        trimmed = trim_messages(
            messages,
            max_tokens=_history_max_messages(),
            token_counter=len,            # 토큰이 아닌 '메시지 개수' 기준
            strategy="last",
            start_on="human",
            include_system=True,
            allow_partial=False,
        )
    except Exception:
        # 트리밍 자체가 실패하면(예: 예기치 못한 메시지 구조) 원본을 그대로 사용.
        return messages
    if not trimmed:
        return messages

    # [안전장치] 캡이 한 턴 분량보다 작으면 트림 결과에 SystemMessage 만 남고
    # 이번 턴 질문(마지막 HumanMessage)이 사라질 수 있다. 원본의 마지막 HumanMessage 가
    # 트림 결과에서 누락됐다면, 질문조차 못 보는 사고를 막기 위해 원본을 그대로 쓴다.
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    if last_human is not None and not any(m is last_human for m in trimmed):
        return messages
    return trimmed


def trim_history(history: list[dict]) -> list[dict]:
    """history dict 리스트를 최근 N개로 트리밍한다(라우터 경로용).

    라우터 노드(complaint/small_talk/semantic/guard)는 history(dict)를 쓰므로,
    기존 변환 헬퍼(history_to_messages / messages_to_history)를 왕복시켜
    trim_message_list 를 재사용한다. 라우터 history 엔 Tool 메시지가 없어
    tool_call 분리 위험이 없고, 변환 과정에서 빈 텍스트만 걸러진다.
    """
    if not history:
        return history
    msgs = history_to_messages(history)
    trimmed = trim_message_list(msgs)
    if trimmed is msgs:
        return history
    return messages_to_history(trimmed)


def agent_pre_model_hook(state: dict) -> dict:
    """create_react_agent 의 pre_model_hook (Agent LLM 호출 직전 트리밍).

    {"llm_input_messages": [...]} 를 반환하면 Checkpointer 의 messages 원본은
    그대로 두고, '이번 LLM 호출에 들어갈 입력'만 트리밍된 목록으로 대체된다.
    (LangGraph 공식 메시지 이력 관리 방식 — 원본 비파괴)
    """
    messages = state.get("messages", [])
    return {"llm_input_messages": trim_message_list(messages)}
