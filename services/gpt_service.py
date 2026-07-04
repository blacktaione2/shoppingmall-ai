"""
OpenAI GPT 서비스 래퍼
- 날짜 컨텍스트 자동 주입
- 멀티턴 대화 이력 지원 (pipeline_context 사이드채널)
"""
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import AsyncOpenAI
from langsmith import traceable
from typing import TypeVar, Type, List
from pydantic import BaseModel

load_dotenv()

# ── OpenAI SDK 직접 호출용 모델 상수 ─────────────────────────────────
# [중요] 이 상수는 'OpenAI SDK 직접 호출 경로'(gpt_service / rag_service / voice_service)
#        전용이다. 'LangChain 경로'(노드 / Agent / 체인)는 graph.model_factory 의
#        PROVIDER_MODEL_MAP 을 단일 출처로 사용하므로 이 상수와 독립적이다.
#        두 경로의 모델명을 맞추려면 양쪽을 각각 수정해야 한다.
GPT_MODEL_MAIN   = "gpt-5.4"        # RAG 응답 / COMPLAINT / SMALL_TALK 본응답
GPT_MODEL_INTENT = "gpt-5.4-mini"   # 인텐트 분류 / 환각 재시도 (저비용)

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 가 .env 에 설정되어 있지 않습니다.")
        # [개선] HTTP 요청 타임아웃(.env LLM_TIMEOUT, 기본 30초). 무한 대기 방지.
        _client = AsyncOpenAI(api_key=api_key, timeout=_openai_timeout())
    return _client


def _openai_timeout() -> float | None:
    """LLM_TIMEOUT(초)을 읽어 OpenAI SDK timeout 으로 사용(0 이하/오류면 None=무제한)."""
    raw = os.getenv("LLM_TIMEOUT", "30")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 30.0
    return val if val > 0 else None


def _date_context() -> str:
    """오늘 날짜 + 계절 텍스트 생성 (시스템 프롬프트에 주입용)"""
    now = datetime.now()
    month = now.month
    if month in (3, 4, 5):
        season = "봄"
    elif month in (6, 7, 8):
        season = "여름"
    elif month in (9, 10, 11):
        season = "가을"
    else:
        season = "겨울"
    return f"오늘은 {now.year}년 {month}월 {now.day}일({season})입니다."


def _build_history_messages(history: List[dict]) -> List[dict]:
    """
    pipeline_context 에서 가져온 이력을 OpenAI messages 형식으로 변환.
    role 이 'user' 면 user, 'bot'/'error' 면 assistant 로 매핑.
    """
    messages = []
    for item in history:
        role = item.get("role", "user")
        text = item.get("text", "")
        if not text:
            continue
        oai_role = "user" if role == "user" else "assistant"
        messages.append({"role": oai_role, "content": text})
    return messages


@traceable(run_type="llm", name="gpt_service.chat_completion")
async def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str = GPT_MODEL_MAIN,
    temperature: float = 0.7,
) -> str:
    """
    텍스트 응답 생성.
    pipeline_context 에 이력이 있으면 멀티턴으로 자동 전환.
    시스템 프롬프트에 날짜/계절 컨텍스트 자동 주입.

    [운영 경로 참고] 이 함수는 레거시 pipeline/router.py 외에도
    graph/guard.py 의 SEMANTIC 환각 재시도(_retry_semantic_answer)가
    OpenAI SDK 직접 호출용으로 재사용한다. graph/ 경로는 set_chat_history() 를
    호출하지 않으므로, 그 호출에서는 get_chat_history() 가 항상 빈 리스트를
    반환해 history_messages 가 비게 된다 — 이는 의도된 동작이다(재시도는
    '컨텍스트만 근거로 보정'이 목적이라 이력을 섞지 않는 편이 낫다는 판단).
    즉 이 함수가 여전히 ContextVar 를 참조한다고 해서 멀티턴 버그가 있는 게
    아니라, 호출부(graph/guard.py)가 의도적으로 세팅을 생략한 것이다.
    """
    from pipeline.pipeline_context import get_chat_history

    client = get_client()
    date_ctx = _date_context()
    full_system = f"{date_ctx}\n\n{system_prompt}"

    history = get_chat_history()
    history_messages = _build_history_messages(history)

    messages = [{"role": "system", "content": full_system}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_prompt})

    response = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    return response.choices[0].message.content or ""


T = TypeVar("T", bound=BaseModel)


@traceable(run_type="llm", name="gpt_service.structured_completion")
async def structured_completion(
    system_prompt: str,
    user_message: str,
    response_model: Type[T],
    model: str = GPT_MODEL_INTENT,
    temperature: float = 0.0,
) -> T:
    """Structured Outputs 전용 호출 (인텐트 분류용 — 이력 불필요)"""
    client = get_client()
    completion = await client.beta.chat.completions.parse(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format=response_model,
    )
    message = completion.choices[0].message
    if message.refusal:
        raise ValueError(f"GPT refusal: {message.refusal}")
    if message.parsed is None:
        raise ValueError("Structured Outputs 파싱 결과가 비어 있습니다.")
    return message.parsed