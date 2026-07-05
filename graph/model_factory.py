"""
graph/model_factory.py
멀티 모델 추상화 팩토리 — GPT / Gemini / Claude / DeepSeek

[목적]
- LLM 제공자(provider)를 환경변수로 교체 가능하게 추상화한다.
  · LangChain 의 BaseChatModel 인터페이스(ainvoke/bind_tools/with_structured_output)는
    provider 와 무관하게 동일하므로, '인스턴스 생성 지점'만 추상화하면
    노드/Agent/체인 코드는 전혀 바꾸지 않아도 된다.
- 벤치마크: provider 를 바꿔가며 토큰/품질/레이턴시/비용을 정량 비교한다.

[provider × 역할 → 실제 모델명] (2026-06 기준, .env 로 오버라이드 가능)
  openai    : main=gpt-5.4              intent=gpt-5.4-mini
  gemini    : main=gemini-3.1-flash-lite  intent=gemini-3.1-flash-lite  ← 가성비 포지션 비교용
  anthropic : main=claude-sonnet-4-6    intent=claude-haiku-4-5-20251001
  deepseek  : main=deepseek-v4-flash    intent=deepseek-v4-flash  ← 신규 추가

[DeepSeek 주의사항]
- 모델명: deepseek-v4-flash (기존 deepseek-chat 은 2026-07-24 deprecated)
- LangChain 패키지: langchain-deepseek (pip install langchain-deepseek)
- 환경변수: DEEPSEEK_API_KEY
- 도구 호출/Structured Output 지원: ✅ (deepseek-v4-flash 기준)
- 복잡한 long-chain Agent 작업에서 tool-calling 안정성 이슈 보고 있음 →
  벤치마크에서 Agent 경로 안정성을 별도 모니터링 필요.

[Gemini 3+ temperature 주의사항]
- langchain-google-genai 가 Gemini 3+ 모델에서 temperature 를 자동으로 1.0 으로
  override 한다 (temperature < 1.0 에서 무한루프 및 성능저하 방지 목적).
- 인텐트 분류 노드(temperature=0.0 의도)에서 실제 적용 여부를 확인해야 한다.
- 회피 방법: 분류 노드에서 gemini provider 사용 시 특이 결과 모니터링 필요.

[안전장치]
- lazy import: gemini/anthropic/deepseek 패키지는 해당 provider 선택 시에만 import.
- 키 부재 시 명확한 RuntimeError (어느 키가 필요한지 안내).
- 알 수 없는 provider → openai 로 폴백 + 경고 로그.

[범위]
- 이 팩토리는 'LangChain 경로(노드/Agent)' 전용이다.
  services/gpt_service·rag_service·voice_service 의 '순수 OpenAI SDK 직접 호출'은
  대상이 아니다(그쪽은 openai 고정).
"""
import logging
import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ModelRole(str, Enum):
    """모델 역할 (복잡 작업 vs 간단 작업)."""
    MAIN = "main"       # RAG/공감/Agent 본응답 등 고품질
    INTENT = "intent"   # 인텐트 분류/라우팅/환각 재시도 등 저비용


# provider × 역할 → 기본 모델명 (.env 로 오버라이드 가능)
# gemini: 2.5-flash → 3.1-flash-lite 교체 (provider 비교에서 가성비 포지션 대표로 채택)
# deepseek 신규 추가 (v4-flash: tool calling + structured output 지원)
_DEFAULT_MODELS = {
    "openai": {
        ModelRole.MAIN:   "gpt-5.4",
        ModelRole.INTENT: "gpt-5.4-mini",
    },
    "gemini": {
        ModelRole.MAIN:   "gemini-3.1-flash-lite",   # ← 2.5-flash 에서 교체(가성비 포지션 대표)
        ModelRole.INTENT: "gemini-3.1-flash-lite",
    },
    "anthropic": {
        ModelRole.MAIN:   "claude-sonnet-4-6",
        ModelRole.INTENT: "claude-haiku-4-5-20251001",
    },
    "deepseek": {                                # ← 신규
        ModelRole.MAIN:   "deepseek-v4-flash",
        ModelRole.INTENT: "deepseek-v4-flash",
    },
}

# .env 오버라이드 키 (provider+role 별)
#   예) OPENAI_MODEL_MAIN, GEMINI_MODEL_INTENT, DEEPSEEK_MODEL_MAIN ...
_ENV_OVERRIDE_TMPL = "{provider}_MODEL_{role}"


def get_provider() -> str:
    """현재 LLM provider (환경변수 LLM_PROVIDER, 기본 openai)."""
    return os.getenv("LLM_PROVIDER", "openai").strip().lower()


def _get_timeout() -> float | None:
    """[개선] LLM 호출 타임아웃(초). .env LLM_TIMEOUT 으로 설정(기본 30초).

    OpenAI/Anthropic/Gemini/DeepSeek 의 LangChain 래퍼는 모두 생성자 timeout 인자를
    HTTP 요청 타임아웃으로 사용한다. 0 이하/파싱 실패면 None(무제한, 기존 동작)으로 둔다.
    이렇게 생성 시점에 주입해야 .with_config(timeout=) 같은 Runnable 레벨이 아니라
    실제 네트워크 요청에 타임아웃이 걸린다.
    """
    raw = os.getenv("LLM_TIMEOUT", "30")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 30.0
    return val if val > 0 else None


def resolve_model_name(provider: str, role: ModelRole) -> str:
    """provider+role 에 대한 실제 모델명을 결정한다(.env 오버라이드 우선)."""
    env_key = _ENV_OVERRIDE_TMPL.format(
        provider=provider.upper(), role=role.value.upper()
    )
    override = os.getenv(env_key)
    if override:
        return override.strip()
    table = _DEFAULT_MODELS.get(provider, _DEFAULT_MODELS["openai"])
    return table[role]


def _require_key(env_name: str, provider: str) -> str:
    key = os.getenv(env_name)
    if not key:
        raise RuntimeError(
            f"{provider} provider 를 사용하려면 {env_name} 가 .env 에 설정되어야 합니다."
        )
    return key


def _create_openai(model_name: str, temperature: float):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model_name,
        api_key=_require_key("OPENAI_API_KEY", "openai"),
        temperature=temperature,
        timeout=_get_timeout(),   # [개선] HTTP 요청 타임아웃(무한 대기 방지)
    )


def _create_gemini(model_name: str, temperature: float):
    """gemini-3.1-flash-lite 로 교체.

    [Gemini 3+ temperature 주의]
    langchain-google-genai 는 Gemini 3+ 에서 temperature < 1.0 이면
    자동으로 1.0 으로 override 한다 (공식 동작).
    인텐트 분류 노드의 temperature=0.0 의도가 무효화될 수 있다.

    gemini-3 이상 + temperature<1.0 조합이면 경고 로그를 남겨 운영 중 발생
    빈도를 추적한다(override 는 라이브러리 내부 동작이라 막을 수 없음 —
    가시성 확보가 목적).
    """
    if _is_gemini_3_plus(model_name) and temperature < 1.0:
        logger.warning(
            "Gemini 3+(%s) 요청 temperature=%.2f 는 langchain-google-genai 가 "
            "내부적으로 1.0 으로 override 할 수 있습니다. 인텐트 분류처럼 "
            "낮은 temperature 가 필요한 호출이면 분류 정확도 회귀 여부를 확인하세요.",
            model_name, temperature,
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        raise RuntimeError(
            "gemini provider 를 쓰려면 'langchain-google-genai' 패키지가 필요합니다. "
            "pip install langchain-google-genai"
        ) from e
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=_require_key("GOOGLE_API_KEY", "gemini"),
        temperature=temperature,
        timeout=_get_timeout(),   # [개선] HTTP 요청 타임아웃(무한 대기 방지)
    )


def _is_gemini_3_plus(model_name: str) -> bool:
    """모델명 문자열에서 'gemini-N' 의 N 이 3 이상인지 판별한다(경고 로그용 휴리스틱).

    "gemini-3.1-flash" → 3, "gemini-2.5-flash" → 2 처럼 두 번째 하이픈 구간의
    선행 정수만 본다. 파싱 실패 시 안전하게 False(경고 미발생)로 처리한다.
    """
    try:
        major = model_name.lower().split("gemini-", 1)[1].split(".", 1)[0].split("-", 1)[0]
        return int(major) >= 3
    except (IndexError, ValueError):
        return False


def _create_anthropic(model_name: str, temperature: float):
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic provider 를 쓰려면 'langchain-anthropic' 패키지가 필요합니다. "
            "pip install langchain-anthropic"
        ) from e
    return ChatAnthropic(
        model=model_name,
        api_key=_require_key("ANTHROPIC_API_KEY", "anthropic"),
        temperature=temperature,
        timeout=_get_timeout(),   # [개선] HTTP 요청 타임아웃(무한 대기 방지)
    )


def _create_deepseek(model_name: str, temperature: float):
    """DeepSeek V4 Flash 신규 지원.

    [구현 방식]
    langchain-deepseek 공식 패키지(ChatDeepSeek)를 사용한다.
    ChatOpenAI + base_url 방식도 가능하지만, 공식 LangChain 문서가
    "DeepSeek 는 provider-specific 패키지(ChatDeepSeek)를 사용하라"고
    명시하므로 공식 패키지를 선택한다.
    → reasoning_content 등 DeepSeek 전용 필드도 올바르게 처리됨.

    [모델명]
    deepseek-v4-flash: tool calling + structured output 지원.
    (deepseek-chat 은 2026-07-24 deprecated → v4-flash 직접 사용)

    [비용]
    $0.14/1M input, $0.28/1M output (2026-06 기준).
    캐시 히트 시 input $0.0028/1M (98% 할인).

    [주의]
    복잡한 multi-hop Agent 작업에서 tool-calling 안정성 이슈 보고 있음.
    쇼핑몰 챗봇 수준(단순 RAG/분류)에서는 문제없지만
    멀티 Agent 경로에서 응답 이상 시 LLM_PROVIDER 를 openai 로 전환 권장.
    """
    try:
        from langchain_deepseek import ChatDeepSeek
    except ImportError as e:
        raise RuntimeError(
            "deepseek provider 를 쓰려면 'langchain-deepseek' 패키지가 필요합니다. "
            "pip install langchain-deepseek"
        ) from e
    return ChatDeepSeek(
        model=model_name,
        api_key=_require_key("DEEPSEEK_API_KEY", "deepseek"),
        temperature=temperature,
        timeout=_get_timeout(),   # [개선] HTTP 요청 타임아웃(무한 대기 방지)
        # [버그 수정] deepseek-v4-flash 는 기본적으로 thinking mode 로 동작하는데,
        # 이 모드는 강제 도구 호출(tool_choice)을 지원하지 않는다.
        # .with_structured_output() 이 내부적으로 tool_choice 를 강제하므로,
        # classify_node 의 인텐트 분류가 매번 400 Thinking mode does not support
        # this tool_choice 로 실패했다(DeepSeek API 의 알려진 제약, DeepSeek-V3
        # 저장소 issue #1376 등에서 동일 현상 다수 보고됨). 이 프로젝트는 단순
        # 분류/RAG 용도라 추론 모드가 애초에 불필요해 꺼도 기능 손실이 없다.
        extra_body={"thinking": {"type": "disabled"}},
    )


_CREATORS = {
    "openai":    _create_openai,
    "gemini":    _create_gemini,
    "anthropic": _create_anthropic,
    "deepseek":  _create_deepseek,   # 신규
}


def create_chat_model(role: ModelRole, temperature: float = 0.7):
    """현재 provider 에 맞는 ChatModel 인스턴스를 생성한다.

    Args:
        role: MAIN(고품질) 또는 INTENT(저비용).
        temperature: 샘플링 온도.
            · openai/anthropic/deepseek: 지정값 그대로 적용.
            · gemini 3+: 1.0 미만은 langchain 내부에서 1.0 으로 override 될 수 있음.
    Returns:
        LangChain BaseChatModel (provider 무관 동일 인터페이스).
    """
    provider = get_provider()
    creator = _CREATORS.get(provider)
    if creator is None:
        logger.warning("알 수 없는 LLM_PROVIDER=%r → openai 로 폴백", provider)
        provider = "openai"
        creator = _CREATORS["openai"]

    model_name = resolve_model_name(provider, role)
    logger.debug("create_chat_model: provider=%s role=%s model=%s",
                 provider, role.value, model_name)
    return creator(model_name, temperature)