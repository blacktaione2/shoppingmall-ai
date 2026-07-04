"""
graph/model_policy.py
동적 모델 선택 정책 레이어

[목적]
- 작업 복잡도(TaskComplexity)에 따라 적절한 LLM 을 자동 선택한다.
  · SIMPLE  : 인사/잡담 등 가벼운 응답 → get_intent_llm() (저비용 역할, 기본 gpt-5.4-mini)
  · COMPLEX : RAG/공감응답 등 품질 중요 → get_main_llm() (고품질 역할, 기본 gpt-5.4)
  · 두 역할 모두 .env LLM_PROVIDER 에 따라 실제 모델이 바뀐다(model_factory.py 공용,
    provider 고정 아님 — 아래 예시는 기본값인 openai 기준).
- 환경변수 DYNAMIC_MODEL_SELECTION 으로 on/off 토글.
  · off(기본) 이면 항상 COMPLEX 역할(get_main_llm)을 반환 → 기존 동작과 동일(안전).
  · on 이면 복잡도에 따라 분기.

[설계 의도]
- 기존 get_main_llm / get_intent_llm 은 그대로 두고, 그 위에 정책 함수를 둔다.
  → 노드는 select_llm(complexity) 만 호출하면 되고, 모델 상수/온도는 정책이 캡슐화.
- 알 수 없는 complexity 는 COMPLEX(get_main_llm)로 안전 폴백 → 품질 우선.
- 실제 비용 절감 효과는 LangSmith 트레이스로 검증한 뒤 적용 범위를 넓힌다.
"""
import logging
import os
from enum import Enum

from dotenv import load_dotenv

from graph.llm import get_main_llm, get_intent_llm

load_dotenv()

logger = logging.getLogger(__name__)


class TaskComplexity(str, Enum):
    """작업 복잡도 분류 (모델 선택 기준)."""
    SIMPLE = "simple"     # 가벼운 응답 → 저비용 모델 후보
    COMPLEX = "complex"   # 품질 중요 → 고품질 모델


def _dynamic_enabled() -> bool:
    """환경변수로 동적 모델 선택 활성화 여부 판정.

    매 호출마다 읽어 런타임 토글이 가능하게 한다(테스트/운영 중 전환 용이).
    'true'/'1'/'yes'(대소문자 무시)만 활성으로 본다.
    """
    return os.getenv("DYNAMIC_MODEL_SELECTION", "false").strip().lower() in (
        "true", "1", "yes",
    )


def select_llm(complexity: TaskComplexity, temperature: float = 0.7):
    """복잡도에 맞는 LLM 인스턴스를 반환한다.

    Args:
        complexity: 작업 복잡도.
        temperature: 샘플링 온도(노드 성격에 맞게 호출 측에서 지정).
    Returns:
        LangChain ChatOpenAI(바인딩) 인스턴스.

    [동작]
      - 동적 선택 OFF → 항상 get_main_llm() (기존 동작 보존).
      - 동적 선택 ON  → SIMPLE 은 get_intent_llm(), 그 외는 get_main_llm().
    """
    if not _dynamic_enabled():
        return get_main_llm(temperature=temperature)

    if complexity == TaskComplexity.SIMPLE:
        logger.debug("동적 모델 선택: SIMPLE → get_intent_llm()")
        return get_intent_llm(temperature=temperature)

    # COMPLEX 또는 알 수 없는 값 → 고품질 모델로 안전 폴백
    return get_main_llm(temperature=temperature)
