"""
services/pricing.py
모델별 토큰 단가표 + 비용 환산.

[목적]
- (provider, model) → (input_per_1k, output_per_1k) USD 단가를 제공한다.
- 토큰 사용량을 받아 비용(USD)을 추정한다(estimate_cost).

[설계 메모]
- 단가는 시점에 따라 바뀌므로 '문서화된 기준 단가'를 코드에 두되, .env 로
  언제든 오버라이드할 수 있게 한다(배포 후 실단가 반영 용이).
  · 오버라이드 키 형식: PRICE_{PROVIDER}_{MODELKEY}_INPUT / _OUTPUT (USD per 1K tokens)
    예) PRICE_OPENAI_GPT_5_4_INPUT=0.005
    (MODELKEY 는 모델명을 대문자화하고 영숫자 외 문자를 '_' 로 치환한 값)
- 미등록 모델 → 단가 0 + 경고. 비용을 0 으로 안전 처리해 측정이 응답을 깨지 않게 한다.
  (실측 단계에서 .env 로 단가를 채우면 자동 반영된다)
- 이 단가표의 숫자는 '문서 템플릿용 자리표시 기준값'이다. 실제 청구 단가는
  배포 시점에 각 provider 공식 가격표로 .env 에서 갱신할 것(PHASE3_BENCHMARK.md 참고).
"""
import logging
import os
import re

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# (provider, model) → (input_usd_per_1k, output_usd_per_1k)
# ⚠️ 자리표시 기준값. 배포 시 각 provider 공식 가격표로 .env 에서 갱신.
# 출처: 2026-06 기준 각 provider 공식 pricing 페이지
_DEFAULT_PRICING = {
    # ── OpenAI ────────────────────────────────────────────────────────
    ("openai", "gpt-5.4"):          (0.005,   0.015),
    ("openai", "gpt-5.4-mini"):     (0.0006,  0.0024),
    # ── Google Gemini ─────────────────────────────────────────────────
    # gemini-2.5-flash → gemini-3.1-flash 교체
    # gemini-3.1-flash: 고객지원 챗봇 최저비용 선택지 ($0.075/1M input)
    ("gemini", "gemini-3.1-flash"): (0.000075, 0.0003),
    # 구버전 호환 (기존 .env 에 2.5-flash 설정된 경우 폴백)
    ("gemini", "gemini-2.5-flash"): (0.0003,  0.0025),
    # ── Anthropic ─────────────────────────────────────────────────────
    ("anthropic", "claude-sonnet-4-6"):          (0.003,  0.015),
    ("anthropic", "claude-haiku-4-5-20251001"):  (0.0008, 0.004),
    # ── DeepSeek ─────────────────────────────────────────────────────
    # 신규 추가
    # deepseek-v4-flash: $0.14/1M input, $0.28/1M output (2026-06 기준)
    # 캐시 히트 시 input $0.0028/1M (98% 할인) — 캐시 적용 시 별도 계산 필요
    # deepseek-chat 은 v4-flash 의 alias (2026-07-24 deprecated 예정)
    ("deepseek", "deepseek-v4-flash"): (0.00014, 0.00028),
    ("deepseek", "deepseek-chat"):     (0.00014, 0.00028),  # alias 호환
}


def _model_key(model: str) -> str:
    """모델명을 .env 오버라이드 키 조각으로 정규화(대문자 + 비영숫자→'_')."""
    return re.sub(r"[^0-9A-Za-z]+", "_", model).upper().strip("_")


def _env_override(provider: str, model: str) -> tuple[float, float] | None:
    """PRICE_{PROVIDER}_{MODELKEY}_INPUT/_OUTPUT 오버라이드를 읽는다(둘 다 있을 때만)."""
    base = f"PRICE_{provider.upper()}_{_model_key(model)}"
    in_raw = os.getenv(f"{base}_INPUT")
    out_raw = os.getenv(f"{base}_OUTPUT")
    if in_raw is None or out_raw is None:
        return None
    try:
        return float(in_raw), float(out_raw)
    except ValueError:
        logger.warning("단가 오버라이드 파싱 실패: %s_INPUT/_OUTPUT", base)
        return None


def get_unit_price(provider: str, model: str) -> tuple[float, float]:
    """(input_usd_per_1k, output_usd_per_1k) 단가를 반환한다.

    우선순위: .env 오버라이드 > 기본 단가표 > (0.0, 0.0)+경고.
    """
    override = _env_override(provider, model)
    if override is not None:
        return override

    price = _DEFAULT_PRICING.get((provider, model))
    if price is None:
        logger.warning(
            "단가 미등록: provider=%s model=%s → 비용 0 처리(.env 로 PRICE_* 설정 가능)",
            provider, model,
        )
        return (0.0, 0.0)
    return price


def estimate_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """토큰 사용량으로 비용(USD)을 추정한다.

    Args:
        provider: 'openai' | 'gemini' | 'anthropic' | 'deepseek'.
        model: 실제 모델명(예: 'gpt-5.4').
        prompt_tokens: 입력 토큰 수.
        completion_tokens: 출력 토큰 수.
    Returns:
        추정 비용(USD, float). 단가 미등록이면 0.0.

    [메모]
    - prompt/completion 을 구분 집계할 수 없는 경로(total 만 아는 경우)는
      호출 측에서 completion_tokens=0, prompt_tokens=total 로 넣고
      input 단가만 적용하는 식의 근사도 가능하나, 정확도가 떨어지므로
      가능하면 입출력을 구분해 넘기는 것을 권장한다.
    """
    in_price, out_price = get_unit_price(provider, model)
    cost = (prompt_tokens / 1000.0) * in_price + (completion_tokens / 1000.0) * out_price
    return round(cost, 6)
