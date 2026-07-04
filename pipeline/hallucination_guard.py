"""
pipeline/hallucination_guard.py
환각 방어 가드 — 순수 검증 로직 (레거시 모듈, 일부 함수는 현재도 재사용됨)

[레거시 안내]
이 파일은 LangGraph 전환 이전의 if/else 파이프라인(pipeline/router.py) 용으로
작성됐다. router.py 자체는 더 이상 운영 경로에서 호출되지 않지만(현재 운영
경로는 graph/builder.py 기반), 이 파일의 순수 검증 함수들
(_validate_semantic_answer, _has_factual_claim, GUIDANCE_SUFFIX 등)은
graph/guard.py 가 그대로 import 해 재사용하고 있다(단일 출처 유지 목적).
즉 이 파일을 통째로 지우면 graph/guard.py 가 깨지므로 삭제하면 안 되고,
아래에서 ContextVar(pipeline_context)를 참조하는 진입점 함수만 레거시다.

[적용 범위 및 전략]
- SEMANTIC_SEARCH (필수)
    1차: GPT 응답에서 "N원" 가격을 정규식으로 추출 → ChromaDB 검색 컨텍스트(hits)의
         metadata.price 집합과 비교. 컨텍스트에 없는 가격이 언급되거나,
         가격/상품명이 모두 컨텍스트와 무관하면 환각으로 판정.
         (정중한 거절 응답은 REJECTION_KEYWORDS 로 판별해 검증 대상에서 제외)
    재시도: 1차 검증 실패 시 gpt-5.4-mini 로 "컨텍스트만 근거로 다시 작성" 지시와 함께
            1회만 재호출 → 재검증.
    폴백: 재시도 후에도 실패하면 고정 안내 문구로 대체.

- COMPLAINT / SMALL_TALK (경량)
    재호출 없이, 응답에 주문번호 패턴(ORD-YYYYMMDD-NNNN)이나 "환불 완료/배송 완료/
    오늘·내일 도착" 같은 단정적 사실 표현이 보이면 안내 문구를 답변 끝에 append.
    (이 두 인텐트는 실제 주문/배송 데이터에 접근 권한이 없으므로 그런 단정은
     전부 GPT의 추측이며, 재생성 대신 가벼운 안내로 보완하는 것이 비용/지연/UX 측면에서 최선)

- STRUCTURED_QUERY / FAQ / ORDER_INQUIRY
    GPT 미사용 → 환각 가능성 없음 → pass-through

[RAG 컨텍스트(hits) 획득 — 레거시 진입점(guard_answer)만 해당]
- pipeline.pipeline_context.get_rag_context() 로 조회 (ContextVar 사이드채널,
  semantic_handler.py 가 검색 직후 set_rag_context(hits) 로 저장해둔 값)
  현재 운영 경로(graph/guard.py)는 이 ContextVar 대신 state["rag_hits"] 를
  인자로 직접 전달받으므로 이 사이드채널을 타지 않는다.
"""
import logging
import re

from schemas.intent_schema import IntentResult, IntentType
from pipeline.pipeline_context import get_rag_context
from services import gpt_service, rag_service

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# SEMANTIC_SEARCH 검증
# ═════════════════════════════════════════════════════════════════════════

# "12,345원" / "12345원" 형태의 가격 추출 (콤마 포함, '원' 직전 숫자)
_PRICE_PATTERN = re.compile(r"[\d][\d,]*(?=\s*원)")

# GPT가 "관련 상품 없음/추천 어려움"으로 정중히 거절한 응답 판별용 키워드.
# rag_service.SYSTEM_PROMPT 의 거절 안내 문구에 맞춰 작성했으며,
# 실전 응답 패턴을 보며 추가/수정하기 쉽도록 상수로 분리해둔다.
REJECTION_KEYWORDS = [
    "찾지 못했",
    "관련 상품이 없",
    "추천하기 어려",
    "추천드리기 어려",
]

# 1차 검증 실패 시 gpt-5.4-mini 재시도용 시스템 프롬프트 보강 문구
_RETRY_SYSTEM_SUFFIX = (
    "\n\n[중요] 이전 답변에는 아래 상품 컨텍스트와 일치하지 않는 가격 또는 상품 정보가 "
    "포함되어 있었습니다. 반드시 상품 컨텍스트에 명시된 상품명과 가격만 사용하여 "
    "다시 답변하세요. 컨텍스트에 없는 정보는 절대 언급하지 마세요."
)

# 재시도 후에도 검증에 실패했을 때 사용하는 최종 안전 문구
# (다른 핸들러의 "[추천]" 프리픽스와 통일감 유지 - 워딩 조정 시 이 상수만 수정)
SEMANTIC_FALLBACK_MESSAGE = (
    "[추천] 죄송합니다, 정확한 상품 정보를 확인하기 어려워요. 다른 검색어로 다시 시도해 주세요."
)


def _extract_prices(text: str) -> set[int]:
    """답변 텍스트에서 'N원' 형태의 가격을 모두 추출해 정수 집합으로 반환한다."""
    prices: set[int] = set()
    for match in _PRICE_PATTERN.findall(text):
        cleaned = match.replace(",", "")
        if cleaned.isdigit():
            prices.add(int(cleaned))
    return prices


def _is_rejection(answer: str) -> bool:
    """GPT가 컨텍스트와 무관함을 인정하고 정중히 거절한 응답인지 판별한다.

    거절 응답은 컨텍스트의 가격/상품명을 언급하지 않는 것이 정상이므로
    환각 검증 대상에서 제외한다.
    """
    return any(keyword in answer for keyword in REJECTION_KEYWORDS)


# 상품명 토큰의 최소 길이 (1글자 토큰은 우연 매칭 노이즈가 커서 제외)
_MIN_NAME_TOKEN_LEN = 2


def _collect_valid_prices(hits: list[dict]) -> set[int]:
    """검색 컨텍스트(hits)에 '근거로 인정되는' 가격 집합을 만든다.

    metadata.price 뿐 아니라 상품 설명(document) 텍스트에 적힌 'N원' 가격도
    포함한다. GPT가 설명에 등장한 가격(예: '정가 99,000원')을 인용한 경우를
    환각으로 오판(false positive)하지 않기 위함. (설명도 엄연히 제공된 컨텍스트)
    """
    valid_prices: set[int] = set()
    for hit in hits:
        meta = hit.get("metadata") or {}
        price = meta.get("price")
        if isinstance(price, (int, float)):
            valid_prices.add(int(price))
        # 설명 텍스트 내 'N원' 도 컨텍스트 근거로 인정
        document = hit.get("document") or ""
        valid_prices |= _extract_prices(document)
    return valid_prices


def _collect_name_tokens(hits: list[dict]) -> set[str]:
    """상품명을 공백 기준 토큰(2글자 이상)으로 분해한 집합을 만든다.

    전체 상품명 문자열 매칭('롱패딩 자켓' in answer)은 GPT 가 '롱패딩'처럼
    줄여 부르면 실패(false negative)한다. 토큰 단위로 하나라도 답변에
    등장하면 컨텍스트 상품을 가리킨 것으로 인정한다.
    """
    tokens: set[str] = set()
    for hit in hits:
        meta = hit.get("metadata") or {}
        name = meta.get("product_name")
        if not name:
            continue
        for token in str(name).split():
            if len(token) >= _MIN_NAME_TOKEN_LEN:
                tokens.add(token)
    return tokens


def _validate_semantic_answer(answer: str, hits: list[dict]) -> bool:
    """SEMANTIC 응답이 ChromaDB 컨텍스트(hits)와 부합하는지 1차(정규식) 검증한다.

    Args:
        answer : "[추천] " 프리픽스가 포함된 최종 응답 문자열
        hits   : chroma_service.search_similar() 결과
                 (각 원소: {"id", "document", "metadata": {"product_name", "price", ...}, "distance"})

    Returns:
        True  : 통과 (컨텍스트 가격/상품명에 근거하거나, 정상적인 거절 응답)
        False : 환각 의심 (컨텍스트에 없는 가격 언급, 또는 가격/상품명 어느 근거도 없는 응답)

    [판정 규칙]
      0) 정중한 거절 응답이면 통과 (컨텍스트 미언급이 정상).
      1) 답변에 언급된 가격 중 컨텍스트(metadata+설명)에 없는 값이 하나라도 있으면 환각.
      2) (1 통과 후) 근거 신호 = '유효 가격 언급' 또는 '상품명 토큰 등장' 중 하나라도 있어야 함.
         둘 다 없으면 컨텍스트와 무관한 응답으로 보고 환각 처리.
    """
    # 0) 정중한 거절 응답은 컨텍스트 언급이 없는 것이 정상 → 통과
    if _is_rejection(answer):
        return True

    valid_prices = _collect_valid_prices(hits)
    name_tokens = _collect_name_tokens(hits)
    mentioned_prices = _extract_prices(answer)

    # 1) 가격 검증: 컨텍스트에 없는 가격 언급 → 환각
    if mentioned_prices and not mentioned_prices.issubset(valid_prices):
        return False

    # 2) 근거 신호 검증: 유효 가격 언급(1을 통과) 또는 상품명 토큰 등장 중 하나는 있어야 함
    has_price_grounding = bool(mentioned_prices)
    has_name_grounding = any(token in answer for token in name_tokens)
    if not (has_price_grounding or has_name_grounding):
        return False

    return True


async def _retry_semantic_answer(question: str, hits: list[dict]) -> str:
    """1차 검증 실패 시 gpt-5.4-mini 로 1회 재시도 (저비용/저지연 보정 호출)."""
    context = rag_service.build_product_context(hits)
    user_prompt = (
        f"[고객 질문]\n{question}\n\n"
        f"[상품 컨텍스트]\n{context}\n\n"
        f"위 상품 컨텍스트만 근거로 고객 질문에 답변하세요."
    )
    retry_system_prompt = rag_service.SYSTEM_PROMPT + _RETRY_SYSTEM_SUFFIX
    return await gpt_service.chat_completion(
        system_prompt=retry_system_prompt,
        user_prompt=user_prompt,
        model=gpt_service.GPT_MODEL_INTENT,  # gpt-5.4-mini: 재시도는 저비용 모델로 충분
        temperature=0.0,
    )


async def _guard_semantic(question: str, answer: str) -> str:
    """SEMANTIC_SEARCH 환각 가드 본체."""
    hits = get_rag_context()

    # hits 가 없거나(0건) → semantic_handler 의 고정 안내 문구(GPT 미호출) → 검증 불필요
    if not hits:
        return answer

    if _validate_semantic_answer(answer, hits):
        return answer

    logger.warning("SEMANTIC 환각 의심 → gpt-5.4-mini 1회 재시도. question=%r", question)
    try:
        retried = await _retry_semantic_answer(question, hits)
    except Exception:
        logger.exception("환각 재시도 호출 중 오류 발생")
        return SEMANTIC_FALLBACK_MESSAGE

    retried_answer = f"[추천] {retried.strip()}"
    if _validate_semantic_answer(retried_answer, hits):
        return retried_answer

    logger.warning("재시도 후에도 환각 의심 → 안전 문구로 폴백. question=%r", question)
    return SEMANTIC_FALLBACK_MESSAGE


# ═════════════════════════════════════════════════════════════════════════
# COMPLAINT / SMALL_TALK 경량 검증
# ═════════════════════════════════════════════════════════════════════════

# 주문번호 패턴 (ORD-YYYYMMDD-NNNN)
_ORDER_ID_PATTERN = re.compile(r"ORD-\d{8}-\d{4}")

# 단정적 사실 표현 패턴들.
# COMPLAINT/SMALL_TALK 핸들러는 실제 주문/배송/환불 데이터에 접근하지 않으므로,
# 이런 표현은 전부 GPT의 추측/환각으로 간주한다.
_FACT_CLAIM_PATTERNS = [
    re.compile(r"환불[\s\S]{0,10}(완료|처리)"),   # "환불 완료/처리되었습니다" 등
    re.compile(r"배송[\s\S]{0,10}완료"),           # "배송이 완료되었습니다" 등
    re.compile(r"(오늘|내일)[\s\S]{0,10}(도착|배송)"),  # "내일 도착합니다" 등
]

# 단정적 사실 표현 감지 시 답변 끝에 덧붙이는 안내 문구
GUIDANCE_SUFFIX = "\n\n📌 정확한 주문/배송 상태는 '주문조회' 메뉴에서 확인해 주세요."


def _has_factual_claim(answer: str) -> bool:
    """COMPLAINT/SMALL_TALK 응답에 근거 없는 단정적 사실 표현이 있는지 검사한다."""
    if _ORDER_ID_PATTERN.search(answer):
        return True
    return any(pattern.search(answer) for pattern in _FACT_CLAIM_PATTERNS)


# ═════════════════════════════════════════════════════════════════════════
# 진입점
# ═════════════════════════════════════════════════════════════════════════

async def guard_answer(question: str, answer: str, intent_result: IntentResult) -> str:
    """인텐트별 환각 방어 진입점 (chat.py 시그니처 변경 없음)."""
    if intent_result.intent == IntentType.SEMANTIC_SEARCH:
        return await _guard_semantic(question, answer)

    if intent_result.intent in (IntentType.COMPLAINT, IntentType.SMALL_TALK):
        if _has_factual_claim(answer):
            return answer + GUIDANCE_SUFFIX
        return answer

    # STRUCTURED_QUERY / FAQ / ORDER_INQUIRY: GPT 미사용 → pass-through
    return answer
