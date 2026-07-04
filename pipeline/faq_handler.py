"""
pipeline/faq_handler.py
FAQ 인텐트 핸들러 (레거시 진입점 + 현재도 재사용되는 검색 로직)

[상태] handle_faq()는 레거시(pipeline/router.py 전용) 진입점이지만,
내부 _search_faq_sync() 는 graph/nodes.py 의 faq_node 가 그대로 재사용한다
(단일 출처 유지 목적, 중복 구현 방지).

[검색 전략: C안 - 혼합형]
  Step 1) 인텐트 분류기가 추출한 entities.keywords 로 LIKE 검색
  Step 2) (keywords 없음 또는 결과 없음) → 원본 질문을 공백 분리한
          토큰으로 폴백 검색
  Step 3) 그래도 없으면 고정 안내 문구 반환 (GPT 사용 ❌ - 아키텍처 확정)

[응답 정책]
  - 1순위 매칭 FAQ 1건을 본문으로 출력
  - 추가 매칭이 있으면 "이런 질문도 있어요" 형태로 질문 제목만 노출
"""
import asyncio

from schemas.intent_schema import IntentResult
from database.oracle_db import search_faq_by_keywords

# 폴백 토큰화 시 한 글자 조사/접속어 노이즈 제거를 위한 최소 길이
_MIN_TOKEN_LEN = 2

# 매칭 실패 시 고정 안내 문구 (Q2-A안 확정: GPT 폴백 없음)
_FAQ_NOT_FOUND_MESSAGE = (
    "죄송합니다. 관련 FAQ를 찾지 못했습니다.\n"
    "자세한 문의는 고객센터(1234-5678, 평일 09:00~18:00)를 이용해 주세요."
)


def _tokenize_question(question: str) -> list[str]:
    """원본 질문을 공백 기준 분리 → 2글자 이상 토큰만 추출 (폴백용)

    형태소 분석기 없이 동작해야 하므로 단순 공백 split 사용.
    '배송은 언제 오나요?' → ['배송은', '언제', '오나요?'] 처럼
    조사가 붙은 토큰은 LIKE '%배송은%' 으로 매칭률이 떨어질 수 있으나,
    Step 1 의 GPT 추출 keywords 가 1차 방어선이므로 폴백 수준에서는 허용.
    """
    tokens = [t.strip() for t in question.split()]
    return [t for t in tokens if len(t) >= _MIN_TOKEN_LEN]


def _format_faq_answer(matches: list[dict]) -> str:
    """매칭 FAQ 리스트 → 사용자 응답 문자열 포맷

    - matches[0] : 본문으로 전체 노출
    - matches[1:] : 질문 제목만 '관련 FAQ' 로 추가 안내
    """
    top = matches[0]
    lines = [f"[FAQ] {top['question']}", "", top["answer"]]

    related = matches[1:]
    if related:
        lines.append("")
        lines.append("📌 이런 질문도 있어요:")
        for item in related:
            lines.append(f"  • {item['question']}")

    return "\n".join(lines)


def _search_faq_sync(question: str, intent_result: IntentResult) -> str:
    """동기 검색 로직 본체 (oracledb 가 동기 드라이버이므로 분리)

    asyncio.to_thread 로 감싸서 호출 → 이벤트 루프 블로킹 방지
    """
    # ── Step 1: GPT 추출 키워드 검색 ─────────────────────────────
    keywords = intent_result.entities.keywords or []
    matches = search_faq_by_keywords(keywords) if keywords else []

    # ── Step 2: 폴백 - 원본 질문 토큰 검색 ──────────────────────
    if not matches:
        tokens = _tokenize_question(question)
        # Step 1 과 동일 키워드면 결과도 동일하므로 중복 검색 방지
        fallback_tokens = [t for t in tokens if t not in keywords]
        if fallback_tokens:
            matches = search_faq_by_keywords(fallback_tokens)

    # ── Step 3: 매칭 실패 → 고정 문구 (GPT 폴백 없음) ───────────
    if not matches:
        return _FAQ_NOT_FOUND_MESSAGE

    return _format_faq_answer(matches)


async def handle_faq(question: str, intent_result: IntentResult) -> str:
    """FAQ 인텐트 핸들러 진입점

    router.py HANDLER_MAP 시그니처 규약 준수:
        async (question: str, intent_result: IntentResult) -> str

    oracledb 는 동기 드라이버이므로 to_thread 로 워커 스레드에 위임하여
    FastAPI 이벤트 루프가 DB I/O 동안 블로킹되지 않도록 한다.
    """
    return await asyncio.to_thread(_search_faq_sync, question, intent_result)
