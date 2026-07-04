"""
scripts/test_integration.py [LEGACY — 레거시 pipeline/router.py 대상 통합 테스트 스크립트]
현재 운영 경로(graph/ 기반)는 tests/test_graph_routing.py 등이 커버한다.

실행 방법 (둘 다 동작):
    프로젝트 루트(shoppingmall_ai/)에서
        python scripts/test_integration.py        # 직접 실행
        python -m scripts.test_integration         # 모듈 실행
    .env 에 OPENAI_API_KEY / DB_USER / DB_PASSWORD / DB_DSN / CHROMA_HOST / CHROMA_PORT 필요.

    ※ 'python scripts/test_integration.py' 로 직접 실행하면
      파이썬이 sys.path[0] 에 스크립트 폴더(scripts/)를 넣기 때문에 프로젝트 루트의
      pipeline/ 패키지를 찾지 못해 ModuleNotFoundError 가 발생한다.
      아래 부트스트랩으로 프로젝트 루트를 sys.path 에 직접 추가하여 두 실행 방식 모두 지원한다.

[구성]
1) 환각 가드 로직 단위 검증 (deterministic, 외부 호출 없음)
   - hallucination_guard 의 내부 검증 함수(_validate_semantic_answer, _has_factual_claim,
     _extract_prices)를 mock 데이터로 직접 호출. OpenAI/Oracle/ChromaDB 호출이 전혀
     없으므로 외부 서비스 없이도 가드의 "판정 로직" 자체가 맞는지 확인 가능.
   - 밑줄(_) 접두 함수를 직접 import 하는 것은 화이트박스 테스트 목적의 의도적 선택.
   - 이 섹션의 실패 개수만 프로세스 종료코드(exit code)에 반영한다 (CI 친화).

2) 6개 인텐트 풀파이프라인 시나리오 (외부 서비스 의존)
   - classify_intent → route_intent → guard_answer 전체 파이프라인을 실제로 실행.
   - 각 시나리오는 독립적으로 try/except 로 감싸 한 시나리오 실패(예: ChromaDB 미기동으로
     SEMANTIC 실패)가 나머지 시나리오 실행을 막지 않도록 함.
   - 환경 의존적(도커/DB 기동 여부)이므로 종료코드에는 반영하지 않고 콘솔에만 표시한다.
"""
import os
import sys

# ── 프로젝트 루트를 sys.path 에 추가 (직접 실행 시 import 보장) ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import asyncio

from pipeline.intent_classifier import classify_intent
from pipeline.router import route_intent
from pipeline.hallucination_guard import (
    guard_answer,
    _validate_semantic_answer,
    _has_factual_claim,
    _extract_prices,
)


# ═════════════════════════════════════════════════════════════════════════
# 콘솔 출력 + 결과 집계 헬퍼
# ═════════════════════════════════════════════════════════════════════════

class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"


# 단위 검증 통과/실패 카운터 (exit code 산정용)
_unit_pass = 0
_unit_fail = 0


def _ok(msg: str) -> None:
    print(f"{_C.GREEN}✅ {msg}{_C.RESET}")


def _fail(msg: str) -> None:
    print(f"{_C.RED}❌ {msg}{_C.RESET}")


def _info(msg: str) -> None:
    print(f"{_C.CYAN}{msg}{_C.RESET}")


def _section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{_C.BOLD}{bar}\n {title}\n{bar}{_C.RESET}")


def _check(condition: bool, msg: str) -> None:
    """단위 검증 1건: 결과를 카운트하고 색상 출력한다."""
    global _unit_pass, _unit_fail
    if condition:
        _unit_pass += 1
        _ok(msg)
    else:
        _unit_fail += 1
        _fail(msg)


# ═════════════════════════════════════════════════════════════════════════
# 1) 환각 가드 로직 단위 검증 (mock 데이터, 외부 호출 없음)
# ═════════════════════════════════════════════════════════════════════════

# rag_service.build_product_context / chroma_service.search_similar 결과 형식과 동일한 mock hits
_MOCK_HITS = [
    {
        "id": "p1",
        "document": "겨울에 따뜻하게 입을 수 있는 롱패딩입니다.",
        "metadata": {"product_name": "롱패딩 자켓", "category": "아우터", "price": 89000},
        "distance": 0.12,
    },
    {
        "id": "p2",
        "document": "가볍게 걸치기 좋은 경량 패딩 베스트입니다.",
        "metadata": {"product_name": "경량 패딩 베스트", "category": "아우터", "price": 45000},
        "distance": 0.20,
    },
]


def test_guard_logic() -> None:
    _section("1. 환각 가드 로직 단위 검증 (mock 데이터, API/DB 호출 없음)")

    # ── 1-1) _extract_prices ────────────────────────────────────────
    extracted = _extract_prices("이 상품은 12,345원이고, 다른 상품은 6000원입니다.")
    expected = {12345, 6000}
    _check(extracted == expected, f"가격 추출: {extracted} (기대 {expected})")

    # ── 1-2) _validate_semantic_answer (SEMANTIC 환각 검증) ─────────
    semantic_cases = [
        (
            "정상: 컨텍스트 가격/상품명 일치",
            "롱패딩 자켓을 추천해요. 가격은 89,000원입니다.",
            True,
        ),
        (
            "정상: 가격 없이 상품명만 언급 (paraphrase 토큰 매칭)",
            "이번 겨울엔 롱패딩 어떠세요? 따뜻하게 보내실 수 있어요.",
            True,
        ),
        (
            "환각 의심: 컨텍스트에 없는 가격",
            "롱패딩 자켓을 추천해요. 가격은 120,000원입니다.",
            False,
        ),
        (
            "정상: 컨텍스트와 무관해 정중히 거절",
            "죄송하지만 조건에 맞는 관련 상품이 없어서 추천하기 어려워요.",
            True,
        ),
        (
            "환각 의심: 가격/상품명 모두 컨텍스트와 무관한 추천",
            "가성비 좋은 운동화를 추천드려요!",
            False,
        ),
    ]
    for name, answer, expected_result in semantic_cases:
        result = _validate_semantic_answer(answer, _MOCK_HITS)
        _check(result == expected_result, f"{name} → 검증결과={result} (기대={expected_result})")

    # ── 1-3) _has_factual_claim (COMPLAINT/SMALL_TALK 경량 검증) ────
    fact_claim_cases = [
        ("주문번호 직접 언급", "ORD-20260601-0001 주문은 환불 완료되었습니다.", True),
        ("배송완료 단정", "고객님 주문은 배송이 완료되었습니다.", True),
        ("내일 도착 단정", "주문하신 상품은 내일 도착할 예정입니다.", True),
        ("일반 공감 (사실 단정 없음)", "정말 불편하셨겠어요, 진심으로 사과드립니다.", False),
    ]
    for name, answer, expected_result in fact_claim_cases:
        result = _has_factual_claim(answer)
        _check(result == expected_result, f"{name} → 감지결과={result} (기대={expected_result})")


# ═════════════════════════════════════════════════════════════════════════
# 2) 6개 인텐트 풀파이프라인 시나리오
# ═════════════════════════════════════════════════════════════════════════

# (라벨, 질문, 검증함수|None) - 정상 케이스 + 엣지 케이스
# 검증함수가 있으면 classify_intent 직후 intent_result 를 받아 _check() 로 결정적 단위 검증을 1건 추가한다
# (이 부분은 temperature=0 으로 결정적이므로, 일반 시나리오와 달리 exit code 에도 반영된다)
def _check_price_min_extraction(intent_result) -> None:
    """'이상' 표현이 price_min 으로 추출되는지 검증 (price_max 오추출 회귀 방지)."""
    _check(
        intent_result.entities.price_min == 3000000,
        f"'300만원 이상' → price_min=3000000 추출 "
        f"(entities={intent_result.entities.model_dump(exclude_none=True)})",
    )


SCENARIOS: list[tuple[str, str, "object"]] = [
    ("STRUCTURED_QUERY", "5만원 이하 신발 보여줘", None),
    ("STRUCTURED_QUERY (0건 엣지)", "300만원 이상 신발 보여줘", _check_price_min_extraction),
    ("SEMANTIC_SEARCH", "겨울에 따뜻하게 입을만한 옷 추천해줘", None),
    ("FAQ", "배송은 며칠 걸려요?", None),
    ("FAQ (매칭 실패 가능 엣지)", "포장 선물용으로 따로 포장도 해주나요?", None),
    ("ORDER_INQUIRY (전체 목록)", "내 주문 내역 보여줘", None),
    ("ORDER_INQUIRY (단건, order_id 추출)", "ORD-20260601-0001 주문 배송 상태 알려줘", None),
    ("ORDER_INQUIRY (존재하지 않는 주문번호 엣지)", "ORD-99999999-9999 주문 어디까지 왔어?", None),
    ("COMPLAINT", "주문한 옷이 파손돼서 왔어요. 정말 화가 나네요.", None),
    ("SMALL_TALK", "안녕! 너는 누구야?", None),
]

# 시나리오 성공/실패 카운터 (콘솔 표시용 - exit code 에는 미반영)
_scenario_ok = 0
_scenario_fail = 0


async def run_scenario(label: str, question: str, validate=None) -> None:
    global _scenario_ok, _scenario_fail
    print(f"\n{_C.BOLD}--- {label} ---{_C.RESET}")
    print(f"Q: {question}")
    try:
        intent_result = await classify_intent(question)
        entities = intent_result.entities.model_dump(exclude_none=True, exclude_defaults=True)
        _info(
            f"  intent={intent_result.intent.value} "
            f"confidence={intent_result.confidence:.2f} "
            f"entities={entities}"
        )

        if validate is not None:
            validate(intent_result)

        raw_answer = await route_intent(question, intent_result)
        final_answer = await guard_answer(question, raw_answer, intent_result)

        print(f"A: {final_answer}")
        _scenario_ok += 1
        _ok(f"{label} 완료")
    except Exception as e:
        _scenario_fail += 1
        _fail(f"{label} 실패: {type(e).__name__}: {e}")


async def run_full_pipeline_scenarios() -> None:
    _section("2. 6개 인텐트 풀파이프라인 시나리오 (실제 OpenAI/Oracle/ChromaDB 호출)")
    _info(
        "※ 일부 시나리오가 실패해도(예: ChromaDB 미기동 → SEMANTIC 실패) "
        "나머지 시나리오는 계속 진행됩니다.\n"
        "※ ORDER_INQUIRY (단건) 시나리오에서 entities 에 order_id 가 보이지 않으면, "
        "분류기 프롬프트의 order_id 추출 규칙을 다시 점검하세요."
    )
    for label, question, validate in SCENARIOS:
        await run_scenario(label, question, validate)


# ═════════════════════════════════════════════════════════════════════════
# 진입점
# ═════════════════════════════════════════════════════════════════════════

def _print_summary() -> None:
    _section("테스트 요약")
    print(
        f"{_C.BOLD}[단위 검증]{_C.RESET} "
        f"{_C.GREEN}{_unit_pass} 통과{_C.RESET} / "
        f"{_C.RED}{_unit_fail} 실패{_C.RESET}  (← exit code 산정 대상)"
    )
    print(
        f"{_C.BOLD}[파이프라인 시나리오]{_C.RESET} "
        f"{_C.GREEN}{_scenario_ok} 완료{_C.RESET} / "
        f"{_C.YELLOW}{_scenario_fail} 실패(환경 의존, exit code 미반영){_C.RESET}"
    )


async def main() -> None:
    test_guard_logic()
    await run_full_pipeline_scenarios()
    _print_summary()


if __name__ == "__main__":
    asyncio.run(main())
    # 단위 검증(deterministic)이 하나라도 실패하면 비정상 종료코드 반환 (CI/자동화 친화)
    sys.exit(1 if _unit_fail else 0)