"""
pipeline/structured_handler.py
STRUCTURED_QUERY 인텐트 핸들러

- 동적 SQL(oracle_db.search_products_structured)로 상품을 정밀 검색
- GPT 미사용: 검색 결과를 템플릿으로 직접 포맷 (환각 가능성 없음 → 환각 방어 대상 아님)
- 비동기: 동기 DB 함수를 asyncio.to_thread 로 감싸 FastAPI 이벤트 루프를 블로킹하지 않음
"""
import asyncio

from database.oracle_db import search_products_structured
from schemas.intent_schema import IntentResult

# 응답 프리픽스 (SEMANTIC "[추천]" 과 구분되는 STRUCTURED 식별자)
# ※ 프론트/라우터에서 인텐트별 렌더링 분기에 쓰일 수 있으므로 단순/고정 문자열 유지
_PREFIX = "[검색결과]"

# 결과 최대 건수 (챗봇 UI 가독성 → 5건)
_RESULT_LIMIT = 5

# 0건일 때 안내 메시지
_NO_RESULT_MSG = (
    "조건에 맞는 상품을 찾지 못했어요. 😢\n"
    "카테고리나 가격 조건을 조금 바꿔서 다시 검색해 보시겠어요?"
)


def _format_price(price) -> str:
    """가격을 천단위 콤마 + '원' 형식으로. None 이면 '가격문의'"""
    if price is None:
        return "가격문의"
    return f"{int(price):,}원"


def _format_product_line(idx: int, product: dict) -> str:
    """상품 1건을 응답 텍스트 블록으로 변환 (품절 시 [품절] 태그)"""
    name = product.get("product_name", "이름없음")
    category = product.get("category", "-")
    price_text = _format_price(product.get("price"))
    stock = product.get("stock")

    # 품절 판정: stock 이 0 이하 (음수 방어 포함)
    soldout = stock is not None and stock <= 0
    soldout_tag = " [품절]" if soldout else ""
    stock_text = f"{int(stock)}개" if stock is not None else "정보없음"

    return (
        f"{idx}. {name}{soldout_tag}\n"
        f"   카테고리: {category} | 가격: {price_text} | 재고: {stock_text}"
    )


async def handle_structured(question: str, intent_result: IntentResult) -> str:
    """STRUCTURED_QUERY 처리 진입점

    Args:
        question      : 사용자 원본 질문
                        (현재 템플릿 응답엔 직접 쓰지 않지만, 로깅/향후 확장 대비 + SEMANTIC 핸들러와
                         시그니처 통일을 위해 수신)
        intent_result : 인텐트 분류기 결과. intent_result.entities 는 항상 Entities
                        pydantic 인스턴스(default_factory=Entities)이므로 dict 분기 불필요
                        (기존 _get_entity dict/object 양쪽 대응 헬퍼는 제거됨 — IntentResult 만 받는다)

    Returns:
        str : "[검색결과]" 프리픽스가 붙은 사용자 응답 텍스트
    """
    entities = intent_result.entities
    category = entities.category
    price_min = entities.price_min
    price_max = entities.price_max
    keywords = entities.keywords
    sort_by = entities.sort_by

    # 동기 DB 함수를 비동기 컨텍스트에서 안전 실행 (이벤트 루프 비블로킹)
    products = await asyncio.to_thread(
        search_products_structured,
        category=category,
        price_min=price_min,
        price_max=price_max,
        keywords=keywords,
        sort_by=sort_by,
        limit=_RESULT_LIMIT,
    )

    # 0건 → 고정 안내
    if not products:
        return f"{_PREFIX} {_NO_RESULT_MSG}"

    # 헤더 + 상품 목록 조립
    header = f"🛍️ 조건에 맞는 상품 {len(products)}개를 찾았어요!"
    lines = [_format_product_line(i + 1, p) for i, p in enumerate(products)]
    body = "\n\n".join(lines)

    return f"{_PREFIX} {header}\n\n{body}"
