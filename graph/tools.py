"""
graph/tools.py
단일 Agent 가 자율 호출하는 도구(Tool) 모음.

[설계 원칙]
- 기존 핸들러 로직(database.oracle_db, services.*, pipeline.faq_handler)을
  재사용한다. Tool 은 '얇은 래퍼'일 뿐 새 검색 로직을 만들지 않는다.
- Tool 반환은 LLM 이 읽는 '문자열'이어야 한다. (dict 반환 금지)
- 서버 측 값(member_id/is_guest)은 LLM 이 모르므로 InjectedState 로 주입한다.
  LLM 에게 인자로 받게 두면 환각/위변조 위험.
- semantic 도구의 rag_hits(dict 리스트)는 문자열로 표현 불가하므로,
  Command(update=...) 로 State["rag_hits"] 에 기록한다 → guard 노드가 읽는다.

[도구 목록]
  1) search_products       : 구조화 상품검색 (카테고리/가격/정렬)
  2) semantic_search       : 의미 기반 상품검색 (ChromaDB RAG) + rag_hits 기록
  3) search_faq            : FAQ 검색
  4) get_my_orders         : 주문조회 (InjectedState member_id, 게스트 차단)
  5) request_refund        : 환불 신청 (Human-in-the-loop: interrupt 로 사용자 확인)

[Human-in-the-loop — request_refund]
- 환불처럼 '되돌리기 어려운 민감 작업'은 Agent 가 자율 실행하지 않고,
  langgraph.types.interrupt 로 그래프를 '일시정지'시켜 사용자 확인을 받는다.
- interrupt(payload) 호출 시 그래프가 멈추고 결과에 __interrupt__ 가 실린다.
  → 라우터(agent_chat)가 이를 감지해 confirm 요청을 클라이언트에 내려보낸다.
  → 사용자가 승인/거부하면 Command(resume=...) 로 같은 thread 를 재개한다.
- 재개가 필요하므로 '영속 thread_id'가 필수다 → 로그인 회원만 허용(게스트 차단).
  (게스트는 요청마다 1회성 thread_id 라 resume 자체가 불가능)
"""
import asyncio
import logging
from typing import Annotated, Optional

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt

from schemas.intent_schema import IntentResult, IntentType, Entities
from database.oracle_db import search_products_structured, fetch_orders, fetch_order_by_id
from services import rag_service
from services import notification_service
from pipeline.faq_handler import _search_faq_sync

# 기존 노드의 포맷터를 재사용 (중복 구현 방지)
from graph.nodes import (
    _format_product_line,
    _STRUCTURED_NO_RESULT_MSG,
    _SEMANTIC_NO_HIT_MSG,
    SEMANTIC_TOP_K,
    _format_order_list,
    _format_order_detail,
    _format_order_not_found,
    _GUEST_ORDER_BLOCKED_MESSAGE,
)

logger = logging.getLogger(__name__)

_STRUCTURED_LIMIT = 5


# ════════════════════════════════════════════════════════════════════════
# 1) 구조화 상품검색
# ════════════════════════════════════════════════════════════════════════
@tool
async def search_products(
    category: Optional[str] = None,
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    keywords: Optional[list[str]] = None,
    sort_by: Optional[str] = None,
) -> str:
    """카테고리·가격대·정렬 조건으로 상품을 검색한다.

    명확한 조건(예: '5만원 이하 신발', '가격 낮은 순')이 있을 때 사용한다.
    Args:
        category: 상품 카테고리 (예: '신발', '상의'). 없으면 전체.
        price_min: 최소 가격(원). 가격 하한이 있을 때만.
        price_max: 최대 가격(원). 가격 상한이 있을 때만.
        keywords: 상품명에 포함될 키워드 목록.
        sort_by: 정렬 기준 ('PRICE_ASC'|'PRICE_DESC'|'LATEST'|'DEFAULT'). 대소문자 무관.
            (router 경로의 SortType enum 과 동일 값. 지정 안 하면 상품ID 오름차순)
    Returns:
        조건에 맞는 상품 목록 문자열. 없으면 안내 문구.
    """
    products = await asyncio.to_thread(
        search_products_structured,
        category=category,
        price_min=price_min,
        price_max=price_max,
        keywords=keywords,
        sort_by=sort_by,
        limit=_STRUCTURED_LIMIT,
    )
    if not products:
        return _STRUCTURED_NO_RESULT_MSG
    header = f"조건에 맞는 상품 {len(products)}개:"
    lines = [_format_product_line(i + 1, p) for i, p in enumerate(products)]
    return header + "\n\n" + "\n\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# 2) 의미 기반 상품검색 (ChromaDB RAG) — rag_hits 를 State 에 기록
# ════════════════════════════════════════════════════════════════════════
@tool
async def semantic_search(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """추상적·감성적 상품 요청을 의미 기반으로 검색한다.

    조건이 모호하거나 분위기/용도 중심일 때 사용한다.
    (예: '겨울에 따뜻하게 입을 옷', '데이트룩 추천', '선물하기 좋은 거')

    Args:
        query: 사용자의 의미 기반 검색 의도(자연어).
    Returns:
        검색된 상품 컨텍스트 문자열. (내부적으로 rag_hits 를 State 에 기록하여
        이후 환각 가드가 검증에 사용한다.)
    """
    # [RAG 고도화] 검색+재랭킹 공통 파이프라인 사용 (semantic_node 와 동일 흐름)
    from graph.rag_pipeline import search_and_rerank
    hits = await search_and_rerank(query, top_n=SEMANTIC_TOP_K)

    if not hits:
        return Command(update={
            "rag_hits": [],
            "messages": [ToolMessage(_SEMANTIC_NO_HIT_MSG, tool_call_id=tool_call_id)],
        })

    # LLM 에게는 '상품 컨텍스트 문자열'을 돌려주고, hits 원본은 State 에 기록.
    context = rag_service.build_product_context(hits)
    return Command(update={
        "rag_hits": hits,
        "messages": [ToolMessage(context, tool_call_id=tool_call_id)],
    })


# ════════════════════════════════════════════════════════════════════════
# 3) FAQ 검색
# ════════════════════════════════════════════════════════════════════════
@tool
async def search_faq(question: str) -> str:
    """배송/교환/환불/회원 등 자주 묻는 질문(FAQ)을 검색한다.

    상품 검색이 아니라 '정책/이용안내'성 질문일 때 사용한다.
    (예: '배송 얼마나 걸려요?', '환불 규정 알려줘')

    Args:
        question: 사용자의 FAQ 질문(자연어).
    Returns:
        매칭된 FAQ 답변 문자열. 없으면 안내 문구.
    """
    # _search_faq_sync 는 intent_result.entities.keywords 를 보지만,
    # 없으면 질문 토큰으로 폴백하므로 빈 IntentResult 를 넘기면 된다.
    empty_intent = IntentResult(
        intent=IntentType.FAQ, entities=Entities(), confidence=1.0,
    )
    return await asyncio.to_thread(_search_faq_sync, question, empty_intent)


# ════════════════════════════════════════════════════════════════════════
# 4) 주문조회 — member_id 는 InjectedState, 게스트 차단
# ════════════════════════════════════════════════════════════════════════
@tool
async def get_my_orders(
    state: Annotated[dict, InjectedState],
    order_id: Optional[str] = None,
) -> str:
    """로그인한 회원의 주문 내역을 조회한다.

    '내 주문', '배송 어디까지 왔어', 특정 주문번호 조회 등에 사용한다.
    member_id 는 서버가 자동 주입하므로 LLM 이 지정할 필요가 없다.

    Args:
        order_id: 특정 주문번호(숫자 문자열, 예: '3'). 없으면 전체 목록.
    Returns:
        주문 목록/상세 문자열. 게스트면 로그인 안내.
    """
    # 게스트 차단 (InjectedState 로 받은 서버 값으로 판정)
    if state.get("is_guest") or not state.get("member_id"):
        return _GUEST_ORDER_BLOCKED_MESSAGE

    member_id = state["member_id"]
    order_id = order_id.strip() if isinstance(order_id, str) else None
    if order_id:
        order = await asyncio.to_thread(fetch_order_by_id, member_id, order_id)
        return _format_order_detail(order) if order else _format_order_not_found(order_id)
    orders = await asyncio.to_thread(fetch_orders, member_id)
    return _format_order_list(orders)


# ════════════════════════════════════════════════════════════════════════
# 5) 환불 신청 — Human-in-the-loop (interrupt 로 사용자 확인 후 진행)
# ════════════════════════════════════════════════════════════════════════
_REFUND_GUEST_BLOCKED_MSG = (
    "환불 신청은 로그인 후 이용하실 수 있어요. 로그인 뒤 다시 시도해 주세요."
)


@tool
async def request_refund(
    order_id: str,
    state: Annotated[dict, InjectedState],
    reason: Optional[str] = None,
) -> str:
    """주문에 대한 환불을 '신청'한다 (사용자 최종 확인 후 접수).

    환불은 되돌리기 어려운 민감 작업이라, 이 도구는 곧바로 접수하지 않고
    먼저 사용자에게 진행 여부를 확인받는다(Human-in-the-loop). 사용자가
    승인해야만 접수되고, 거부하면 취소된다.

    '환불하고 싶어', '3번 주문 환불해줘' 같은 환불 의사가 명확할 때 사용한다.

    Args:
        order_id: 환불할 주문번호.
        reason: 환불 사유(선택). 사용자가 사유를 말했으면 담아 전달.
    Returns:
        접수/취소 결과 문자열.
    """
    # 게스트 차단 — 재개(resume)에 필요한 영속 thread_id 가 없으므로 원천 차단.
    if state.get("is_guest") or not state.get("member_id"):
        return _REFUND_GUEST_BLOCKED_MSG

    oid = order_id.strip() if isinstance(order_id, str) else ""
    if not oid:
        return "환불할 주문번호를 알려주세요."

    # [검증] interrupt(사용자 확인)를 띄우기 전에 주문이 실제로 존재하고 '본인 것'인지
    # 먼저 확인한다. fetch_order_by_id 는 member_id 로 스코프되므로, 없는 주문이거나
    # 타인 주문이면 None 을 돌려준다. 존재하지 않는 주문에 환불 확인을 띄우는 것을 막는다.
    member_id = state["member_id"]
    order = await asyncio.to_thread(fetch_order_by_id, member_id, oid)
    if order is None:
        return _format_order_not_found(oid)

    # ── interrupt: 그래프를 멈추고 사용자 확인을 기다린다 ──────────────────
    # payload 가 결과의 __interrupt__ 로 실려 클라이언트에 전달된다.
    # 사용자의 응답(Command(resume=...))이 decision 으로 들어온다.
    decision = interrupt({
        "type": "confirm_refund",
        "order_id": oid,
        "reason": reason or "",
        "prompt": f"주문 {oid}에 대한 환불을 신청할까요? 진행하려면 승인해 주세요.",
    })

    # resume 값 해석: 문자열 'approve'/'reject' 또는 dict({"approved": bool}) 모두 허용.
    approved = False
    if isinstance(decision, str):
        approved = decision.strip().lower() in ("approve", "yes", "y", "승인", "true")
    elif isinstance(decision, dict):
        approved = bool(decision.get("approved"))
    elif isinstance(decision, bool):
        approved = decision

    if not approved:
        return f"주문 {oid} 환불 신청을 취소했어요. 변동 사항은 없습니다."

    # [주의] 실제 ORDERS 상태 변경/환불 처리는 쇼핑몰(Spring Boot) 책임 영역이다.
    #        이 도구는 'AI 챗봇을 통한 환불 신청 접수' 단계만 시뮬레이션한다.
    #        DB 쓰기를 하지 않으므로 스키마/Spring Boot 에 영향이 없다.
    logger.info("환불 신청 접수: member_id=%s order_id=%s reason=%r",
                member_id, oid, reason)
    # 관리자 알림(best-effort) — 실패해도 사용자 응답(접수 완료 안내)은 그대로 진행.
    await notification_service.send_refund_admin_email(oid, member_id, reason)
    suffix = f" (사유: {reason})" if reason else ""
    return (
        f"주문 {oid}에 대한 환불 신청이 접수되었습니다{suffix}. "
        "처리 현황은 마이페이지 주문내역에서 확인하실 수 있어요."
    )


# Agent 에 바인딩할 도구 목록 (등록 순서가 LLM 의 선택 우선순위에 영향 없음)
ALL_TOOLS = [search_products, semantic_search, search_faq, get_my_orders, request_refund]
