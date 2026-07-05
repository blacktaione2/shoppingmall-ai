"""
graph/nodes.py
LangGraph 노드 함수 모음.

[노드 계약]
- 각 노드는 async def node(state: ShoppingState) -> dict
  · 입력: 현재 State
  · 반환: '갱신할 키'만 담은 dict (LangGraph 가 자동 병합)
- GPT 사용 노드는 내부에서 LangChain LCEL 체인(prompt | llm | parser)을 호출.
- GPT 미사용 노드(structured/faq/order)는 기존 핸들러 로직/서비스 함수를 그대로 재사용.

[기존 패턴 대체 매핑]
- ContextVar set_rag_context → semantic_node 가 state["rag_hits"] 로 반환
- ContextVar 멀티턴 history   → 각 노드가 state["history"] 를 직접 사용
- 프리픽스([추천]/[검색결과] 등) → 전부 제거 (intent 는 state.intent_result 로 식별)
"""
import asyncio
import logging
import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from schemas.intent_schema import IntentResult, IntentType, coerce_intent_result
from graph.state import ShoppingState
from graph.llm import (
    get_main_llm,
    get_intent_llm,
    date_system_prefix,
    history_to_messages,
)
from graph.model_policy import select_llm, TaskComplexity

# 기존 분류 프롬프트 재사용 (단일 출처)
from pipeline.intent_classifier import INTENT_SYSTEM_PROMPT

# GPT 미사용 핸들러 로직 재사용
from database.oracle_db import search_products_structured
from services import rag_service
from pipeline.faq_handler import _search_faq_sync
from database.oracle_db import fetch_orders, fetch_order_by_id, fetch_all_products

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# 1) classify_node — 인텐트 분류 (gpt-5.4-mini + Structured Output)
# ════════════════════════════════════════════════════════════════════════
async def classify_node(state: ShoppingState) -> dict:
    """사용자 질문을 6개 인텐트 중 하나로 분류한다.

    [변경] 기존 services.gpt_service.structured_completion(OpenAI SDK 직접)
           → LangChain ChatOpenAI.with_structured_output(IntentResult).
           프롬프트(INTENT_SYSTEM_PROMPT)는 기존 것을 그대로 재사용한다.
    실패 시 기존 동작과 동일하게 SMALL_TALK(confidence=0.0) 폴백.
    """
    question = state["question"]
    history = state.get("history", [])
    try:
        llm = get_intent_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(IntentResult)
        prompt = ChatPromptTemplate.from_messages([
            ("system", INTENT_SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ])
        chain = prompt | structured_llm
        result: IntentResult = await chain.ainvoke({
            "question": question,
            "history": history_to_messages(history),
        })
        logger.info(
            "intent=%s confidence=%.2f question=%s",
            result.intent.value, result.confidence, question,
        )
        # 체크포인트 직렬화 안전성: State 에는 dict(primitive)만 기록한다.
        # mode="json" 이어야 IntentType/SortType enum 이 순수 문자열로 변환된다.
        return {"intent_result": result.model_dump(mode="json")}
    except Exception:
        logger.exception("인텐트 분류 실패, SMALL_TALK 폴백: %s", question)
        return {
            "intent_result": IntentResult(
                intent=IntentType.SMALL_TALK,
                confidence=0.0,
            ).model_dump(mode="json")
        }


# ════════════════════════════════════════════════════════════════════════
# 2) structured_node — STRUCTURED_QUERY (GPT 미사용)
# ════════════════════════════════════════════════════════════════════════
_STRUCTURED_RESULT_LIMIT = 5
_STRUCTURED_NO_RESULT_MSG = (
    "조건에 맞는 상품을 찾지 못했어요. 😢\n"
    "카테고리나 가격 조건을 조금 바꿔서 다시 검색해 보시겠어요?"
)


def _format_price(price) -> str:
    if price is None:
        return "가격문의"
    return f"{int(price):,}원"


def _format_product_line(idx: int, product: dict) -> str:
    name = product.get("product_name", "이름없음")
    category = product.get("category", "-")
    price_text = _format_price(product.get("price"))
    stock = product.get("stock")
    soldout = stock is not None and stock <= 0
    soldout_tag = " [품절]" if soldout else ""
    stock_text = f"{int(stock)}개" if stock is not None else "정보없음"
    return (
        f"{idx}. {name}{soldout_tag}\n"
        f"   카테고리: {category} | 가격: {price_text} | 재고: {stock_text}"
    )


async def structured_node(state: ShoppingState) -> dict:
    """동적 SQL 상품 검색 → 템플릿 포맷팅. (프리픽스 제거)"""
    entities = coerce_intent_result(state["intent_result"]).entities
    products = await asyncio.to_thread(
        search_products_structured,
        category=entities.category,
        price_min=entities.price_min,
        price_max=entities.price_max,
        keywords=entities.keywords,
        sort_by=entities.sort_by,
        limit=_STRUCTURED_RESULT_LIMIT,
    )
    if not products:
        return {"raw_answer": _STRUCTURED_NO_RESULT_MSG}

    header = f"🛍️ 조건에 맞는 상품 {len(products)}개를 찾았어요!"
    lines = [_format_product_line(i + 1, p) for i, p in enumerate(products)]
    body = "\n\n".join(lines)
    return {"raw_answer": f"{header}\n\n{body}"}


# ════════════════════════════════════════════════════════════════════════
# 3) semantic_node — SEMANTIC_SEARCH (ChromaDB + RAG, 멀티모델 팩토리)
# ════════════════════════════════════════════════════════════════════════
_CHEAP_MARKERS = ("제일 싼", "가장 싼", "제일 저렴", "가장 저렴")
_EXPENSIVE_MARKERS = ("제일 비싼", "가장 비싼", "제일 비싸", "가장 비싸")


def _detect_price_superlative(question: str) -> str | None:
    """질문에 최저가/최고가를 묻는 표현이 있으면 'cheap'/'expensive', 없으면 None."""
    if any(m in question for m in _CHEAP_MARKERS):
        return "cheap"
    if any(m in question for m in _EXPENSIVE_MARKERS):
        return "expensive"
    return None


def _match_history_products(history: list[dict]) -> list[dict]:
    """이전 봇 발화에 실제로 언급된 상품명을 카탈로그와 대조해 최신 정보로 재조회.

    [정확도 보강] "그 중 제일 싼 건?" 같은 가격 비교 질문은 LLM 자유생성이 숫자
    비교를 틀릴 수 있어(예: 히스토리 속 3개 중 실제 최저가가 아닌 걸 고름),
    이전 대화에 실제로 언급된 상품만 추려 파이썬으로 결정적으로 정렬한다.
    게스트도 client_history 를 매 요청 보내므로(state["history"]) 로그인 여부와
    무관하게 동작한다(체크포인터 전용 State 필드로 만들면 게스트는 혜택을 못 받음).
    """
    bot_text = " ".join(h.get("text", "") for h in history if h.get("role") == "bot")
    if not bot_text:
        return []
    all_products = fetch_all_products()
    # [버그 수정] 상품명이 텍스트에 있다는 것만으론 부족하다 — "조합 추천도 좋아요"처럼
    # 가격 없이 이름만 곁가지로 한 번 더 언급되는 경우까지 후보로 잡혀서, 원래
    # 추천 대상이 아니었던 상품이 최저가로 잘못 뽑히는 문제가 있었다. 가격도 같은
    # 텍스트에 함께 언급된 상품만 "실제로 추천된 상품"으로 인정한다.
    return [p for p in all_products
            if p.get("stock", 0) > 0
            and p["product_name"] in bot_text
            and _price_mentioned_exactly(p["price"], bot_text)]


def _to_rag_hit(product: dict) -> dict:
    """Oracle 상품 dict(플랫 구조) → 기존 ChromaDB hits 형식으로 변환.

    [버그 수정] guard.py 의 _validate_semantic_answer 는 hit["metadata"]["price"]/
    ["product_name"] 을 읽는 ChromaDB 형식을 기대하는데, fetch_all_products() 는
    플랫 dict 를 반환한다. 그대로 rag_hits 에 넣으면 가드가 가격/상품명 근거를
    하나도 못 읽어 정상 답변까지 매번 환각으로 오판(재시도 실패 → 안전문구 대체)한다.
    """
    return {
        "id": product.get("product_id"),
        "document": product.get("description") or "",
        "metadata": {
            "product_name": product.get("product_name"),
            "category": product.get("category"),
            "price": product.get("price"),
        },
        "distance": 0.0,
    }


def _price_mentioned_exactly(price, text: str) -> bool:
    """price 의 포맷 문자열이 text 안에 '더 큰 숫자의 일부'가 아니라 정확히 등장하는지 확인.

    [버그 수정] 단순 in 체크는 "35,000원"이 "135,000원"의 부분 문자열이라서
    통과해버려, 가격이 실제로 언급 안 된 상품(예: 조합 추천에 이름만 곁가지로
    나온 상품)이 우연히 다른 상품 가격의 일부와 겹쳐 후보로 잘못 인정될 수 있다.
    앞에 숫자가 이어지지 않는 위치에서만 매칭되도록 정규식 lookbehind 로 방지한다.
    """
    price_str = _format_price(price)
    pattern = r"(?<!\d)" + re.escape(price_str)
    return bool(re.search(pattern, text))


_REFERENCE_MARKERS = ("방금", "아까", "그거", "그것", "저거", "그 상품", "이거")
_PRICE_ASK_MARKERS = ("얼마", "가격")


def _detect_price_reference(question: str) -> bool:
    """'방금 말한 거 얼마야?' 처럼 직전 발화 속 단일 상품 가격을 되묻는 질문인지 판단."""
    return (any(m in question for m in _REFERENCE_MARKERS)
            and any(m in question for m in _PRICE_ASK_MARKERS))


def _match_last_bot_turn_products(history: list[dict]) -> list[dict]:
    """전체 히스토리가 아니라 '가장 최근 봇 발화 1개'에서만 상품을 추출.

    [비교 경로(_match_history_products)와의 차이] 비교 질문("그 중 제일 싼 건")은
    여러 턴에 걸쳐 언급된 후보 전체가 대상이지만, 단일 참조 질문("방금 말한 거")은
    바로 직전 봇 발화 하나만 봐야 한다 — 그렇지 않으면 훨씬 이전 턴의 상품과
    헷갈릴 수 있다.
    """
    bot_turns = [h.get("text", "") for h in history if h.get("role") == "bot"]
    if not bot_turns:
        return []
    all_products = fetch_all_products()
    # [버그 수정] _match_history_products 와 동일한 이유로, 가격도 같이
    # 언급된 상품만 인정한다.
    return [p for p in all_products
            if p.get("stock", 0) > 0
            and p["product_name"] in bot_turns[-1]
            and _price_mentioned_exactly(p["price"], bot_turns[-1])]


SEMANTIC_TOP_K = 4
_SEMANTIC_NO_HIT_MSG = (
    "죄송합니다, 조건에 맞는 상품을 찾지 못했어요. 다른 검색어로 다시 시도해 주세요."
)


async def semantic_node(state: ShoppingState) -> dict:
    """질문 임베딩 → ChromaDB top-k → RAG 응답 생성(rag_service → get_main_llm, 멀티모델 팩토리).

    [provider 일관성] 인텐트 분류(classify_node, get_intent_llm)와 RAG 답변 생성
    (rag_service.generate_rag_response → get_main_llm) 모두 .env LLM_PROVIDER 에 따라
    전환된다. 환각 재시도(graph/guard.py, get_intent_llm)까지 동일하게 전환되므로,
    라우터 경로 전체가 하나의 provider 로 일관되게 동작한다(provider 벤치마크 공정성).

    [핵심] hits 를 state["rag_hits"] 로 반환 → guard_node 가 그대로 읽는다.
           (기존 ContextVar set_rag_context 완전 대체)
    멀티턴 history 는 state["history"] 를 rag_service 에 직접 전달.
    """
    question = state["question"]
    history = state.get("history", [])

    # [정확도 보강] 이전에 언급된 상품 중 최저가/최고가를 묻는 질문은 LLM 대신
    # 결정적 비교로 답한다 (분류 규칙상 카테고리 등 구체 조건이 있으면 이미
    # structured_node 로 갔으므로, 여기 도달했다면 "모호한 참조"만 남은 상태).
    superlative = _detect_price_superlative(question)
    if superlative:
        mentioned = _match_history_products(history)
        if mentioned:
            mentioned.sort(key=lambda p: p["price"], reverse=(superlative == "expensive"))
            target = mentioned[0]
            label = "가장 저렴한" if superlative == "cheap" else "가장 비싼"
            answer = (f"이전에 안내해드린 상품 중 {label} 건 "
                      f"{target['product_name']}({_format_price(target['price'])})이에요.")
            return {"rag_hits": [_to_rag_hit(p) for p in mentioned], "raw_answer": answer}

    # [정확도 보강] "방금 말한 거 얼마야?" 처럼 직전 발화 속 단일 상품을 재확인하는
    # 질문도 임베딩 검색(질문 자체가 상품명과 무관해 엉뚱한 상품을 끌고 올 수 있음)
    # 대신 직전 봇 발화에서 결정적으로 추출한다. 상품이 2개 이상 언급돼 모호하면
    # 억지로 추측하지 않고 기존 경로로 안전하게 폴백한다.
    if not superlative and _detect_price_reference(question):
        recent = _match_last_bot_turn_products(history)
        if len(recent) == 1:
            target = recent[0]
            answer = (f"방금 말씀드린 상품은 {target['product_name']}이고, "
                      f"가격은 {_format_price(target['price'])}입니다.")
            return {"rag_hits": [_to_rag_hit(target)], "raw_answer": answer}

    # [RAG 고도화] 검색+재랭킹 공통 파이프라인 사용 (라우터/Agent 일관성)
    # 로그인 회원이면 member_id 를 넘겨 취향 벡터 혼합(개인화 ON 시).
    #             게스트(member_id=None)면 기존과 동일한 순수 질문 검색.
    from graph.rag_pipeline import search_and_rerank
    hits = await search_and_rerank(
        question, top_n=SEMANTIC_TOP_K, member_id=state.get("member_id"),
    )

    # 0건 → GPT 호출 없이 고정 안내 (rag_hits 는 빈 리스트로 반환)
    if not hits:
        return {"rag_hits": [], "raw_answer": _SEMANTIC_NO_HIT_MSG}

    answer = await rag_service.generate_rag_response(question, hits, history=history)
    return {"rag_hits": hits, "raw_answer": answer}


# ════════════════════════════════════════════════════════════════════════
# 4) faq_node — FAQ (GPT 미사용)
# ════════════════════════════════════════════════════════════════════════
async def faq_node(state: ShoppingState) -> dict:
    """FAQ DB 검색 (기존 _search_faq_sync 재사용, 동기 → to_thread).

    [주의] 기존 _format_faq_answer 는 본문 앞에 '[FAQ] {질문}' 을 붙인다.
           이는 프리픽스가 아니라 'FAQ 질문 제목'을 보여주는 본문 일부이므로
           strip_known_prefix 제거 대상과 성격이 다르다. 그대로 둔다.
    """
    intent_result = coerce_intent_result(state["intent_result"])
    question = state["question"]
    answer = await asyncio.to_thread(_search_faq_sync, question, intent_result)
    return {"raw_answer": answer}


# ════════════════════════════════════════════════════════════════════════
# 5) order_node — ORDER_INQUIRY (GPT 미사용)
# ════════════════════════════════════════════════════════════════════════
_ORDER_STATUS_EMOJI = {
    "결제대기": "💳",
    "결제완료": "💰",
    "배송준비중": "⏳",
    "배송중": "🚚",
    "배송완료": "✅",
    "주문취소": "❌",
    "환불완료": "↩️",
}
_ORDER_FORMAT_ERROR_MSG = (
    "주문 정보를 표시하는 중 오류가 발생했어요. 잠시 후 다시 시도해 주시거나 "
    "고객센터(1234-5678)로 문의해 주세요."
)


def _format_items(items: list[dict]) -> str:
    """주문 상품 목록 포맷팅.

    필드 접근은 .get() + 기본값으로 방어한다 — 조인 결과에 NULL 이 섞여도
    (예: 삭제된 상품) KeyError 로 응답 전체가 죽지 않고 나머지 정보는 표시한다.
    """
    lines = []
    for item in items or []:
        name = item.get("product_name") or "상품정보없음"
        quantity = item.get("quantity")
        qty_text = f"x{quantity}" if quantity is not None else ""
        price = item.get("price")
        price_text = f"({price:,}원)" if isinstance(price, (int, float)) else "(가격정보없음)"
        lines.append(f"   - {name} {qty_text} {price_text}".replace("  ", " ").rstrip())
    return "\n".join(lines)


def _format_order_list(orders: list[dict]) -> str:
    if not orders:
        return "주문 내역이 없습니다."
    blocks = [f"회원님의 주문 내역 {len(orders)}건입니다."]
    for order in orders:
        status = order.get("status") or "상태정보없음"
        emoji = _ORDER_STATUS_EMOJI.get(status, "📦")
        total_price = order.get("total_price")
        total_text = f"{total_price:,}원" if isinstance(total_price, (int, float)) else "정보없음"
        blocks.append(
            f"📦 **{order.get('order_id', '-')}** ({order.get('order_date', '-')})\n"
            f"{_format_items(order.get('items') or [])}\n"
            f"   - 총 결제금액: {total_text}\n"
            f"   - 배송상태: {status} {emoji}"
        )
    return "\n\n".join(blocks)


def _format_order_detail(order: dict) -> str:
    status = order.get("status") or "상태정보없음"
    emoji = _ORDER_STATUS_EMOJI.get(status, "📦")
    total_price = order.get("total_price")
    total_text = f"{total_price:,}원" if isinstance(total_price, (int, float)) else "정보없음"
    return (
        f"주문번호 **{order.get('order_id', '-')}** 조회 결과입니다.\n\n"
        f"📦 주문일: {order.get('order_date', '-')}\n"
        f"{_format_items(order.get('items') or [])}\n"
        f"- 총 결제금액: {total_text}\n"
        f"- 배송상태: **{status}** {emoji}"
    )


def _format_order_not_found(order_id: str) -> str:
    return (
        f"주문번호 '{order_id}'에 해당하는 주문 내역을 찾을 수 없습니다.\n"
        f"주문번호를 다시 확인해 주세요. (예: 1, 2, 3)"
    )


async def order_node(state: ShoppingState) -> dict:
    """실DB(ORDERS/ORDER_ITEM/PRODUCT) 주문 조회.

    [변경] 기존엔 mock/order_mock 의 고정 데이터를 읽었으나, 이제 실제 DB 를 조회한다.
           반환 dict 구조는 Mock 과 동일(어댑터)이라 _format_order_* 는 무수정 재사용.
           게스트(member_id=None) 차단은 edges 에서 이미 분기되므로 여기 도달 시점엔
           member_id 가 보장된다. 다만 향후 라우팅 변경으로 member_id 가 비는 경로가
           생기더라도 '엉뚱한 회원(예: 1번)의 주문을 노출'하는 사고를 원천 차단하기 위해,
           member_id 가 없으면 임의 폴백 없이 로그인 안내로 즉시 반환한다.
    """
    member_id = state.get("member_id")
    if not member_id:
        return {"raw_answer": _GUEST_ORDER_BLOCKED_MESSAGE}

    order_id = coerce_intent_result(state["intent_result"]).entities.order_id
    order_id = order_id.strip() if isinstance(order_id, str) else None

    try:
        if order_id:
            order = await asyncio.to_thread(fetch_order_by_id, member_id, order_id)
            body = _format_order_detail(order) if order else _format_order_not_found(order_id)
        else:
            orders = await asyncio.to_thread(fetch_orders, member_id)
            body = _format_order_list(orders)
    except Exception:
        # 조회/포맷팅 중 예기치 못한 예외(커넥션 오류 등)는 500 대신 안내 문구로
        # 폴백한다. 예외는 로그로 남겨 원인 추적은 유지한다.
        logger.exception("주문 조회/포맷팅 실패: member_id=%s order_id=%s", member_id, order_id)
        body = _ORDER_FORMAT_ERROR_MSG
    return {"raw_answer": body}


# ════════════════════════════════════════════════════════════════════════
# 6) complaint_node — COMPLAINT (gpt-5.4, 감정 공감)
# ════════════════════════════════════════════════════════════════════════
_COMPLAINT_SYSTEM_PROMPT = (
    "당신은 온라인 쇼핑몰의 고객 응대 상담사입니다. "
    "고객이 불만이나 항의를 표현하고 있습니다. "
    "고객의 감정에 깊이 공감하며 진심으로 사과하는 톤으로 답변하세요. "
    "단, 당신은 실제 주문/배송/환불 데이터에 접근할 수 없으므로 "
    "'환불 처리했습니다', '내일 도착합니다'처럼 구체적인 사실을 단정적으로 말하지 마세요. "
    "공감 표현 후, 필요하다면 '주문조회' 기능이나 고객센터(1234-5678, 평일 09:00~18:00) "
    "이용을 안내하세요. 답변은 2~3문장으로 간결하게 작성하세요."
)
_COMPLAINT_FALLBACK = (
    "불편을 드려 죄송합니다. 자세한 사항은 고객센터(1234-5678)로 문의해 주세요."
)


async def complaint_node(state: ShoppingState) -> dict:
    """gpt-5.4 감정 공감 응답 (LCEL 체인). 멀티턴 history 주입."""
    question = state["question"]
    history = state.get("history", [])
    emotion = coerce_intent_result(state["intent_result"]).emotion or "불편함"

    system = f"{date_system_prefix()}\n\n{_COMPLAINT_SYSTEM_PROMPT}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("history"),
        ("human", "[고객 감정: {emotion}]\n[고객 메시지]\n{question}"),
    ])
    chain = prompt | get_main_llm(temperature=0.7) | StrOutputParser()
    answer = await chain.ainvoke({
        "history": history_to_messages(history),
        "emotion": emotion,
        "question": question,
    })
    answer = (answer or "").strip() or _COMPLAINT_FALLBACK
    return {"raw_answer": answer}


# ════════════════════════════════════════════════════════════════════════
# 7) small_talk_node — SMALL_TALK (gpt-5.4, 가벼운 응답)
# ════════════════════════════════════════════════════════════════════════
# [수정] 스코프 제한: 기존 프롬프트는 쇼핑몰과 무관한 질문(주식 시황 등)에도 실제 내용으로
# 답변해 챗봇의 역할 경계가 무너지는 문제가 있었다. 인사·가벼운 잡담에는 친근하게 응대하되,
# 무관한 주제는 내용 답변 없이 "쇼핑 특화 AI" 라는 점을 상황에 맞게 정중히 안내하도록 변경.
_SMALL_TALK_SYSTEM_PROMPT = (
    "당신은 온라인 쇼핑몰의 친근한 AI 챗봇입니다. "
    "고객의 인사나 가벼운 잡담(안부, 감사 인사 등)에는 밝고 친근하게 응답하세요. "
    "단, 당신은 쇼핑에 특화된 AI입니다. 주식 시황, 뉴스, 정치, 수학 문제 풀이, 코드 작성, "
    "일반 상식 등 쇼핑몰과 무관한 주제의 질문에는 그 내용에 대해 답하지 말고, "
    "쇼핑 관련 질문에만 답변할 수 있다는 점을 상황에 맞는 자연스러운 표현으로 정중히 안내하세요. "
    "이때 상품 검색·추천, 주문 조회, 배송 문의 등 도와드릴 수 있는 일을 함께 제안하면 좋습니다. "
    "답변은 1~2문장으로 짧게 작성하세요."
)
_SMALL_TALK_FALLBACK = "안녕하세요! 무엇을 도와드릴까요?"


async def small_talk_node(state: ShoppingState) -> dict:
    """gpt-5.4 가벼운 응답 (LCEL 체인). 멀티턴 history 주입."""
    question = state["question"]
    history = state.get("history", [])

    system = f"{date_system_prefix()}\n\n{_SMALL_TALK_SYSTEM_PROMPT}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    chain = prompt | select_llm(TaskComplexity.SIMPLE, temperature=0.8) | StrOutputParser()
    answer = await chain.ainvoke({
        "history": history_to_messages(history),
        "question": question,
    })
    answer = (answer or "").strip() or _SMALL_TALK_FALLBACK
    return {"raw_answer": answer}


# ════════════════════════════════════════════════════════════════════════
# 7-1) guest_block_node — 게스트 주문조회 차단 (GPT 미사용)
# ════════════════════════════════════════════════════════════════════════
_GUEST_ORDER_BLOCKED_MESSAGE = (
    "주문 조회는 로그인 후 이용하실 수 있어요. "
    "로그인하시면 주문 내역을 확인해 드릴게요. 😊"
)


async def guest_block_node(state: ShoppingState) -> dict:
    """게스트가 ORDER_INQUIRY 를 요청한 경우의 안내 응답.

    기존 chat.py 의 _GUEST_ORDER_BLOCKED_MESSAGE 분기를 그래프 노드로 이동.
    GPT 미사용 → guard 는 pass-through.
    """
    return {"raw_answer": _GUEST_ORDER_BLOCKED_MESSAGE}


# ════════════════════════════════════════════════════════════════════════
# 8) guard_node — 환각 가드 (기존 hallucination_guard 재사용)
# ════════════════════════════════════════════════════════════════════════
async def guard_node(state: ShoppingState) -> dict:
    """인텐트별 환각 방어.

    [변경] 기존 guard 는 hits 를 ContextVar 로 읽었으나,
           이제 state["rag_hits"] 를 graph.guard 헬퍼에 직접 넘긴다.
           (graph/guard.py 가 state 기반 검증 로직을 담당)
    """
    from graph.guard import guard_answer_state

    final = await guard_answer_state(
        question=state["question"],
        answer=state["raw_answer"],
        intent_result=coerce_intent_result(state["intent_result"]),
        rag_hits=state.get("rag_hits", []),
        history=state.get("history", []),
    )
    return {"final_answer": final}


# ════════════════════════════════════════════════════════════════════════
# 9) append_message_node — checkpointer 메시지 누적
# ════════════════════════════════════════════════════════════════════════
async def append_message_node(state: ShoppingState) -> dict:
    """이번 턴의 질문/답변을 state["messages"] 에 누적한다.

    [목적] (설계 문제 1 해결)
      checkpointer 가 보존하는 messages 에 'HumanMessage(질문) + AIMessage(최종답변)'를
      함께 넣어야, 다음 턴에서 질문-답변 쌍이 온전한 맥락으로 복원된다.
      답변을 빼먹으면 다음 턴 컨텍스트가 '질문만 있는 반쪽'이 되어 버린다.

    [동작]
      add_messages reducer 덕분에 여기서 반환한 2개 메시지는 기존 목록 뒤에
      append 된다. (덮어쓰기가 아니라 누적)

    [범위]
      게스트는 config(thread_id) 없이 invoke 되므로 이 누적분이 영속되지 않고
      해당 요청 처리 후 폐기된다. 즉 게스트에게는 사실상 no-op 에 가깝다.
      (그래도 그래프 흐름 일관성을 위해 모든 경로가 이 노드를 통과한다.)
    """
    return {
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=state["final_answer"]),
        ]
    }
