"""
services/rag_service.py
RAG 응답 생성 전담 모듈.
- 날짜/계절 컨텍스트 자동 주입
- 멀티턴 대화 이력 지원
- 응답 생성은 멀티모델 팩토리(get_main_llm)를 경유한다 → .env LLM_PROVIDER 에 따라
  gpt-5.4/gemini/claude/deepseek 로 전환된다(provider 비교 공정성). 기존 OpenAI 직접
  호출(RAG_MODEL 상수 고정)을 걷어내고 complaint/small_talk 노드와 동일한 LCEL 패턴으로 통일.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from graph.llm import get_main_llm, date_system_prefix, history_to_messages

RAG_TEMPERATURE = 0.3

# 질문/컨텍스트는 f-string 으로 미리 조립하지 않고 '자리표시자'로 넘긴다.
# 이유: ChatPromptTemplate 은 문자열 안의 '{...}' 를 전부 템플릿 변수로 해석한다.
#       상품 설명에 중괄호(예: '사이즈 {S/M/L}')가 섞이면 KeyError 로 죽으므로,
#       question/context 를 완성 문자열이 아니라 ainvoke 인자로 안전하게 채운다.
#       (graph/guard.py 재시도 경로도 이 템플릿을 그대로 재사용 — 단일 출처)
RAG_HUMAN_TEMPLATE = (
    "[고객 질문]\n{question}\n\n"
    "[상품 컨텍스트]\n{context}\n\n"
    "위 상품 컨텍스트만 근거로 고객 질문에 답변하세요."
)


def build_product_context(hits: list[dict]) -> str:
    if not hits:
        return "(검색된 상품 없음)"
    lines = []
    for idx, hit in enumerate(hits, start=1):
        meta = hit.get("metadata", {}) or {}
        name = meta.get("product_name") or "(이름 없음)"
        category = meta.get("category") or "(카테고리 없음)"
        price = meta.get("price")
        price_str = f"{int(price):,}원" if isinstance(price, (int, float)) else "(가격 정보 없음)"
        description = hit.get("document") or meta.get("description") or "(설명 없음)"
        lines.append(
            f"[상품 {idx}]\n"
            f"- 상품명: {name}\n"
            f"- 카테고리: {category}\n"
            f"- 가격: {price_str}\n"
            f"- 설명: {description}"
        )
    return "\n\n".join(lines)


SYSTEM_PROMPT = (
    "당신은 온라인 쇼핑몰의 친절한 상품 추천 도우미입니다. "
    "반드시 아래 '상품 컨텍스트'에 제공된 상품 정보만을 근거로 답변하세요. "
    "컨텍스트에 없는 상품명·가격·사양을 절대 지어내지 마세요. "
    "고객의 질문과 잘 맞는 상품이 있으면 상품명·가격·특징을 자연스럽게 엮어 한국어로 추천하세요. "
    "코디 조합이나 스타일링 제안은 컨텍스트에 실제로 언급된 내용에 근거해서만 하고, "
    "컨텍스트에 없는 조합·활용법을 임의로 지어내지 마세요. "
    "만약 제공된 상품들이 고객의 질문과 전혀 관련이 없다면, 무리하게 추천하지 말고 "
    "관련 상품을 찾지 못했다고 정중히 안내한 뒤 다른 검색어를 제안하세요. "
    "이전 대화 이력이 있으면 맥락을 이어서 자연스럽게 답변하세요. "
    "이미 대화가 진행 중이면(이전 대화 이력이 있으면) 다시 인사말로 시작하지 말고 "
    "바로 본론부터 답하세요. "
    "답변은 간결하고 매력적으로 작성하고, 마크다운 표는 사용하지 마세요."
)


async def generate_rag_response(
    question: str,
    hits: list[dict],
    history: list[dict] | None = None,
) -> str:
    """질문 + 검색 상품 컨텍스트 + 대화 이력 → RAG 응답.

    응답 생성은 get_main_llm()(멀티모델 팩토리)을 경유하므로 .env LLM_PROVIDER 에
    따라 gpt-5.4/gemini/claude/deepseek 로 전환된다. temperature 는 RAG_TEMPERATURE.

    [history 처리] history 를 인자로 직접 받는다.
      - history 가 None 이면 기존 ContextVar(get_chat_history) 폴백
        → 구버전 파이프라인(pipeline/semantic_handler.py)/오프라인 테스트 호환성 유지.
      - LangGraph 노드는 state["history"] 를 명시적으로 넘긴다.

    [중괄호 안전] question/context 는 f-string 으로 미리 조립하지 않고 ainvoke 인자로
    넘긴다(RAG_HUMAN_TEMPLATE 주석 참고). 상품 설명에 '{...}' 가 있어도 안전하다.
    """
    from pipeline.pipeline_context import get_chat_history

    context = build_product_context(hits)
    full_system = f"{date_system_prefix()}\n\n{SYSTEM_PROMPT}"

    if history is None:
        history = get_chat_history()

    prompt = ChatPromptTemplate.from_messages([
        ("system", full_system),
        MessagesPlaceholder("history"),
        ("human", RAG_HUMAN_TEMPLATE),
    ])
    chain = prompt | get_main_llm(temperature=RAG_TEMPERATURE) | StrOutputParser()
    answer = await chain.ainvoke({
        "history": history_to_messages(history),
        "question": question,
        "context": context,
    })
    # 기존 동작 유지: 빈 응답이어도 폴백 문구를 새로 만들지 않고 그대로 반환한다
    # (환각 가드가 빈/무관 답변을 재시도 경로로 잡아내는 안전망이 이미 있음).
    return (answer or "").strip()