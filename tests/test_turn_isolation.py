"""
tests/test_turn_isolation.py
로그인 멀티턴에서 '턴 간 상태 격리'를 검증하는 회귀 테스트.

[배경 — 재발 방지]
checkpointer 는 State 채널 값을 thread 단위로 영속하므로, 라우터가 매 턴
리셋/기준선을 잡지 않으면 두 가지 오염이 발생한다.
  1) rag_hits 스테일: SEMANTIC 턴의 hits 가 다음 턴(FAQ 등)에 그대로 복원되어
     chat.py 가 무관한 응답에 sources 를 부착.
  2) 메트릭 누적 이중 집계: result["messages"] 에 과거 턴 이력이 함께 복원되어
     토큰/도구호출/비용이 턴 수에 비례해 부풀려짐(PHASE 3 비교 데이터 오염).
"""
import asyncio
import os

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from schemas.intent_schema import IntentResult, IntentType, coerce_intent_result
from graph import builder, nodes
from graph.agent_builder import build_agent
from graph.metrics import (
    collect_message_metrics,
    collect_token_breakdown,
    snapshot_prior_message_ids,
    filter_new_messages,
)


# ════════════════════════════════════════════════════════════════════════
# 1) rag_hits 턴 간 격리 (라우터 그래프)
# ════════════════════════════════════════════════════════════════════════
def test_rag_hits_reset_between_turns(monkeypatch):
    """SEMANTIC 턴의 rag_hits 가 다음 FAQ 턴으로 새지 않는다.

    chat.py 와 동일하게 init_state 에 rag_hits=[] 를 넣어 invoke 하면,
    checkpointer 에 남아 있던 직전 턴 hits 가 이번 턴 결과에 복원되지 않아야 한다.
    (sources 오부착 방지의 1차 방어. 2차 방어는 chat.py 의 인텐트 조건)
    """
    def _ir(intent):
        return IntentResult(intent=intent, confidence=1.0).model_dump(mode="json")

    async def classify_semantic(state):
        return {"intent_result": _ir(IntentType.SEMANTIC_SEARCH)}

    async def classify_faq(state):
        return {"intent_result": _ir(IntentType.FAQ)}

    async def fake_semantic(state):
        return {
            "rag_hits": [{"id": "1", "metadata": {"product_name": "패딩", "price": 1000},
                          "distance": 0.1}],
            "raw_answer": "패딩 추천!",
        }

    async def fake_faq(state):
        return {"raw_answer": "배송은 2~3일 걸립니다."}

    async def fake_guard(state):
        return {"final_answer": state["raw_answer"]}

    monkeypatch.setattr(nodes, "semantic_node", fake_semantic, raising=True)
    monkeypatch.setattr(nodes, "faq_node", fake_faq, raising=True)
    monkeypatch.setattr(nodes, "guard_node", fake_guard, raising=True)

    cfg = {"configurable": {"thread_id": "turn-isolation-token"}}

    def _base_state(question):
        # chat.py process_chat_pipeline 의 로그인 init_state 와 동일 구조
        return {"question": question, "member_id": 1, "is_guest": False,
                "history": [], "rag_hits": []}

    async def run():
        # 턴 1: SEMANTIC → rag_hits 채워짐
        monkeypatch.setattr(nodes, "classify_node", classify_semantic, raising=True)
        builder._compiled_app = None
        app = builder.build_graph()
        r1 = await app.ainvoke(_base_state("겨울옷 추천"), config=cfg)
        assert len(r1.get("rag_hits", [])) == 1

        # 턴 2: FAQ → 직전 턴 hits 가 복원되면 안 된다
        monkeypatch.setattr(nodes, "classify_node", classify_faq, raising=True)
        builder._compiled_app = None
        app2 = builder.build_graph()
        r2 = await app2.ainvoke(_base_state("배송 얼마나 걸려요"), config=cfg)

        ir2 = coerce_intent_result(r2["intent_result"])
        assert ir2.intent == IntentType.FAQ
        assert r2.get("rag_hits", []) == [], "스테일 rag_hits 가 턴을 넘어 복원됨"

    asyncio.run(run())
    builder._compiled_app = None  # 다른 테스트 오염 방지


# ════════════════════════════════════════════════════════════════════════
# 2) intent_result 체크포인트 직렬화 불변식
# ════════════════════════════════════════════════════════════════════════
def test_intent_result_stored_as_dict(monkeypatch):
    """classify_node 는 State 에 dict(primitive)만 기록한다.

    Pydantic 인스턴스가 체크포인트에 저장되면 Redis(msgpack) 영속화에서
    미등록 타입 경고/차단 대상이 되므로, 채널 값 타입 자체를 검증한다.
    """
    from langchain_core.runnables import RunnableLambda

    class _FakeLLM:
        def with_structured_output(self, _schema):
            # classify_node 의 LCEL 체인(prompt | structured_llm)에 그대로 꽂히도록
            # Runnable 로 반환한다.
            return RunnableLambda(
                lambda _msgs: IntentResult(intent=IntentType.FAQ, confidence=0.9)
            )

    monkeypatch.setattr(nodes, "get_intent_llm", lambda temperature=0.0: _FakeLLM(),
                        raising=True)

    out = asyncio.run(nodes.classify_node({"question": "배송 문의"}))
    stored = out["intent_result"]
    assert isinstance(stored, dict)
    assert stored["intent"] == "FAQ"          # enum 이 아닌 순수 문자열(mode="json")
    assert coerce_intent_result(stored).intent == IntentType.FAQ


# ════════════════════════════════════════════════════════════════════════
# 3) 메트릭 턴별 집계 (누적 이중 집계 방지)
# ════════════════════════════════════════════════════════════════════════
class _FakeAnswerModel(FakeMessagesListChatModel):
    """도구 호출 없이 즉시 최종 답변을 내는 페이크 모델(토큰 메타 포함)."""
    def bind_tools(self, tools, **kwargs):
        return self


def test_metrics_count_only_new_messages_per_turn():
    """로그인 멀티턴에서 토큰/집계가 '이번 턴 신규분'으로 한정된다.

    스냅샷(snapshot_prior_message_ids) → 필터(filter_new_messages) 흐름은
    agent_chat / multi_agent_chat / mcp_agent_chat 라우터와 동일하다.
    기준선 없이는 턴2 집계가 240(누적)으로 부풀려지는 회귀를 방지한다.
    """
    model = _FakeAnswerModel(responses=[
        AIMessage(content="답변1",
                  usage_metadata={"input_tokens": 100, "output_tokens": 20,
                                  "total_tokens": 120}),
        AIMessage(content="답변2",
                  usage_metadata={"input_tokens": 100, "output_tokens": 20,
                                  "total_tokens": 120}),
    ])
    app = build_agent(model=model, checkpointer=MemorySaver(), force_rebuild=True)
    cfg = {"configurable": {"thread_id": "metrics-isolation-token"},
           "recursion_limit": 12}

    async def run():
        for turn in (1, 2):
            prior_ids = await snapshot_prior_message_ids(app, cfg)
            result = await app.ainvoke(
                {"messages": [HumanMessage(f"질문{turn}")],
                 "member_id": 1, "is_guest": False, "rag_hits": []},
                config=cfg,
            )
            new_messages = filter_new_messages(result["messages"], prior_ids)
            _, total_tokens, _ = collect_message_metrics(new_messages)
            prompt, completion = collect_token_breakdown(new_messages)
            assert total_tokens == 120, f"턴{turn} 누적 이중 집계: {total_tokens}"
            assert (prompt, completion) == (100, 20)

    asyncio.run(run())


def test_filter_new_messages_guest_passthrough():
    """게스트(새 thread → prior_ids 빈 집합)는 전체 메시지가 그대로 집계된다."""
    msgs = [HumanMessage("q"), AIMessage("a")]
    assert filter_new_messages(msgs, set()) is msgs
