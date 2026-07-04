"""
graph/builder.py
LangGraph StateGraph 조립 + compile.

[그래프 구조]
    START
      ↓
    classify
      ↓ (route_by_intent: conditional)
      ├─ structured ──┐
      ├─ semantic   ──┤
      ├─ faq        ──┤
      ├─ order      ──┤
      ├─ complaint  ──┤
      ├─ small_talk ──┤
      └─ guest_block ─┤
                      ↓
                   guard
                      ↓
              append_message
                      ↓
                    END

[설계 메모]
- 핸들러 노드는 모두 raw_answer 를 채우고 guard 로 합류한다.
- guard 노드가 final_answer 를 만든다. (프리픽스 제거는 더 이상 불필요 →
  핸들러가 애초에 프리픽스를 안 붙이므로 strip 단계 자체가 사라졌다)
- guard 다음의 append_message 는 이번 턴 질문/답변을 messages 에 누적하는 노드다
  (checkpointer 가 이 messages 를 보존해 멀티턴을 구현한다).
- CHAT_HISTORY 저장은 그래프 밖(chat.py)에서 처리한다.
  · 저장은 '로그인 사용자만' + Oracle I/O 라 그래프 책임에서 분리하는 편이
    재사용/테스트(그래프 단독 실행)에 유리하다.
- compiled app 은 최초 호출 시 1회 생성해 재사용한다(build_graph 싱글톤).
  · '모듈 로드(import) 시'가 아니라 'build_graph() 최초 호출 시'다 — 정상
    기동 경로에서는 main.py 의 lifespan(비동기)이 set_checkpointer() 직후
    build_graph() 를 호출해 미리 컴파일해둔다(첫 요청 지연 제거 목적).
    lifespan 없이(예: 일부 테스트) build_graph() 가 먼저 불리면 get_checkpointer()
    의 MemorySaver 안전망으로 컴파일된다.

[Redis checkpointer 전환]
- 체크포인터를 모듈 로드 시점에 '고정 생성'하던 방식을 폐기하고,
  앱 기동(main.py lifespan)에서 set_checkpointer() 로 '주입'하는 방식으로 바꿨다.
  · REDIS_URL 이 있으면 main.py 가 AsyncRedisSaver 를 만들어 주입 → 영속화.
  · 없으면 MemorySaver 폴백 → 기존 로컬/테스트 동작 그대로(회귀 없음).
- 이렇게 한 이유:
  AsyncRedisSaver 는 연결/인덱스 셋업(asetup())에 await 가 필요한데,
  모듈 로드 시점(동기)에서는 이를 안전하게 수행할 수 없다. 따라서 '언제
  체크포인터가 준비되는가'를 lifespan(비동기)으로 미루고, builder 는 주입받은
  것을 그대로 compile 에 쓰기만 한다.
- 노드/엣지/chat.py 의 invoke·aget_state 호출부는 일절 바뀌지 않는다
  (MemorySaver 와 AsyncRedisSaver 모두 BaseCheckpointSaver 동일 계약).
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import ShoppingState
from graph.edges import route_by_intent
from graph import nodes

_compiled_app = None  # 컴파일된 그래프 싱글톤

# checkpointer 는 더 이상 모듈 로드 시점에 고정 생성하지 않는다.
# main.py lifespan 에서 set_checkpointer() 로 주입한다.
#   - Redis 활성: AsyncRedisSaver 주입 → thread 별 messages 가 Redis 에 영속.
#   - Redis 미설정/폴백: MemorySaver 주입 → 기존 인메모리 동작(재시작 시 소실).
# set_checkpointer() 없이 build_graph() 가 호출되면(=주입 누락) None 가드가
# 안전망으로 MemorySaver 를 깐다(테스트가 lifespan 없이 build_graph 만 부를 때 대비).
_checkpointer = None


def set_checkpointer(checkpointer) -> None:
    """앱 기동 시 사용할 체크포인터를 주입한다(lifespan 에서 호출).

    반드시 build_graph() 보다 먼저 호출해야 한다. 그래프가 이미 컴파일된 뒤
    호출하면 이미 만들어진 _compiled_app 의 체크포인터는 바뀌지 않는다
    (compile 시점에 바인딩되므로). 따라서 lifespan 순서는
    set_checkpointer() → build_graph() 를 지킨다.
    """
    global _checkpointer
    _checkpointer = checkpointer


def get_checkpointer():
    """chat.py 가 aget_state(config) 로 직접 조회할 때 사용하는 체크포인터 접근자.

    주입 전(None)이라면 안전망으로 MemorySaver 를 깐다. 정상 기동 경로에서는
    lifespan 이 항상 먼저 주입하므로 이 폴백은 거의 타지 않는다(테스트 보호용).
    """
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


def build_graph():
    """StateGraph 를 조립해 컴파일된 실행 객체를 반환(싱글톤).

    compile 에 쓰는 체크포인터는 get_checkpointer() 가 돌려준다.
    lifespan 이 set_checkpointer() 로 미리 주입했다면 그 인스턴스(Redis/Memory)를,
    누락됐다면 안전망 MemorySaver 를 사용한다.
    """
    global _compiled_app
    if _compiled_app is not None:
        return _compiled_app

    builder = StateGraph(ShoppingState)

    # ── 노드 등록 ─────────────────────────────────────────────────────
    builder.add_node("classify", nodes.classify_node)
    builder.add_node("structured", nodes.structured_node)
    builder.add_node("semantic", nodes.semantic_node)
    builder.add_node("faq", nodes.faq_node)
    builder.add_node("order", nodes.order_node)
    builder.add_node("complaint", nodes.complaint_node)
    builder.add_node("small_talk", nodes.small_talk_node)
    builder.add_node("guest_block", nodes.guest_block_node)
    builder.add_node("guard", nodes.guard_node)
    builder.add_node("append_message", nodes.append_message_node)

    # ── 엣지 연결 ─────────────────────────────────────────────────────
    builder.add_edge(START, "classify")

    # classify → (조건부) 핸들러 노드
    builder.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "structured": "structured",
            "semantic": "semantic",
            "faq": "faq",
            "order": "order",
            "complaint": "complaint",
            "small_talk": "small_talk",
            "guest_block": "guest_block",
        },
    )

    # 모든 핸들러 노드 → guard 합류
    for handler in (
        "structured", "semantic", "faq", "order",
        "complaint", "small_talk", "guest_block",
    ):
        builder.add_edge(handler, "guard")

    # guard → append_message → END
    # guard 직후 append_message 노드에서 이번 턴 질문/답변을
    #             messages 에 누적한 뒤 종료. (checkpointer 가 이 messages 를 보존)
    builder.add_edge("guard", "append_message")
    builder.add_edge("append_message", END)

    # checkpointer 를 달아 compile.
    # 로그인 사용자는 config={"configurable":{"thread_id": chat_token}} 로 invoke 시
    # thread 별 messages 가 자동 보존/복원된다. 게스트는 1회성 UUID thread_id 라 비영속.
    # 체크포인터 실체(Memory/Redis)는 get_checkpointer() 가 결정한다.
    _compiled_app = builder.compile(checkpointer=get_checkpointer())
    return _compiled_app
