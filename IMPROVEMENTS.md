# 개선 적용 내역 (FastAPI 측 / Spring Boot 영향 없음)

> 모든 변경은 FastAPI 내부 로직이며, Spring Boot ↔ FastAPI 의 3개 접점
> (CHAT_TOKEN 브릿지 / `/admin/products/*` 동기화 / 공유 Oracle DB)을
> 일절 건드리지 않는다. 기존 응답 스키마·엔드포인트도 그대로다(회귀 0).

---

## 1. [버그] 중복 ChromaDB 검색 제거
**파일:** `graph/rag_pipeline.py`
`search_and_rerank()` 안에서 `chroma_service.search_similar(...)` 가
동일 인자로 2번 연속 호출되고 있었다(두 번째가 첫 번째 결과를 덮어씀).
→ 1줄 제거. SEMANTIC 검색마다 ChromaDB 왕복이 절반으로 줄어든다.

---

## 2. [성능] Oracle 커넥션 풀
**파일:** `database/oracle_db.py`, `main.py`, `.env`
매 DB 호출마다 새 커넥션을 만들던 방식을 `create_pool` 재사용으로 전환.

- `init_pool()` / `close_pool()` 를 `main.py` lifespan 에 연결.
- **안전장치:** 풀 생성 실패 시 `_pool=None` 으로 두고 `get_connection()` 이
  기존 직접연결로 자동 폴백 → Wallet/네트워크 이슈 환경에서도 회귀 없음.
- `get_connection()` 반환은 풀/직접연결 모두 `with` 호환 →
  기존 모든 호출부(`with get_connection() as conn:`) 무수정.
- 풀에서 얻은 커넥션은 `with` 종료 시 close()=풀 반납(실제 단절 아님).
- `.env` 추가: `ORACLE_POOL_MIN=2`, `ORACLE_POOL_MAX=10`, `ORACLE_POOL_INCREMENT=1`
  (ARM 2 OCPU 기준 보수적 기본값. Autonomous DB 세션 한도 내).

---

## 3. [안정성] LLM 호출 타임아웃
**파일:** `graph/model_factory.py`, `services/gpt_service.py`,
`services/rag_service.py`, `.env`
OpenAI API 가 응답하지 않을 때 요청이 무한 대기하던 문제 차단.

- LangChain 경로(노드/Agent): 4개 provider 생성자
  (`ChatOpenAI`/`ChatGoogleGenerativeAI`/`ChatAnthropic`/`ChatDeepSeek`)에
  `timeout=_get_timeout()` 주입.
- OpenAI SDK 직접 경로(레거시 pipeline/ 경로의 `gpt_service`): `AsyncOpenAI(timeout=...)` 주입.
  (RAG 응답/환각 재시도는 이후 `get_main_llm()`/`get_intent_llm()` 기반 LangChain 경로로
  옮겨져 위 4개 provider 생성자 타임아웃을 그대로 상속받는다 — 20번 항목 참고)
- `.env` 추가: `LLM_TIMEOUT=30` (초). 0/빈값/오류 시 무제한(=기존 동작).
- **주의:** Runnable 레벨 `.with_config(timeout=)` 가 아니라 '생성자 timeout'
  이어야 실제 네트워크 요청에 타임아웃이 걸린다.

---

## 4. [운영] 깊은 헬스체크
**파일:** `main.py`
기존 `/health`(프로세스 생존만) 외에 `/health/deep` 신설.

- Oracle(`SELECT 1 FROM DUAL`) / ChromaDB(`heartbeat`) / Redis(`ping`)를
  각각 독립 점검 → `components` 로 상태 표시.
- Redis 는 `REDIS_URL` 미설정(MemorySaver 폴백) 시 `skipped` 처리(에러 아님).
- 하나라도 실패하면 `status="degraded"`(HTTP 200 유지, 본문으로 판별).
- 배포 후 smoke test / 모니터링 프로브용.

---

## 5. [입력 검증] 공백 전용 질문 차단
**파일:** `schemas/chat_schema.py`
`min_length=1` 은 공백 1글자(`" "`)를 통과시켜 인텐트 분류 LLM 을
헛호출(토큰/비용 낭비)할 수 있었다.
→ `question` 에 `field_validator` 추가: `strip()` 후 비면 422,
   아니면 앞뒤 공백 제거된 값으로 정규화. (max_length=1000 등은 기존 유지)

---

## 6. [문서] Redis TTL 적용 지점 주석
**파일:** `graph/checkpointer.py`
게스트는 매 요청 1회성 `guest-{uuid4}` thread_id 라, Redis 영속화를
켜면 재조회 안 되는 thread 가 쌓일 수 있다. 단 `langgraph-checkpoint-redis`
0.5.0 의 TTL 인자 형식이 버전에 민감하므로, **실제 Redis 를 켜는 시점에
검증 후 활성화**하도록 적용 지점만 주석으로 명시(코드 동작 변경 없음).
현재 `REDIS_URL=` (빈 값) → MemorySaver 폴백이라 미해당.

---

## 변경 파일 요약
| 파일 | 변경 |
|------|------|
| `graph/rag_pipeline.py` | 중복 search_similar 1줄 제거 |
| `database/oracle_db.py` | 커넥션 풀 + 실패 시 직접연결 폴백 |
| `main.py` | 풀 init/close, `/health/deep` |
| `graph/model_factory.py` | 4개 provider 타임아웃 주입 |
| `services/gpt_service.py` | AsyncOpenAI 타임아웃 |
| `services/rag_service.py` | AsyncOpenAI 타임아웃 |
| `schemas/chat_schema.py` | 공백 전용 question 차단 + HITL 응답/요청 스키마 |
| `graph/checkpointer.py` | Redis TTL 적용 지점 주석 |
| `.env` | LLM_TIMEOUT, ORACLE_POOL_* 추가 |

---

## 7. [기능] Human-in-the-loop — 환불 신청 (interrupt/resume)
**파일:** `graph/tools.py`, `graph/agent_builder.py`, `routers/agent_chat.py`,
`routers/mcp_agent_chat.py`, `schemas/chat_schema.py`, `tests/test_human_in_the_loop.py`

환불처럼 되돌리기 어려운 작업을 Agent가 자율 실행하지 않고 사용자 확인을 받는다.

- `request_refund` 도구 신규: 내부에서 `langgraph.types.interrupt(payload)` 호출 →
  그래프 일시정지 → 결과에 `__interrupt__` 적재.
- `/chat/agent`: interrupt 감지 시 `interrupt_pending=True` + payload 응답.
- `/chat/agent/resume` 신규 엔드포인트: `Command(resume="approve"/"reject")`로
  같은 thread(chat_token) 재개. Checkpointer가 State를 보존하므로 추가 인프라 0.
- **경계 1 — 게스트 차단**: 1회성 thread_id라 재개 불가 → `InjectedState`로 도구
  진입 즉시 차단(interrupt 전).
- **경계 2 — MCP Agent 제외**: 재개는 단일 Agent 그래프에서만 가능하므로
  `build_agent(include_refund=False)`로 MCP 경로에서 환불 도구 제외.
  (단일 Agent 싱글톤 캐시는 force_rebuild=True라 오염되지 않음)
- 실제 ORDERS 변경은 하지 않음(DB 쓰기 0) → 스키마/Spring Boot 영향 없음.
- 신규 테스트 6개 전원 통과(interrupt 발생/approve/reject/게스트 차단/도구 등록/MCP 제외).

## 8. [기능] RAGAs — RAG 정량 평가 (오프라인 전용)
**파일:** `scripts/evaluate_rag.py`, `requirements.txt`

RAG 응답의 정확도(Faithfulness/Answer Relevancy/Context Precision)를 정량 측정한다.

- 운영 파이프라인의 `search_and_rerank()` + `generate_rag_response()`를 그대로
  import해 재사용 → 실제 서버와 동일 경로를 평가.
- 운영 코드 0줄 변경(엔드포인트/그래프/스키마 무수정) → 서버/Spring Boot 영향 없음.
- `--make-sample`로 평가셋 템플릿 생성, RAGAs 미설치 시 친절 안내 후 종료.
- `RERANK_ENABLED` 토글로 재랭킹 적용 전후 품질 비교 가능.
- 평가 전용 의존성(`ragas`, `datasets`)은 서버 런타임 불필요(주석으로 분리).

---

## (이하 기존 개선 항목)

### 변경 파일 요약(기능 추가 포함)
| 파일 | 변경 |
|------|------|
| `services/bm25_service.py` | BM25 하이브리드 검색(Sparse 후보, 신규) |
| `graph/rag_pipeline.py` | BM25 후보 병합(_merge_dedup 재사용) — *이후 21번에서 RRF 전용 함수로 교체됨* |
| `.github/workflows/rag_quality.yml` | RAGAs CI 품질 게이트(신규) |
| `scripts/ci_index_fixtures.py` | CI 격리 픽스처 색인(신규) |
| `scripts/ci_fixtures/` | CI 고정 상품/평가셋(신규) |
| `scripts/evaluate_rag.py` | --ci 모드 + 임계값 게이트 |
| `tests/test_bm25_hybrid.py` | BM25 테스트 7개(신규) |
| `DEPLOY_TROUBLESHOOTING.md` | 재배포 문제/해결/최적 절차(신규) |
| `graph/tools.py` | request_refund 도구(interrupt) 추가 |
| `graph/agent_builder.py` | include_refund 파라미터 |
| `routers/agent_chat.py` | interrupt 감지 + /chat/agent/resume |
| `routers/mcp_agent_chat.py` | MCP 경로 환불 도구 제외 |
| `scripts/evaluate_rag.py` | RAGAs 평가 스크립트(신규) |
| `tests/test_human_in_the_loop.py` | HITL 테스트 6개(신규) |
| `requirements.txt` | ragas/datasets 선택 의존성 주석 |

## Spring Boot 영향
없음. 위 3개 접점을 건드리지 않으며, 기존 엔드포인트/응답 스키마도 동일.
(참고: 팀원 코드 수령 시 `FASTAPI_BASE_URL` 만 실제 서버 IP 로 교체할 것 —
SPRINGBOOT_GUIDE.md 11번의 localhost 는 별도 인스턴스 구성에서 수정 필요.)

---

## 9. [버그] Agent 체크포인터 공유 깨짐
**파일:** `routers/agent_chat.py`, `routers/multi_agent_chat.py`

`main.py`가 라우터를 import하는 시점은 lifespan(체크포인터 주입)보다 **이전**이다.
그런데 두 라우터가 모듈 최상단에서 `build_agent()`/`build_multi_agent()`를 즉시 호출해,
체크포인터가 아직 없는 상태(None)에서 안전망 `MemorySaver`가 그래프에 박혀버렸다.
이후 lifespan이 정식 체크포인터(Redis/공유 Memory)를 주입해도 이미 컴파일된
단일·멀티 Agent에는 반영되지 않아, 라우터와 다른 체크포인터를 쓰는 상태가 됐다.

→ `chat.py`와 동일한 lazy getter 패턴(`_get_agent_app()`/`_get_multi_app()`)으로 전환.
   첫 요청 시점에 컴파일해 lifespan이 주입한 체크포인터를 공유받는다.
   (검증: 세 그래프가 동일 체크포인터 인스턴스를 공유 — 수정 전 False → 수정 후 True)

## 10. [버그] 주문번호 추출 형식 불일치
**파일:** `pipeline/intent_classifier.py`, `schemas/intent_schema.py`, `graph/tools.py`

인텐트 분류 프롬프트와 도구 설명이 주문번호를 `ORD-YYYYMMDD-NNNN` 형식으로만
추출하도록 지시했으나, 실제 `ORDERS.ORDER_ID`는 숫자(NUMBER) PK다. 사용자가 화면에
보이는 대로 "3번 주문 환불해줘"라고 하면 `order_id=null`로 처리돼 전체 목록만 나왔다.
→ 추출 규칙·예시를 실제 숫자 형식("3")으로 정정(라우터/단일·멀티 Agent 모두 영향).

## 11. [보안] order_node member_id 폴백 제거
**파일:** `graph/nodes.py`

`member_id = state.get("member_id") or 1`은 member_id가 비면 **회원 1번의 주문을
노출**할 수 있는 위험한 폴백이었다(falsy 함정). 게스트 차단이 edges에 있어 평상시엔
도달하지 않지만, 향후 라우팅 변경 시 사고로 이어질 수 있다.
→ member_id가 없으면 임의 폴백 없이 로그인 안내로 즉시 반환.

## 12. [안정성] 임베딩/음성 호출 타임아웃
**파일:** `services/embed_service.py`, `services/voice_service.py`

3번(LLM 타임아웃)에서 누락됐던 `AsyncOpenAI` 두 곳에 `LLM_TIMEOUT`을 마저 적용.
임베딩·STT/TTS 호출이 응답 없을 때 무한 대기하던 경로를 차단.

## 13. [멀티턴] 이력 트리밍 (컨텍스트 윈도우 초과 방지)
**파일:** `graph/llm.py`, `routers/chat.py`, `graph/agent_builder.py`,
`graph/multi_agent_builder.py`, `.env`, `tests/test_history_trimming.py`

Checkpointer에 누적된 messages가 길어지면 LLM 컨텍스트 윈도우를 초과할 수 있다.
LangChain `trim_messages`로 **원본은 보존하고 LLM 입력만** 최근 N개로 트리밍한다.

- `trim_message_list`/`trim_history`/`agent_pre_model_hook`을 `graph/llm.py`에 집약.
- 라우터: `_resolve_history` 반환 직전 트리밍(complaint/small_talk/semantic 자동 적용).
- Agent: `create_react_agent(pre_model_hook=...)`로 LLM 입력만 교체(원본 비파괴).
- 멀티 Agent: sub-agent는 동일 훅, supervisor/force_finish는 트림 직접 호출.
- **tool_call 쌍 보존**: `strategy="last"` + `start_on="human"`으로 사용자 메시지
  경계에서만 잘라 `AIMessage(tool_calls)`↔`ToolMessage` 분리를 방지.
- **안전장치**: 캡이 한 턴보다 작아 현재 질문이 누락될 경우 원본 폴백.
- `.env` 추가: `HISTORY_MAX_MESSAGES=20`. 신규 테스트 6개 통과.

## 14. [보안/품질] 마이너 개선 묶음
- **관리자 키 비교**(`routers/admin.py`): `!=` → `hmac.compare_digest`(타이밍 공격 방지).
- **환불 사전 검증**(`graph/tools.py`): `request_refund`가 interrupt 전에
  `fetch_order_by_id`로 주문 존재/본인 소유를 확인 → 없는 주문엔 확인창을 띄우지 않음.
  (HITL 테스트에 '주문 없음→interrupt 미발생' 케이스 추가, 6개→7개)
- **측정 비블로킹**(`graph/metrics.py`): `record_metrics`가 이벤트 루프 중이면
  파일 쓰기를 `asyncio.to_thread`로 오프로드(루프 블로킹 방지). 동기 컨텍스트는 그대로.
- **개인화 캐시 스탬피드 방지**(`services/personalization_service.py`): member_id별
  `asyncio.Lock` + 더블체크로, 동시 캐시 미스가 비싼 DB+임베딩을 중복 실행하지 않게 함.

## 15. [정리] 주석/구조 정돈 (회사 제출용)
- 코드 주석의 개발 회차 표현("대화N") 전부 제거(의미 보존).
- 낡은 모델명 표기 `gpt-4o`/`gpt-4o-mini` → 실제 사용 모델 `gpt-5.4`/`gpt-5.4-mini`로 통일
  (단, `voice_service.py`의 OpenAI 대체 모델명 `gpt-4o-transcribe` 등은 실제 제품명이라 보존).
- ChromaDB 배포 설명을 docker-compose → 실제 방식(`chroma run` CLI + systemd)으로 정정.
- 루트 점검 스크립트(`check_*.py`)를 `scripts/dev_checks/`로 이동 + docstring 추가.

## 16. [운영] 성능/비용 대시보드 (metrics.jsonl 시각화)
**파일:** `graph/metrics.py`, `routers/admin.py`, `tests/test_metrics_dashboard.py`

PHASE 3에서 `record_metrics`가 적재하던 `metrics.jsonl`을 실제로 보여주는 대시보드.
계획으로만 있던 "경로별 성능/비용 비교"를 화면으로 완성했다.

- `graph/metrics.py::summarize_metrics()`: metrics.jsonl 을 읽어 경로별
  (라우터/단일/멀티 Agent)·모델별로 요청 수/평균 레이턴시/input·output 토큰/누적 비용 집계.
  파일 없음·깨진 줄은 예외 없이 건너뛰고 available 플래그로 상태를 알린다(읽기 전용).
- `GET /admin/metrics/summary`: 집계 JSON. 기존 `verify_admin_key`(X-ADMIN-KEY) 재사용.
  파일 I/O 는 `run_in_threadpool` 로 넘겨 이벤트 루프를 막지 않는다.
- `GET /admin/metrics`: Chart.js(CDN) 기반 대시보드 HTML 한 장. 페이지는 인증 없이
  열리고, 키 입력 후 그 값으로 /summary 를 호출해 차트를 그린다(키는 브라우저에만 유지).
- **영향 범위 0**: 새 테이블·새 파이썬 의존성 없음, `main.py`·`.env`·Spring Boot·스키마
  전부 무변경(admin.router 는 이미 등록됨, METRICS_* 는 기존 설정 재사용).
- 신규 테스트 5개(집계 순서/깨진 줄 스킵/파일 없음/인증 403·200/HTML 렌더링).
- **Spring Boot 영향 없음**: metrics.jsonl 은 FastAPI 가 자체 적재하는 로그라
  Spring Boot 에 데이터 소스가 없다. 팀 가이드·공유 DB 스키마 변경도 불필요하다.

## 변경 파일 요약(9~16번)
| 파일 | 변경 |
|------|------|
| `routers/agent_chat.py`, `routers/multi_agent_chat.py` | lazy getter로 체크포인터 공유 복구 |
| `pipeline/intent_classifier.py`, `schemas/intent_schema.py`, `graph/tools.py` | 주문번호 숫자 형식 정정 |
| `graph/nodes.py` | order_node member_id 폴백 제거 |
| `services/embed_service.py`, `services/voice_service.py` | AsyncOpenAI 타임아웃 |
| `graph/llm.py` | 트리밍 헬퍼 3종 |
| `routers/chat.py` | history 트리밍 적용 |
| `graph/agent_builder.py`, `graph/multi_agent_builder.py` | pre_model_hook + supervisor 트림 |
| `routers/admin.py` | hmac.compare_digest + 성능/비용 대시보드 |
| `graph/metrics.py` | record_metrics 비블로킹 + summarize_metrics 집계 |
| `services/personalization_service.py` | 캐시 스탬피드 방지(asyncio.Lock) |
| `tests/test_history_trimming.py` | 트리밍 테스트 6개(신규) |
| `tests/test_metrics_dashboard.py` | 대시보드 테스트 5개(신규) |
| `tests/test_human_in_the_loop.py` | 환불 사전검증 케이스 추가 |
| `.env` | HISTORY_MAX_MESSAGES 추가 |
| `scripts/dev_checks/` | check_*.py 이동 + docstring |

## 17. [버그 모음] 코드 리뷰 중 발견된 실동작 버그 4건
**파일:** `graph/tools.py`, `database/oracle_db.py`, `services/bm25_service.py`,
`graph/llm.py`, `graph/nodes.py`

- **Agent `sort_by` 대소문자 불일치**: `search_products` 툴 docstring이 LLM에게
  `'price_asc'`(소문자) 형식을 쓰라고 안내했지만, 실제 매핑 테이블
  (`_STRUCTURED_SORT_MAP`)은 `'PRICE_ASC'`(대문자, router 경로의 `SortType` enum과
  동일)만 인식했다. Agent가 툴 설명 그대로 따르면 정렬 요청이 예외 없이 조용히
  기본 정렬로 무시됐다. → 툴 docstring 정정 + `_resolve_order_by()`에 대소문자
  정규화 + `NEWEST→LATEST` 별칭 추가(화이트리스트 방어는 유지, SQL 인젝션 테스트
  통과 확인).
- **BM25 인덱스 갱신 레이스 컨디션**: `upsert_one`/`delete_one`이 '스냅샷 계산(락)'과
  '재구축(락 밖에서 재호출)'을 분리해서 수행해, 동시 상품 CRUD 시 lost update가
  가능했다. → `_build_index_locked()`로 스냅샷+재구축을 한 락 구간에 원자적으로 묶음.
- **멀티턴 트리밍 안전장치 identity 비교 오류**: `trim_message_list()`가
  `last_human not in trimmed`(내용 기반 `==`)로 검사해, 사용자가 "네"처럼 이전
  턴과 동일한 문자열로 질문하면 실제로는 트림 결과에 이번 턴 메시지가 없어도
  안전장치가 오판 통과할 수 있었다. → `is` 기반 identity 비교로 정정.
- **`order_node` 포맷팅 KeyError 방어**: `_format_items` 등이 `dict['key']` 직접
  접근이라 DB 스키마 변경/조인 NULL 시 500으로 즉시 죽을 수 있었다. →
  `.get()` + 기본값 + `order_node`에 try/except 폴백 추가(예외는 로깅, 사용자에겐
  안내 문구).

전체 기존 테스트(123개) 회귀 없음 확인.

## 18. [정리] 문서/주석 2차 정리 (코드-주석 불일치 전수 점검)
**파일:** `graph/`, `services/`, `routers/`, `database/`, `pipeline/` 전반

- 이전(15번 항목) 정리에서 놓친 "대화N"류 개인 작업노트("[자가피드백 개선]" 등)와
  잔여 `gpt-4o`/`gpt-4o-mini` 표기(`graph/nodes.py`, `services/rag_service.py`,
  레거시 `pipeline/*.py` 등 6곳)를 추가로 발견해 정리.
- 코드 구조 변경 이후 갱신되지 않은 docstring 다수 정정: `graph/builder.py`
  그래프 다이어그램에 `append_message` 노드 누락, "모듈 로드 시 컴파일"이라는
  부정확한 설명(실제로는 lifespan 최초 호출 시), `database/oracle_db.py` 헤더가
  "Oracle 23ai FREE"로 오기(실제는 Wallet 기반 Autonomous DB), `graph/agent_builder.py`/
  `multi_agent_builder.py`가 provider 전환 가능한 `get_main_llm()`을 "gpt-5.4 고정"인
  것처럼 서술, `graph/rag_pipeline.py` 흐름도에 개인화(PHASE 2-②) 단계 누락,
  `graph/observability.py`의 `route_metadata()` 인자 설명이 실제 호출값과
  불일치("voice"는 실제로 안 쓰이고 "mcp_agent"가 빠져 있었음).
- 코드 주석의 "PHASE N-①" 식 구현순서 태그를 전부 제거(스크립트 일괄 처리 후
  잔여 빈 괄호·빈 콤마·헤더 뒤 빈 줄 소실 등 부작용을 수동으로 재확인·복구).
  로드맵 문서(`ROADMAP.md`, `PRESENTATION_SCRIPT.md` 등)의 PHASE 표기는 문서
  구조 자체이므로 유지.
- `Entities.period_days` 필드 제거(19번 항목 참고).

## 19. [스키마] 죽은 필드 `period_days` 제거
**파일:** `schemas/intent_schema.py`

`Entities.period_days`("최근 7일 조회" 등에서 추출하도록 설명된 필드)가 실제로는
`fetch_orders()`를 비롯해 어디서도 읽히지 않는 죽은 필드였다(기간 필터 자체가
구현돼 있지 않음). LLM이 채워봤자 조용히 버려지는 상태라 필드를 삭제.
(주문 기간 필터가 필요해지면 `fetch_orders(member_id, period_days=...)` 형태로
별도 기능 추가 필요 — 이번 스코프에는 미포함.)

## 20. [아키텍처] 라우터 파이프라인 provider 통일 (SEMANTIC 생성 + 환각 재시도)
**파일:** `services/rag_service.py`, `graph/guard.py`

라우터 파이프라인에서 인텐트 분류(`classify_node`)와 COMPLAINT/SMALL_TALK
(`complaint_node`/`small_talk_node`)는 이미 `get_main_llm()`/`get_intent_llm()`을
거쳐 provider 전환이 됐지만, SEMANTIC_SEARCH의 답변 생성(`rag_service.py`)과
환각 가드 재시도(`graph/guard.py`)는 `RAG_MODEL="gpt-5.4"` 상수로 OpenAI SDK를
직접 호출해 `.env`의 `LLM_PROVIDER` 설정과 무관하게 항상 OpenAI가 답했다.
DeepSeek 등으로 provider 벤치마크를 돌리면 "1차 답변은 DeepSeek인데 가드 재시도만
OpenAI가 몰래 보정"하는 비대칭이 생겨, 메트릭 대시보드(16번 항목)에 기록되는
provider 필드가 실제로 응답을 생성한 모델과 달라지는 문제가 있었다.

- `rag_service.generate_rag_response()`: `AsyncOpenAI` 직접 호출 → `get_main_llm()`
  기반 LangChain LCEL 체인(`complaint_node`/`small_talk_node`와 동일 패턴)으로 교체.
- `graph/guard.py`의 `_retry_semantic_answer()`: `gpt_service.chat_completion
  (model=GPT_MODEL_INTENT 고정)` → `get_intent_llm()` 기반 체인으로 교체(저비용
  역할 티어는 유지, provider만 전환되게 함).
- **안전장치**: 질문/상품 컨텍스트를 f-string으로 미리 조립해 템플릿에 넣지 않고
  `RAG_HUMAN_TEMPLATE` 자리표시자로 안전하게 전달(상품 설명에 `{S/M/L}` 같은
  중괄호가 섞여도 `ChatPromptTemplate`의 `KeyError`를 유발하지 않음 — 실제 재현
  테스트로 검증).
- `@traceable` 데코레이터 제거(LangChain 자동 트레이싱과 중복 방지, 다른 LCEL
  노드와 일관성).
- 함수 시그니처/모듈 상수명(`SYSTEM_PROMPT`, `RAG_TEMPERATURE`,
  `build_product_context`) 전부 유지 → 호출부(레거시 `pipeline/semantic_handler.py`,
  `scripts/evaluate_rag.py` 포함) 무수정.
- **의도적 스코프 제외**: "가드 재시도가 실패하면 가장 안정적인 OpenAI로 fallback"
  같은 정책은 이번엔 도입하지 않았다. 먼저 모든 provider가 동일 조건에서 측정되게
  통일한 뒤, 실측 데이터에서 특정 provider의 재시도 회복력이 유의미하게 낮다고
  확인되면 그때 근거를 남기며 명시적으로 채택하는 편이 벤치마크 신뢰성 측면에서
  더 낫다고 판단.
- 기존 테스트 전부(monkeypatch 기반 23개 포함) 회귀 없음 확인 + provider별
  모델 해석(`resolve_model_name`) 단위 검증 통과.

## 변경 파일 요약(이번 정리, 17~20번)
| 파일 | 변경 |
|------|------|
| `graph/tools.py`, `database/oracle_db.py` | Agent sort_by 대소문자/별칭 방어 |
| `services/bm25_service.py` | upsert/delete 레이스 컨디션 수정 |
| `graph/llm.py` | 트리밍 안전장치 identity 비교로 정정 |
| `graph/nodes.py` | order 포맷팅 방어적 접근 + try/except 폴백 |
| `schemas/intent_schema.py` | `period_days` 필드 제거 |
| `services/rag_service.py` | OpenAI 직접 호출 → `get_main_llm()` LCEL 체인 |
| `graph/guard.py` | 환각 재시도 → `get_intent_llm()` LCEL 체인 |
| `graph/`, `services/`, `routers/`, `database/`, `pipeline/` 전반 | 주석 정확성 정리, PHASE 태그 제거 |

---

## 21. [재검토 3차] 코드 리뷰 발견 5건 + BM25 하이브리드 융합 알고리즘을 RRF로 교체
**파일:** `database/oracle_db.py`, `graph/nodes.py`, `services/bm25_service.py`,
`graph/metrics.py`, `services/personalization_service.py`, `graph/rag_pipeline.py`,
`tests/test_bm25_hybrid.py`, `.env.example`

20번 정리 이후 재검토에서 5건을 추가로 발견해 수정했다(회귀 없음, 모든 변경은 관점
A/B/C 검토 후 반영). 가장 비중이 큰 것은 하이브리드 검색 융합 알고리즘 교체(RRF).

- **[버그] `ORDER_STATUS` 영문 enum ↔ 한글 라벨 불일치**: `ORDERS.ORDER_STATUS`는
  DB CHECK 제약상 `SHIPPED`/`DELIVERED` 등 영문만 저장되는데, 화면 포맷 함수와
  이모지 맵(`_ORDER_STATUS_EMOJI`)은 한글 키("배송중" 등)를 전제로 작성돼 있었다.
  실DB 연동 후 영문 코드가 그대로 노출되고 이모지가 전부 기본값(📦)으로 빠지는
  버그였다(기존 어댑터 단위 테스트는 픽스처에 한글을 직접 넣어 검증해 이 불일치를
  잡지 못함). → `oracle_db._to_status_label()` 신설(PENDING/PAID/PREPARING/
  SHIPPED/DELIVERED/CANCELLED/REFUNDED → 한글 7종), `_ORDER_STATUS_EMOJI`에
  대응 4키(💳/💰/❌/↩️) 추가. 매핑에 없는 값·기존 한글 입력은 원본 그대로 반환해
  레거시 데이터/테스트와 호환.
- **[버그] BM25 hit 가격이 Decimal이라 컨텍스트에서 소실**: `bm25_service._row_to_hit`가
  Oracle `Decimal` 가격을 그대로 실어, `rag_service.build_product_context`의
  `isinstance(price,(int,float))` 검사에 걸려 "(가격 정보 없음)"으로 표시되는 버그가
  `BM25_ENABLED=true` 경로에서만 있었다(ChromaDB hit은 색인 시 float 변환이라 정상).
  → `_to_price()` 헬퍼로 float 정규화, ChromaDB hit과 스키마 완전 동일화.
- **[안정성] `record_metrics`의 백그라운드 태스크 GC 유실 위험**: `loop.create_task(...)`
  반환값을 어디에도 보관하지 않아, 파이썬 공식 문서가 경고하는 대로 완료 전 GC될 수
  있었다(드물게 메트릭 1줄 유실). → 모듈 레벨 `_pending_writes: set` 신설, 태스크를
  강참조로 보관하고 `add_done_callback`으로 완료 시 자동 정리(누수 없음).
- **[하드닝] 개인화 캐시/락 무한 증가 + `clear_cache`의 in-flight 락 파괴**:
  `_cache`/`_locks`의 TTL은 '읽을 때만' 만료돼, 다시 오지 않는 회원 항목이 영구
  잔존해 장기 구동 시 메모리가 서서히 늘어났다. 또 `clear_cache(member_id)`가
  계산 중(in-flight)인 코루틴이 쥔 락까지 pop해버리면, 그 순간 새 락이 생성돼
  동시 캐시 미스 스탬피드 방어(14번 항목)가 좁은 창에서 뚫릴 수 있었다.
  → `PERSONALIZATION_CACHE_MAX`(기본 512) 상한 도입: 상한 도달 시 만료 항목 우선
  정리 → 그래도 초과면 저장시각이 오래된 순으로 축출(`_evict_if_needed`). 락 정리는
  `lock.locked()`가 `False`일 때만 pop하도록 정정(`_drop_member`, `clear_cache`).
- **[아키텍처] BM25+Dense 융합을 RRF(Reciprocal Rank Fusion)로 교체 — 융합 편향 제거**:
  기존 방식은 BM25 자체 정규화(`distance = 1 - score/max_score`)로 만든 hit을
  `_merge_dedup()`(distance 최소값 채택)에 그대로 태워 병합했다. 이 조합은 BM25
  리스트의 1등이 항상 `distance=0.0`이 되어, 벡터 검색 결과가 아무리 좋아도 키워드
  1등 상품이 무조건 최상위를 차지하는 **융합 편향(fusion bias)**이 있었다
  (`RERANK_ENABLED=false`일 때 하이브리드가 사실상 순수 BM25로 퇴화 — "두 검색의
  장점을 상호 보완한다"는 하이브리드 검색의 정의와 충돌).
  → `graph/rag_pipeline._merge_text_rrf()` 신규 도입. 절대 점수(distance/score)를
  융합 단계에서 완전히 배제하고, id별 `RRF(id) = Σ_L 1/(k + rank_L(id))`(순위만)로
  점수를 매기는 스케일-프리 알고리즘으로 교체(`RRF_K`, 기본 60 = 원논문 표준값).
  Dense/Sparse 양쪽에서 검증된 상품이 자연히 최상위로 올라간다.
  - **관점 C 보호(회귀 0, 함수 분리법)**: CLIP 멀티모달 병합용 `_merge_dedup()`는
    1바이트도 건드리지 않았다. `_merge_text_rrf()`는 텍스트 하이브리드(BM25+Dense)
    경로 전용 완전 독립 함수 — CLIP 경로 사이드 이펙트 가능성 자체를 차단.
  - RRF 결과도 하류 스키마는 100% 동일(`distance` 필드 유지 방식으로 인코딩,
    `rrf_score`는 로깅용 부가 키) → `rerank_service.attach_scores()`,
    `hits_to_sources()` 등 하류 코드 전부 무수정.
  - `BM25_ENABLED=false`(기본)면 이 함수 자체가 호출되지 않아 기존 동작 100% 불변.
  - 신규 테스트 5개 추가(`test_bm25_hybrid.py`, 7→12개): 양쪽 등장 상품 최상위 우선,
    **BM25 Top-1(distance=0.0) 편향이 더 이상 무조건 이기지 않음을 직접 증명**하는
    케이스 포함, 빈 sparse 폴백, 반환 스키마/정렬 유지, `RRF_K` 파싱 폴백.
- `.env.example`: `RRF_K=60`, `PERSONALIZATION_CACHE_MAX=512` 옵션 문서화(둘 다
  기본값 존재라 필수 아님).
- 전체 93개 Python 파일 `py_compile` 전수 통과, `pyflakes` 신규 미정의/미사용 없음 확인.

## 변경 파일 요약(21번)
| 파일 | 변경 |
|------|------|
| `database/oracle_db.py` | `_to_status_label()` — ORDER_STATUS 영문→한글 매핑 |
| `graph/nodes.py` | `_ORDER_STATUS_EMOJI` 4키 추가(결제대기/결제완료/주문취소/환불완료) |
| `services/bm25_service.py` | `_to_price()` — hit 가격 Decimal→float 정규화 |
| `graph/metrics.py` | `_pending_writes` — 백그라운드 기록 태스크 강참조+자동정리 |
| `services/personalization_service.py` | 캐시 상한(`CACHE_MAX`) + in-flight 락 보존 |
| `graph/rag_pipeline.py` | `_merge_text_rrf()`/`_rrf_k()` 신규 — RRF 융합(CLIP 경로 무영향) |
| `tests/test_bm25_hybrid.py` | RRF 테스트 5개 추가(7→12개) |
| `.env.example` | `RRF_K`, `PERSONALIZATION_CACHE_MAX` 추가 |

## 22. [재검토 4차] 코드 전수 재검토 발견 3건 + 문서-코드 불일치 1건

이전(21번) 정리 이후 코어 요청 경로(graph/, routers/, database/, services/ 전체,
schemas/)를 처음부터 다시 정독하며 4건을 추가로 발견해 수정했다(회귀 없음).

- **[버그] `personalization_service._locks` 딕셔너리 누수**: `get_preference_vector()`의
  두 예외 경로(`fetch_purchase_history` 실패, `get_embeddings_by_ids` 실패)는
  `_cache_put()`을 타지 않고 바로 `return None`했다. 그런데 `_evict_if_needed()`/
  `clear_cache()`의 락 정리는 전부 `_cache.items()`를 순회하며 동작하므로,
  애초에 `_cache`에 들어간 적 없는 member_id 의 `_locks` 항목은 상한(`CACHE_MAX`)
  관리 대상에서 완전히 빠져 영구 잔존했다(21번 항목에서 캐시 자체의 무한 증가는
  막았지만 이 두 예외 경로는 놓침). Oracle/ChromaDB 가 일시적으로 불안정한 구간에서
  다양한 member_id 가 이 경로를 반복해서 타면 `_locks`가 서서히 증가할 수 있었다.
  → 같은 함수의 다른 '결과 없음' 분기(히스토리 0건/가중치 0건/임베딩 0건)와 동일하게
  이 두 예외 경로도 `_cache_put(member_id, None)`을 호출하도록 통일. 실패도 TTL 동안
  캐싱되어 `_locks` 정리 대상에 포함되고, 부수적으로 장애 구간의 재조회 폭주도 줄어든다.

- **[버그] `routers/voice.py` — 긴 STT 결과가 처리되지 않은 500 을 유발**:
  `/chat/voice`는 STT 인식 결과를 그대로 `ChatRequest(question=question, ...)`에
  넣어 생성하는데, `ChatRequest.question`은 `max_length=1000`(문자 기준) 제약이 있다.
  25MB(수 분 분량) 음성은 STT 결과가 1000자를 가볍게 넘을 수 있고, 이 경우
  `ChatRequest(...)` 생성 시점에 pydantic `ValidationError`가 라우터 안 어떤
  `try/except`에도 걸리지 않고 그대로 터져, 다른 모든 실패 경로가 친절한 안내
  문구로 폴백하는 것과 달리 처리되지 않은 500 이 나갔다.
  → `save_chat_history`의 QUESTION 길이 절단과 동일한 패턴으로, `ChatRequest` 생성
  직전에 1000자 초과 시 절단(경고 로그만 남기고 앞부분 유지)하도록 방어 추가.

- **[개선] `graph/rag_pipeline.search_and_rerank()` — BM25 하이브리드 시 후보 풀이 얕음**:
  `n_candidates`는 기존에 `RERANK_ENABLED`에만 반응해, 재랭킹이 꺼진 상태(기본값)로
  `BM25_ENABLED=true`를 켜면 Dense/Sparse 양쪽 다 `top_n`(보통 4)개만 가져온 뒤
  RRF 로 융합했다. RRF는 "두 리스트 모두에서 검증된 상품"을 끌어올리는 알고리즘이라
  후보 풀이 얕으면(교집합 여지가 거의 없으면) 융합의 recall 개선 효과가 설계 의도보다
  약해진다. → `n_candidates` 산정 조건에 `bm25_service.is_enabled()`를 추가해,
  재랭킹·BM25 하이브리드 둘 중 하나라도 켜져 있으면 `RERANK_CANDIDATES`만큼 넓게
  가져오도록 통일. 최종 개수는 기존과 동일하게 `rerank_service.rerank()`의
  `hits[:top_n]` 슬라이스가 담당하므로 응답 스키마/개수는 불변(회귀 없음).
  `test_search_candidate_count`(재랭킹 OFF 시 `BM25_ENABLED` 기본값도 OFF라 결과 불변)
  등 기존 테스트와 충돌 없음을 코드로 확인.

- **[문서] `ROADMAP.md` / `PRESENTATION_SCRIPT.md` 엔드포인트 표 오류**: 두 문서 모두
  FAQ 엔드포인트를 `GET /faq`로 표기하고 있었으나, `routers/faq.py`는
  `APIRouter(prefix="/chat")` + `@router.get("/faq")`라 실제 경로는 `GET /chat/faq`다
  (라우터 자체 docstring은 이미 정확하게 "/chat/faq"라고 적혀 있었음 — 문서만 갱신이
  안 됨). Spring Boot 연동 시 잘못된 경로로 연결될 위험이 있어 문서 두 곳을 실제
  코드 기준으로 정정했다. 코드 변경 없음(문서 전용 수정).

## 변경 파일 요약(22번)
| 파일 | 변경 |
|------|------|
| `services/personalization_service.py` | 구매이력/임베딩 조회 예외 경로에도 `_cache_put(None)` 적용 → `_locks` 누수 차단 |
| `routers/voice.py` | STT 결과 1000자 초과 시 `ChatRequest` 생성 전 절단(처리되지 않은 500 방지) |
| `graph/rag_pipeline.py` | `n_candidates` 조건에 `bm25_service.is_enabled()` 추가(하이브리드 시 후보 풀 확대) + docstring 동기화 |
| `ROADMAP.md`, `PRESENTATION_SCRIPT.md` | FAQ 엔드포인트 표기 `/faq` → `/chat/faq` 정정(문서 전용) |


---

## 23. [재검토 5차] 실증 재현 버그 2건 + 체크포인트 직렬화 전환 + 정리 일괄

> 이번 재검토는 의심 지점마다 재현 스크립트를 먼저 작성해 실동작을 확인한 뒤
> 수정했고, 두 버그 모두 회귀 방지 테스트(`tests/test_turn_isolation.py`)로 고정했다.
> 테스트 130 → 134개 통과.

### 23-1. [버그] 로그인 멀티턴 스테일 `rag_hits` → 무관한 턴에 `sources` 오부착
**파일:** `routers/chat.py`
checkpointer 는 State 채널을 thread 단위로 영속하는데, `chat.py` 만 `init_state`
에서 `rag_hits` 를 리셋하지 않았다(agent/multi-agent 라우터와 비대칭). SEMANTIC
턴 이후의 모든 비-SEMANTIC 턴(FAQ/주문/잡담)에 직전 검색 hits 가 복원되어,
`if rag_hits:` 만 보던 sources 부착 로직이 무관한 응답에 상품 카드를 실어 보냈다.
→ 2중 방어로 수정: ① 게스트/로그인 양쪽 `init_state` 에 `"rag_hits": []` 리셋,
② sources 부착 조건에 `intent == SEMANTIC_SEARCH` 추가.

### 23-2. [버그] Agent 경로 로그인 멀티턴 메트릭 누적 이중 집계
**파일:** `graph/metrics.py`, `routers/agent_chat.py`, `routers/multi_agent_chat.py`,
`routers/mcp_agent_chat.py`
로그인 사용자는 `result["messages"]` 에 과거 턴 이력 전체가 복원되는데, 집계가
전체 messages 를 순회해 토큰/도구호출/비용이 턴 수에 비례해 부풀려졌다
(재현: 매 턴 실사용 120 토큰 → 턴2 집계 240). metrics.jsonl 기반 PHASE 3
경로 비교와 `ChatResponse.total_tokens` 모두 오염.
→ `snapshot_prior_message_ids()`(invoke 직전 기존 메시지 id 기준선) +
`filter_new_messages()`(신규분만 필터) 를 `graph/metrics.py` 에 추가하고
3개 라우터(+resume 경로)가 공통 사용. 스냅샷 실패 시 빈 집합 → 전체 집계
폴백(측정 실패가 응답을 깨지 않는 기존 정책 유지). 게스트(새 thread)는 동작 불변.
HITL(환불) 턴은 interrupt 이전 구간이 미집계인데, 본 호출이 interrupt_pending
으로 조기 반환해 원래도 미집계였으므로 회귀 아님(과대집계 → 미세 과소집계로 개선).
부수 정리: `mcp_agent_chat.py` 의 중복 `_collect_metrics` 구현을
`graph.metrics.collect_message_metrics` 위임으로 통합.

### 23-3. [아키텍처] 체크포인트 `intent_result` 를 dict(primitive) 저장으로 전환
**파일:** `graph/state.py`, `graph/nodes.py`, `graph/edges.py`, `routers/chat.py`,
`schemas/intent_schema.py`
Pydantic `IntentResult`/`IntentType` 이 체크포인트에 그대로 직렬화되어
langgraph 가 "미등록 타입 — 향후 차단 예정" 경고를 냈다(Redis 영속화 시 실제
리스크). `allowed_msgpack_modules` 등록 방식은 클래스 물리 경로를 저장소에
구워 리팩토링/버전업/멀티워커에 취약하므로 배제하고, **dict 저장 + 경계선 검증**
으로 확정: `classify_node` 가 `model_dump(mode="json")` 로 기록(enum → 순수
문자열), 소비 지점(edges/nodes/guard 경유/chat.py)은 신설
`coerce_intent_result()` 로 복원. IntentResult 직접 주입(테스트 등)도 그대로
허용해 하위호환 유지. 전환 후 직렬화 경고 소멸 확인.

### 23-4. [검색 정확성] LIKE 와일드카드 이스케이프
**파일:** `database/oracle_db.py`, `scripts/test_structured_offline.py`
키워드에 `%`/`_` 가 포함되면 의도치 않은 전체/한글자 와일드카드 매칭이 됐다
(바인드 변수라 인젝션과는 무관). `_escape_like()` + `ESCAPE '\'` 절을
`build_structured_query` / `search_faq_by_keywords` 에 적용.

### 23-5. [잠재함정 제거] LLM 캐시 키에 temperature 포함
**파일:** `graph/llm.py`
기존 `(provider, role)` 캐시 + `.bind(temperature=)` 조합은
`with_structured_output()` 이 `RunnableBinding.__getattr__` 위임으로 '언바운드
원본 모델'에 걸려 바인딩 온도가 조용히 무시되는 함정이 있었다(현재는 INTENT
최초 생성 온도가 0.0 이라 우연히 무해). 캐시 키를 `(provider, role, temperature)`
로 바꾸고 bind 를 제거해 모든 경로에서 생성자 온도를 보장(사용 온도 조합이
소수라 캐시 부담 없음).

### 23-6. [일관성] BM25 hit 의 price NULL 처리 통일
**파일:** `services/bm25_service.py`
`_to_price` 가 None 을 유지해, 무가격 상품이 Dense 경로(색인 규칙 None→0.0)
에선 "0원", BM25 경로에선 "(가격 정보 없음)"으로 갈렸다. 색인 규칙과 동일하게
None→0.0 으로 통일.

### 23-7. [정리] 문서-코드 표기 + 주석 서사 이관 + 정적 분석
- 주석의 env 변수명 오기 8곳 수정: `MODEL_PROVIDER` → 실제 변수 `LLM_PROVIDER`
  (`agent_builder`×2, `model_policy`, `nodes`, `guard`, `multi_agent_builder`,
  `rag_service`×2). 주석대로 .env 를 쓰면 동작하지 않는 실질 오류 문서였음.
- 구모델명 잔존 docstring 4곳: `gemini-2.5-flash` → `gemini-3.1-flash`
  (`pricing.py` 의 2.5-flash 단가는 구버전 폴백 명시라 유지).
- 이전 주석 정리 잔재 3곳 복구: `routers/chat.py` "[LangGraph 전환 — ]" 류
  빈 대시, `scripts/evaluate_rag.py` "- 의 metrics.jsonl" 문장 파손.
- 서사형(변경이력) 주석 12곳을 '현재 설계 근거 1~2줄'로 축약, 이력은 본 문서가
  단일 출처(`oracle_db`, `llm`, `nodes`, `model_factory`, `rag_pipeline`,
  `bm25_service`, `personalization_service`, `voice`).
- 미사용 import 12건 제거(pyflakes 기준 0건 달성): `graph/llm.py`,
  `graph/nodes.py`, `graph/tools.py`, `graph/multi_agent_builder.py`,
  `routers/agent_chat.py`, `routers/multi_agent_chat.py`,
  `scripts/index_products_image.py` 등.

### 스키마 검증 결과 (오탐 방지 기록)
- `CHAT_HISTORY.ANSWER` = CLOB → `save_chat_history` 의 ANSWER 무절단 INSERT 안전.
  `QUESTION VARCHAR2(4000)` → 기존 4000바이트 절단 로직과 정확히 일치.
- `ORDER_ITEM.ID` = IDENTITY 존재 → `_ORDERS_BASE_SQL` 의 `ORDER BY oi.ID` 유효.

## 변경 파일 요약(23번)
`routers/chat.py`, `routers/agent_chat.py`, `routers/multi_agent_chat.py`,
`routers/mcp_agent_chat.py`, `graph/metrics.py`, `graph/state.py`,
`graph/nodes.py`, `graph/edges.py`, `graph/llm.py`, `graph/guard.py`,
`graph/agent_builder.py`, `graph/multi_agent_builder.py`, `graph/model_policy.py`,
`graph/model_factory.py`, `graph/rag_pipeline.py`, `schemas/intent_schema.py`,
`database/oracle_db.py`, `services/bm25_service.py`,
`services/personalization_service.py`, `services/rag_service.py`,
`routers/voice.py`, `scripts/evaluate_rag.py`, `scripts/index_products_image.py`,
`scripts/test_semantic_offline.py`, `scripts/test_structured_offline.py`,
`tests/test_chat_router_graph.py`, `tests/test_multiturn_memory.py`,
`tests/test_turn_isolation.py`(신규)

## Spring Boot 영향
없음. 3개 접점(CHAT_TOKEN 브릿지 / `/admin/products/*` / 공유 Oracle 스키마)과
응답 스키마·엔드포인트 모두 불변. sources 필드는 '부착 조건이 정확해진 것'이지
형식 변경이 아니다.

## 24. [실사용 버그 라운드] 미완료 목록 #3·#4 처리 + 챗봇 팝업 전환 (FastAPI 측)

### #3 개인화 취향 벡터 — 취소/환불 주문 제외
- `database/oracle_db.py` `fetch_purchase_history()` SQL 에
  `AND o.ORDER_STATUS NOT IN ('CANCELLED', 'REFUNDED')` 추가.
- 근거: 취소/환불은 "실제 구매 취향"이 아니며, 취향과 달라 반품한 상품이 오히려
  추천 가중치를 높이는 왜곡이 있었다. 함수 시그니처 불변 → `personalization_service`
  호출부/테스트(monkeypatch 경계) 영향 없음.

### #4 SMALL_TALK 스코프 제한
- `graph/nodes.py` `_SMALL_TALK_SYSTEM_PROMPT` 교체: 인사·가벼운 잡담은 기존대로
  친근 응대, 쇼핑 무관 주제(주식 시황·뉴스·수학·코드 작성 등)는 내용 답변 없이
  "쇼핑 특화 AI" 안내 + 가능한 도움(상품 검색/주문 조회/배송 문의) 제안으로 전환.
- 고정 문구 강제 대신 뉘앙스 지시(합의사항: 토씨 그대로 아님, 상황에 맞게).
- 대안 검토: 인텐트 분류 단계에서 OFF_TOPIC 신설 → 분류기 스키마/라우팅/테스트
  전면 수정 비용 대비 프롬프트 수정으로 충분(SMALL_TALK 노드가 이미 종착지).

### 검증
- `py_compile`/`pyflakes` 0건, pytest 149 passed / 2 failed
  (실패 2건은 `langchain-mcp-adapters` 미설치 환경의 알려진 MCP 테스트 —
  설치 금지 합의, `MCP_ENABLED=false` 운영과 동일 조건).

## Spring Boot 영향
없음. 3개 접점(CHAT_TOKEN 브릿지 / `/admin/products/*` / 공유 Oracle 스키마) 불변.
(챗봇 팝업 전환·추천 캐러셀은 Spring Boot 측 변경으로 별도 zip — FastAPI 계약 불변:
`/chat/stream` payload `question`/`chat_token` 그대로.)

## 25. [배포 직전 라운드] 동시성·멱등성 전수 리뷰 + 보강 6건

배포 전 race condition / partial-write / 멱등성 3개 축 전수 점검 결과.
Partial-write 축은 전 항목 "수정 불필요" 판정(주문 @Transactional 원자성,
AFTER_COMMIT 색인 동기화, CHAT_HISTORY 단건 INSERT+commit 격리 — 설계 검증 완료).

### Spring Boot 수정 5건
1. **[A1] 재고 차감 TOCTOU** — `ProductRepository.findByIdForUpdate()`
   (`PESSIMISTIC_WRITE` = SELECT FOR UPDATE) 신설, 주문 확정 경로
   (createOrder/directOrder)만 잠금 조회로 전환. createOrder 는 productId
   오름차순으로 잠가 다중 상품 동시 주문 간 교착 방지. 읽기 경로는 기존 findById
   유지(조회 성능 무영향). 대안 배제 근거: @Version 은 공유 스키마 컬럼 추가 필요,
   원자적 UPDATE 는 SOLD_OUT 전환 재구성 필요.
2. **[A2] CART UNIQUE(MEMBER_ID, PRODUCT_ID)** — schema.sql 에 UK 추가.
   동시 담기 중복 행 → findByMemberIdAndProductId(단건) NonUniqueResultException
   영구 고장을 DB 레벨에서 차단. (스키마 재배포 확정이라 ALTER 불필요)
3. **[C1] 주문 이중 제출 방지** — detail.js / cart/list.html 제출 시 버튼 잠금
   (다음 틱 disable + BFCache pageshow 복원, 품절 disabled 버튼은 미간섭).
4. **[C2] /orders/checkout GET 제거** — GET 부수효과(URL 접근만으로 주문 생성) 차단.
   실사용처는 cart/list.html POST 폼뿐임을 전수 확인.
5. **[A3+D] System.out → slf4j** — MemberService.saveChatToken(실패 시 세션/DB
   토큰 불일치 경고 로그), AdminAccountInitializer.

### FastAPI 신규 1건
6. **[멱등성] Redis SETNX 중복 요청 차단** — `services/idempotency.py` 신설.
   - 역할 분리: chat_token(세션 식별) vs request_id(요청 멱등키, 클라이언트
     crypto.randomUUID). 세션 토큰을 멱등키로 쓰면 첫 메시지 이후 전부 차단됨.
   - `SET idem:{id} NX EX 300` 선점 실패 → 409 (/ask JSON, /stream 은 기존
     HTTPException→error 이벤트 경로 재사용, 추가 처리 코드 0줄).
   - best-effort: Redis 미설정/미설치/다운/오류 → 통과 (체크포인터와 동일한
     가용성 우선 폴백). request_id 옵션 필드 → 하위호환 100%.
   - 적용 범위 최소화: /chat/ask, /chat/stream 만. voice/agent 계열은 동일 패턴
     확장 가능(범위 통제).
   - 주문(Spring)은 의도적 비채택 — 3중 방어(버튼 disable + PRG + 재고 락)로
     충분, Redis 의존성 신규 추가는 과설계 판단. 비채택 근거 문서화.

### 검증
- pytest **140개 중 138 통과** (신규 test_idempotency.py 6개 포함, MCP 2개는
  알려진 환경 제약). pyflakes 0건, 변경 Java 5파일 구문 파싱 통과, JS node --check 통과.
- ROADMAP.md 테스트 표에서 test_turn_isolation.py 4개 누락 발견·정정
  (기존 표 130개 + 누락 4개 = 직전 실측 134개와 일치 확인).

### 문서
- ROADMAP.md: "PHASE 3 이후 — 배포 직전 운영 보강" 섹션 신설, 테스트/엔드포인트 표 갱신.
- PRESENTATION_SCRIPT.md: "8-1. 동시성·멱등성 방어 설계" 섹션 신설,
  면접 Q&A 2건(동시성/멱등성) 추가, 테스트 현황 140개 기준 갱신.
