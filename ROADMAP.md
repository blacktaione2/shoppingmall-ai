# 쇼핑몰 AI 챗봇 — 개발 로드맵

> 총 3개 Phase, 15개 세부 항목으로 구성된 AI 챗봇 고도화 프로젝트.
> 기존 단순 if/else 파이프라인을 LangGraph 기반으로 전면 재설계하고
> 멀티모달, 개인화, Redis 영속화, 성능 측정까지 단계적으로 구현.

---

## 전체 로드맵 한눈에 보기

```
[기존 레거시 파이프라인] ──→ [PHASE 1] ──→ [PHASE 2] ──→ [PHASE 3]
단순 if/else 인텐트 분류       LangGraph     멀티모달+      Redis+
gpt-4o-mini 분류              전면 재설계    개인화         성능측정
26개 테스트                   85개 테스트   98개 테스트   98개 테스트
```

---

## PHASE 1 — LangGraph 전면 재설계 + AI 기능 완성

> **핵심 목표**: 기존 파이프라인을 StateGraph로 재설계하고 핵심 AI 기능 9가지 완성.
> **테스트**: 85개 통과 (레거시 제외)

### ① LangGraph StateGraph 전면 재설계

**기존 → 변경**:
```
기존: intent = classify() → if intent == "SEMANTIC": ... elif intent == "FAQ": ...
변경: START → classify → [조건부 엣지] → 핸들러 → guard → append_message → END
```

**구현 내용**:
- 10노드 StateGraph 설계 및 컴파일
- `ShoppingState` TypedDict: 노드 간 공유 상태 정의
  - `question`, `history`, `member_id`, `is_guest`
  - `intent_result`, `raw_answer`, `final_answer`
  - `rag_hits` (SEMANTIC 환각 가드 연결용)
- `add_conditional_edges`: 인텐트별 핸들러 자동 라우팅
- ContextVar 사이드채널 제거 → State 직접 전달로 전환
- 기존 INTENT_SYSTEM_PROMPT, 환각 가드 로직 재사용 (중복 없음)

**파일**: `graph/builder.py`, `graph/state.py`, `graph/nodes.py`, `graph/edges.py`

---

### ② 멀티턴 대화 (MemorySaver checkpointer)

**구현 내용**:
- LangGraph `MemorySaver` checkpointer로 대화 이력 자동 보존
- `thread_id` 설계:
  - 로그인 회원: `chat_token` (Spring이 발급한 UUID) → 서버 누적
  - 게스트: 요청마다 새 UUID → stateless (멀티턴 없음)
- `append_message_node`: 매 턴 종료 후 `HumanMessage + AIMessage`를 `messages` 배열에 누적
- 클라이언트 히스토리 폴백: 토큰 없을 때 프론트가 전달한 `history` 배열 사용

**파일**: `graph/builder.py`, `graph/nodes.py` (append_message_node)

---

### ③ STT/TTS (음성 대화)

**구현 내용**:
- **STT**: OpenAI `whisper-1` — 음성 파일(mp3/wav/m4a) → 한국어 텍스트
- **TTS**: OpenAI `tts-1` + `nova` 보이스 — 텍스트 → mp3 bytes → base64 인코딩
- 25MB 파일 크기 제한, multipart/form-data 업로드
- `clean_text_for_tts()`: 이모지·마크다운 기호 제거 (자연스러운 음성 합성)
- 3개 엔드포인트:
  - `POST /chat/transcribe` — STT 단독
  - `POST /chat/tts` — TTS 단독
  - `POST /chat/voice` — STT → AI 파이프라인 → TTS 통합

**파일**: `services/voice_service.py`, `routers/voice.py`

---

### ④ 단일 Agent (create_react_agent)

**구현 내용**:
- LangGraph `create_react_agent` 기반 ReAct 패턴 Agent
- 4개 로컬 도구:
  - `search_products` — 조건 기반 Oracle DB 상품 검색
  - `semantic_search` — ChromaDB 벡터 유사도 검색
  - `search_faq` — Oracle FAQ 테이블 키워드 검색
  - `get_my_orders` — Oracle 주문 실DB 조회 (로그인 필수)
- `InjectedState("member_id")`: 도구 시그니처에 보안 정보 노출 없이 State에서 자동 주입
- Recursion Limit: 12회 (무한 도구 루프 방지)
- 엔드포인트: `POST /chat/agent`

**파일**: `graph/agent_builder.py`, `graph/tools.py`, `routers/agent_chat.py`

---

### ④-a 단일 Agent 확장 — Human-in-the-loop (환불 신청)

**구현 내용**:
- 민감 작업(환불) 자율 실행 방지 → `interrupt`로 사용자 확인 후 진행
- `request_refund` 도구: 내부에서 `langgraph.types.interrupt(payload)` 호출 → 그래프 일시정지
- 2단계 흐름:
  - `POST /chat/agent`: interrupt 감지 시 `interrupt_pending=true` + payload 응답
  - `POST /chat/agent/resume`: `Command(resume="approve"/"reject")`로 같은 thread 재개
- Checkpointer가 멈춘 지점 State 보존 → 추가 인프라 없이 멀티턴 메모리 재사용
- 경계 1: 게스트 차단 (1회성 thread_id라 재개 불가 → InjectedState로 도구 진입 즉시 차단)
- 경계 2: MCP Agent 경로 제외 (`build_agent(include_refund=False)` — 재개 경로 없는 그래프 보호)
- 실제 ORDERS 변경 없음 (DB 쓰기 0 → 스키마/Spring Boot 영향 없음)

**파일**: `graph/tools.py`, `graph/agent_builder.py`, `routers/agent_chat.py`, `routers/mcp_agent_chat.py`, `schemas/chat_schema.py`

---

### ⑤ 멀티 Agent (Supervisor 오케스트레이터)

**구현 내용**:
- Supervisor 패턴:
  - `supervisor` (gpt-5.4-mini): 질문 분석 → product_agent / support_agent / FINISH 라우팅
  - `product_agent`: search_products + semantic_search 도구
  - `support_agent`: search_faq + get_my_orders 도구
- `RouteDecision` Structured Output: supervisor 라우팅 결과 타입 보장
- 핑퐁 방지: `MAX_ROUTING=4` + `recursion_limit=20` 이중 안전장치
- 최종 답변: supervisor가 FINISH 선택 시 직접 `final_answer` 작성
- 엔드포인트: `POST /chat/multi-agent`

**파일**: `graph/multi_agent_builder.py`, `graph/multi_agent_state.py`, `routers/multi_agent_chat.py`

---

### ⑥ LangSmith 모니터링 + 동적 모델 선택

**구현 내용**:
- LangSmith: `LANGSMITH_TRACING=true` 설정 시 모든 LLM 호출 자동 추적
  - 각 노드의 입출력, 토큰 수, 레이턴시 기록
  - `route_metadata()`: `route` 태그로 경로별 필터링 가능
- 동적 모델 선택 (`DYNAMIC_MODEL_SELECTION=true`):
  - `TaskComplexity.SIMPLE` → 저비용 모델
  - `TaskComplexity.COMPLEX` → 고성능 모델
  - 단순 FAQ·주문조회는 자동으로 경량 모델 선택

**파일**: `graph/observability.py`, `graph/model_policy.py`

---

### ⑦ 멀티 모델 추상화 (GPT / Gemini / Claude / DeepSeek)

**구현 내용**:
- `model_factory.py`: `.env`의 `LLM_PROVIDER` 한 줄로 provider 전환
  ```
  LLM_PROVIDER=openai    → gpt-5.4 / gpt-5.4-mini
  LLM_PROVIDER=gemini    → gemini-3.1-flash
  LLM_PROVIDER=anthropic → claude-sonnet-4-6
  LLM_PROVIDER=deepseek  → deepseek-v4-flash
  ```
  (gemini 는 2.5-flash → 3.1-flash 로, deepseek 는 신규 추가로 이후 업데이트됨)
- `lazy import`: 선택된 provider 패키지만 로드 (메모리 절약)
- `(provider, role)` 캐시: 동일 요청 내 모델 객체 재생성 방지
- 모든 노드가 `get_main_llm()` / `get_intent_llm()`으로 추상화 접근
- 라우터 파이프라인의 SEMANTIC 답변 생성(`rag_service.py`)과 환각 재시도
  (`graph/guard.py`)도 이후 동일 팩토리로 통일해, 라우터 경로 전체가
  하나의 provider 로 일관되게 동작한다(벤치마크 공정성 확보).

**파일**: `graph/model_factory.py`, `graph/llm.py`, `services/rag_service.py`, `graph/guard.py`

---

### ⑧ RAG 고도화 (출처 표시 + Cohere 재랭킹)

**구현 내용**:
- ChromaDB top-10 후보 → Cohere rerank-v3.5 → top-4 선별
- 점수 정규화 통일: 재랭킹 적용(Cohere score 0~1) / 미적용(`1/(1+distance)`) 동일 스케일
- `sources` 필드: SEMANTIC_SEARCH 응답에 상품 목록 포함
  ```json
  "sources": [{"product_id": 5, "product_name": "와이드 데님", "price": 72000, "score": 0.91}]
  ```
- 재랭킹 안전장치: Cohere API 장애 시 원본 순서 유지로 자동 폴백

**파일**: `services/rerank_service.py`, `graph/rag_pipeline.py`

---

### ⑧-a RAG 고도화 — 하이브리드 검색 (BM25 Sparse + Dense Fusion, RRF)

**구현 내용**:
- BM25(Sparse, 키워드 정밀 매칭) + ChromaDB 벡터(Dense) 후보 병합
- 고유명사/모델명("나이키 줌 페가수스") 정확 매칭을 BM25로 보완 → recall 향상
- `bm25_service.search()` 반환 스키마를 `chroma_service.search_similar`와 동일하게 설계
  → hit 스키마(id/document/metadata/distance)는 융합 이후에도 그대로 유지
- **[재검토 반영] 융합 알고리즘 = RRF(Reciprocal Rank Fusion), 단순 정규화 아님**:
  - 최초 구현은 BM25 자체 정규화(`distance = 1 - score/max_score`)로 만든 hit을
    `_merge_dedup()`(distance 최소값 채택)에 그대로 태워 병합했다.
  - **문제**: 이 방식은 BM25 리스트의 1등이 항상 `distance=0.0`이 되어, 벡터 검색
    결과가 아무리 좋아도 키워드 1등 상품이 무조건 최상위를 차지했다(RERANK OFF 시
    사실상 순수 BM25로 퇴화 — "하이브리드"라는 이름에 맞지 않는 융합 편향).
  - **해결**: `graph/rag_pipeline._merge_text_rrf()`를 신규 도입. 절대 점수(distance/score)를
    융합 단계에서 완전히 배제하고, id별 `Σ 1/(k + rank)`(순위만)로 점수를 매기는
    스케일-프리 알고리즘으로 교체. Dense/Sparse 양쪽에서 검증된 상품이 자연히
    최상위로 올라간다(진짜 상호보완).
  - **관점 C 보호(회귀 0)**: CLIP 멀티모달 병합용 `_merge_dedup()`는 1바이트도
    건드리지 않았다 — RRF는 텍스트 하이브리드(BM25+Dense) 경로 전용 별도 함수.
  - RRF 결과도 하류 스키마는 동일(`distance` 필드 유지, `rrf_score` 부가 키만 추가)
    → `rerank_service.attach_scores()` 등 하류 코드 무수정.
  - `RRF_K`(.env, 기본 60): 원논문 표준 상수. 순위 1·2등 간 점수 격차를 완만하게 만듦.
- 한국어 토크나이징: 문자 2-gram + 공백 토큰 (형태소 분석기 불필요, 부분 매칭)
- `BM25_ENABLED` 플래그:
  - `false` (기본): BM25 경로 미사용, 기존 동작과 100% 동일
  - `true`: 벡터+BM25 RRF 융합 (RERANK_ENABLED=true와 함께 켜면 Cohere가 최종 재정렬)
- 인메모리 인덱스: lifespan에서 1회 구축, 관리자 CRUD 시 증분 갱신
- `rank_bm25` (순수 Python, torch 불필요 → ARM 서버 부담 0)
- 실패 시 Dense 후보로 폴백 (검색 멈춤 없음)

**파일**: `services/bm25_service.py`, `graph/rag_pipeline.py`, `routers/admin.py`, `main.py`

---

### ⑨ MCP 외부 서비스 연동

**구현 내용**:
- MCP (Model Context Protocol) 어댑터 인프라 구축
- `mcp_config.json`: MCP 서버 등록 파일
- `prefetch_mcp_tools()`: lifespan에서 MCP 도구 미리 로드 (첫 요청 지연 방지)
- 빈 도구 목록 폴백: MCP 서버 미연결 시 로컬 도구만으로 동작
- lazy mcp-agent 빌드: `MCP_ENABLED=true`일 때만 agent 구성
- 엔드포인트: `POST /chat/mcp-agent`

**파일**: `graph/mcp_tools.py`, `routers/mcp_agent_chat.py`

---

## PHASE 2 — 멀티모달 + 개인화 + 주문 실DB 전환

> **핵심 목표**: 이미지 검색, 구매이력 기반 개인화, 주문조회 실DB 전환.
> **테스트**: 98개 통과

### ① 멀티모달 — OpenCLIP Dual Indexing

**구현 내용**:
- OpenCLIP ViT-B-32 모델로 상품 이미지 512차원 임베딩
- ChromaDB 듀얼 컬렉션:
  - `products` (1536차원): 텍스트 임베딩 (기존)
  - `products_image` (512차원): 이미지 임베딩 (신규)
- 검색 결과 merge-deduplication: `product_id` 기준으로 두 컬렉션 결과 통합
- `CLIP_SERVING_ENABLED` 플래그:
  - `false` (기본): CLIP 모델 비로드, 텍스트 검색만 동작
  - `true`: 코드 변경 없이 이미지 검색 활성화
- `IMAGE_URL` 컬럼: 검색 결과 카드 썸네일로도 활용

**파일**: `services/clip_service.py`, `scripts/index_products_image.py`

---

### ② 개인화 추천 레벨 2 (구매이력 임베딩 가중합산)

**구현 내용**:
- 취향 벡터 계산:
  ```
  weight_i = quantity_i × (0.5 ^ (경과일 / 90일))
  취향벡터 = normalize(Σ 상품임베딩_i × weight_i)
  ```
- 검색 벡터 혼합 (α=0.7, β=0.3 기본값):
  ```
  검색벡터 = normalize(0.7 × 질문벡터 + 0.3 × 취향벡터)
  ```
- 적용 경계:
  - 라우터 경로 `semantic_node`에만 적용 (Agent 경로 제외 — PHASE 3 비교 공정성)
  - 게스트·구매이력 0건·DB 장애 → 순수 질문 검색 폴백
- `PERSONALIZATION_CACHE_TTL=300`: 취향 벡터 메모리 캐시 (매 요청 DB 조회 방지)
- **[재검토 반영] `PERSONALIZATION_CACHE_MAX=512`**: 캐시 항목 수 상한. TTL은 '읽을 때만'
  만료되므로, 다시 안 오는 회원 항목이 영구 잔존해 장기 구동 시 메모리가 서서히
  늘어나는 문제가 있었다. 상한 도달 시 만료 항목 우선 정리 → 그래도 초과면 저장
  시각이 오래된 순으로 축출. `clear_cache()`는 계산 중(in-flight)인 락은 건드리지
  않도록 정정(동시 캐시 미스 스탬피드 방어가 캐시 무효화 시점에 뚫리는 것 방지).
- `numpy` 미사용: 순수 Python 리스트 연산으로 의존성 최소화

**파일**: `services/personalization_service.py`

---

### 주문 조회 실DB 전환

**구현 내용**:
- 기존 Mock JSON → Oracle 실DB 연동
- `ORDERS ⋈ ORDER_ITEM ⋈ PRODUCT` 단일 JOIN (N+1 쿼리 방지)
- 어댑터 패턴: 반환 dict 구조를 Mock과 동일하게 유지 → 포맷 함수 무수정 재사용
- `fetch_orders(member_id)` / `fetch_order_by_id(member_id, order_id)` 구현
- `mock/order_mock.py` 보존: 레거시 `pipeline/order_handler.py` 의존성 유지
- **[재검토 반영] `ORDER_STATUS` 영문 enum → 한글 라벨 매핑**: `ORDERS.ORDER_STATUS`는
  DB CHECK 제약상 `SHIPPED`/`DELIVERED` 등 영문만 저장되는데, 화면 표시(`_format_order_*`)와
  이모지 매핑(`_ORDER_STATUS_EMOJI`)은 한글 키를 전제로 작성돼 있어 실DB 연동 시
  영문 코드가 그대로 노출되고 이모지가 전부 기본값(📦)으로 빠지는 불일치가 있었다.
  → `oracle_db._to_status_label()`로 어댑터 단계에서 한글 변환(`PENDING`→"결제대기" 등
  7종), 이모지 맵에도 대응 4키(💳/💰/❌/↩️) 추가. 매핑에 없는 값·레거시 한글 입력은
  원본 그대로 유지해 기존 테스트/데이터와 호환.

---

### 스키마 재설계 (migration_phase2.sql)

**구현 내용**:
- `ORDER_ITEM.COUNT` → `QUANTITY` 리네임 (SQL 예약어 충돌 제거)
- `PRODUCT.IMAGE_URL` VARCHAR2(500) → VARCHAR2(1000) (긴 CDN URL 대비)
- 인덱스 7종 추가:
  - `IX_CHAT_TOKEN_TOKEN` (UNIQUE): 매 챗 요청 토큰 역조회 풀스캔 제거
  - `IX_ORDERS_MEMBER`: 개인화 구매이력 조회
  - `IX_ORDER_ITEM_ORDER` / `IX_ORDER_ITEM_PRODUCT`: 개인화 조인
  - `IX_CART_MEMBER` / `IX_CART_PRODUCT`: FK 인덱스
  - `IX_CHAT_HISTORY_MEMBER`: 대화이력 조회

**파일**: `scripts/migration_phase2.sql`

---

## PHASE 3 — Redis 영속화 + 성능/비용 비교 측정

> **핵심 목표**: MemorySaver → Redis 영속 체크포인터 전환, 경로별 성능 측정 인프라.
> **테스트**: 98개 통과 (유지)

### ① Redis checkpointer (영속 멀티턴 메모리)

**구현 내용**:
- `langgraph-checkpoint-redis==0.5.0` 도입
- `graph/checkpointer.py`: `open_checkpointer()` — REDIS_URL 유무로 분기
  - `REDIS_URL` 설정: `AsyncRedisSaver` → `asetup()` → Redis 인덱스 생성
  - `REDIS_URL` 미설정: `MemorySaver` 폴백 (기존 동작 유지, 회귀 없음)
- `set_checkpointer()` 주입 패턴: lifespan에서 먼저 주입 → `build_graph()` 호출
- `get_checkpointer()` 안전망: 주입 전 호출 시 MemorySaver 자동 폴백
- `_get_graph()` lazy 패턴: 모듈 로드 시점 build_graph() 제거 (체크포인터 주입 전 실행 방지)
- ⚠️ Redis Stack 필수 (일반 redis-server 불가 — RediSearch 모듈 필요)

**파일**: `graph/checkpointer.py`, `graph/builder.py`, `main.py`, `routers/chat.py`

---

### ② 성능/비용 비교 측정 인프라

**구현 내용**:
- `graph/metrics.py`:
  - `RequestMetrics` 데이터클래스: route/provider/model/intent/latency_ms/토큰/비용/도구
  - `collect_message_metrics()`: messages에서 tool_calls/total_tokens/tools_used 집계
  - `collect_token_breakdown()`: input_tokens/output_tokens 구분 집계 (비용 정확도)
  - `record_metrics()`: JSONL 파일 append (`METRICS_ENABLED=true`일 때만)
  - `LatencyTimer`: `with` 블록으로 순수 처리시간 측정
- `services/pricing.py`:
  - provider×model → (input/1K, output/1K) USD 단가표
  - `.env PRICE_*` 오버라이드 지원
  - 미등록 모델 → 비용 0 + 경고 (측정 실패가 응답을 깨지 않음)
- 측정 범위:
  - 라우터 경로: latency만 측정 (토큰은 LangSmith 보완)
  - Agent/멀티Agent: latency + 토큰(입출력 구분) + 도구 + 비용 전부 측정
- 기존 `_collect_metrics()` 중복 → `collect_message_metrics()` 단일 출처로 통합

**파일**: `graph/metrics.py`, `services/pricing.py`, `routers/chat.py`, `routers/agent_chat.py`, `routers/multi_agent_chat.py`

---

### ③ RAGAs — RAG 품질 정량 평가 (오프라인)

**구현 내용**:
- RAGAs로 RAG 응답 품질 3지표 측정: Faithfulness(환각 방어 효과) / Answer Relevancy / Context Precision(재랭킹 효과)
- 운영 코드 0줄 변경: `search_and_rerank()` + `generate_rag_response()` 재사용 (실제 서버와 동일 경로 평가)
- `--make-sample` 평가셋 템플릿 생성, RAGAs 미설치 시 친절 안내 후 종료
- `RERANK_ENABLED` 토글로 재랭킹 적용 전후 품질 비교 가능
- 평가 전용 의존성(`ragas`, `datasets`)은 서버 런타임 불필요 (분리)

**파일**: `scripts/evaluate_rag.py`

---

### ④ RAGAs CI 게이트 (LLMOps — 품질 회귀 방지)

**구현 내용**:
- GitHub Actions 품질 게이트: RAG 파일 변경 PR마다 자동 평가 → 임계값 미달 시 빌드 실패
- `evaluate_rag.py --ci` 모드: Faithfulness 등 기준 미만이면 `exit(1)`
- **운영 인프라 격리 (Option 2)**: 운영 Oracle/ChromaDB 대신 고정 픽스처를 러너 내 임시 ChromaDB 컨테이너에 색인 → 운영 데이터/네트워크 무의존, 재현 가능
- `paths` 필터로 RAG 관련 변경에만 실행 (비용 통제)
- 임계값 보수적 시작(0.6) → 베이스라인 확보 후 점진 상향

**파일**: `.github/workflows/rag_quality.yml`, `scripts/ci_index_fixtures.py`, `scripts/ci_fixtures/`, `scripts/evaluate_rag.py`

---

## PHASE 3 이후 — 배포 직전 운영 보강 (동시성·멱등성·UI)

배포 전 최종 점검(race condition / partial-write / 멱등성 3개 축 전수 리뷰)에서
나온 보강 라운드. 상세 이력은 IMPROVEMENTS.md 24~25번.

### 동시성 (Spring Boot)
- **재고 차감 TOCTOU 수정**: 주문 확정 경로에 `PESSIMISTIC_WRITE`(SELECT FOR UPDATE)
  행 잠금 도입 — 마지막 1개에 동시 주문 2건이 모두 통과하던 초과판매 차단.
  다중 상품 주문은 productId 오름차순으로 잠가 교착 방지.
  (`@Version` 낙관적 락은 공유 스키마 컬럼 추가가 필요해 배제)
- **CART UNIQUE(MEMBER_ID, PRODUCT_ID)**: 동시 담기가 중복 행을 만들면 단건 조회가
  NonUniqueResultException 으로 영구 고장 → DB 제약으로 차단.

### 멱등성 (계층 방어)
- **챗봇 API — Redis SETNX 멱등키**: 클라이언트가 전송마다 `crypto.randomUUID()` 로
  `request_id` 발급 → 서버가 `SET idem:{id} NX EX 300` 선점 실패 시 409 차단.
  세션 토큰(chat_token)과 멱등키(request_id)를 분리한 것이 핵심 —
  세션 토큰을 멱등키로 쓰면 첫 메시지 이후 전부 차단된다.
  Redis 미설정/다운 시 통과(best-effort) — 체크포인터와 동일한 가용성 우선 폴백.
- **주문 — 3중 방어**: 버튼 disable(1차) + PRG(2차) + 재고 행 잠금이 초과분 거절(최후).
  주문까지 Redis 멱등키를 넣는 것은 이 규모에 과설계로 판단해 채택하지 않음.
- `/orders/checkout` GET 제거 — GET 부수효과(URL 접근만으로 주문 생성) 차단.

### 챗봇 UI — 팝업 전환 (Spring Boot)
- 데스크톱: 플로팅 버튼 → `window.open('/chatForm')` 별도 창. 페이지를 이동해도
  대화가 끊기지 않고, 같은 이름의 창을 리로드 없이 재사용(focus)한다.
- 모바일: 기존 인라인 패널 유지(팝업이 새 탭으로 열리는 한계 회피).
- 상품 상세에 같은 카테고리 추천 캐러셀 추가(팀원 UI + `relatedProducts` 컨트롤러 보강).

### 개인화/프롬프트 정합
- 개인화 취향 벡터에서 `CANCELLED`/`REFUNDED` 주문 제외 — 반품 상품이 추천 가중치를
  높이는 왜곡 수정.
- SMALL_TALK 스코프 제한 — 쇼핑 무관 질문(주식 시황 등)은 내용 답변 없이
  "쇼핑 특화 AI" 안내로 전환.

---

## 배포 환경 구성

### 인프라 전환 (단일 서버 통합)

**기존**: Oracle Cloud 인스턴스 2대 분리
- 인스턴스 1 (1GB): Spring Boot :8080
- 인스턴스 2 (1GB): FastAPI :8000 + ChromaDB :8001

**변경**: 단일 서버 통합
- Oracle Cloud **VM.Standard.A1.Flex** (ARM, 4 OCPU, 24GB RAM, 50GB)
- **Always Free** 범위 내 (총 4 OCPU, 24GB 한도)
- Spring Boot + FastAPI + ChromaDB 모두 동일 서버
- `FASTAPI_BASE_URL=http://localhost:8000` (외부 통신 → localhost)
- `ALLOWED_ORIGINS=http://64.110.119.49:8080,...` (CORS 설정)

### 자동 기동 (systemd)

```
chromadb.service (먼저 기동)
    └─ fastapi.service (chromadb.service 완료 후 기동)
```

---

## 테스트 현황

| 파일 | 테스트 수 | 대상 |
|------|----------|------|
| test_graph_routing.py | 8개 | 인텐트별 노드 라우팅(오프라인 단위) |
| test_chat_router_graph.py | 3개 | chat.py ↔ LangGraph end-to-end 통합 |
| test_multiturn_memory.py | 4개 | 멀티턴 하이브리드(checkpointer) 통합 |
| test_history_trimming.py | 6개 | 멀티턴 이력 트리밍(컨텍스트 윈도우 초과 방지) |
| test_agent_graph.py | 4개 | 단일 Agent 그래프 흐름 |
| test_agent_endpoint.py | 2개 | /chat/agent 엔드포인트 통합 |
| test_multi_agent_graph.py | 4개 | 멀티 Agent(Supervisor) 그래프 흐름 |
| test_multi_agent_endpoint.py | 2개 | /chat/multi-agent 엔드포인트 통합 |
| test_mcp_integration.py | 5개 | MCP 연동 |
| test_rag_enhancement.py | 7개 | RAG 고도화(재랭킹 + 출처 표시) |
| test_bm25_hybrid.py | 12개 | BM25 하이브리드 검색 + RRF 융합 |
| test_personalization.py | 9개 | 개인화 취향 벡터 서비스 |
| test_orders_db_adapter.py | 5개 | 주문 실DB 어댑터(`_build_orders_from_rows`) |
| test_voice_endpoints.py | 8개 | STT/TTS/voice 엔드포인트 통합 |
| test_model_factory.py | 6개 | 멀티 모델 추상화 팩토리 |
| test_stream_util.py | 18개 | SSE 스트리밍 유틸(청킹/이벤트 포맷) |
| test_sse_parse_algorithm.py | 5개 | 프론트 SSE 파싱 알고리즘 회귀 |
| test_observability_and_policy.py | 7개 | LangSmith 관측성 + 동적 모델 선택 |
| test_human_in_the_loop.py | 7개 | 환불 interrupt/resume(HITL) |
| test_metrics_dashboard.py | 5개 | 성능/비용 대시보드(집계 + admin 엔드포인트) |
| test_pipeline_context_member_id.py | 3개 | member_id 사이드채널(레거시 파이프라인) |
| test_turn_isolation.py | 4개 | 턴 간 상태 격리(스테일 rag_hits 회귀 방지) |
| test_idempotency.py | 6개 | Redis SETNX 멱등성(중복 차단/폴백/SETNX 계약) |
| **소계(pytest)** | **140개** | **138개 통과 + MCP 2개 실패(알려진 환경 제약)** |
| test_order_handler_member_id.py | 5개 시나리오 | order_handler member_id 연동(비-pytest 독립 스크립트, `python tests/test_order_handler_member_id.py`로 수동 실행) |

> 위 표는 `grep`으로 소스를 직접 스캔해 산출한 실측치다(2026-07-04 기준 재동기화 —
> 직전 표에서 test_turn_isolation.py 4개가 누락돼 있던 것을 정정). MCP 2개 실패는
> `langchain-mcp-adapters` 미설치 환경의 알려진 제약으로, `MCP_ENABLED=false` 운영과
> 동일 조건이다(설치 시 starlette 충돌 — 설치하지 않기로 합의).
> `test_order_handler_member_id.py`는 pytest가 수집하는 `def test_*` 함수가
> 0개라 pytest 총계(140개)에는 포함하지 않고, 5개 체크 시나리오를 담은 독립 스크립트로
> 별도 표기했다.

---

## 엔드포인트 전체 목록

| 메서드 | 경로 | 인증 | 용도 |
|--------|------|------|------|
| POST | /chat/ask | 선택 | 텍스트 응답 (라우터 그래프) · request_id 멱등키 지원 |
| POST | /chat/stream | 선택 | SSE 스트리밍 응답 · request_id 멱등키 지원 |
| POST | /chat/agent | 선택 | 단일 ReAct Agent |
| POST | /chat/agent/resume | 로그인 | 환불 확인 재개 (Human-in-the-loop) |
| POST | /chat/multi-agent | 선택 | 멀티 Agent Supervisor |
| POST | /chat/mcp-agent | 선택 | MCP 외부 서비스 연동 |
| POST | /chat/transcribe | 없음 | STT 단독 (음성→텍스트) |
| POST | /chat/tts | 없음 | TTS 단독 (텍스트→음성) |
| POST | /chat/voice | 선택 | 음성 대화 통합 |
| GET | /chat/faq | 없음 | FAQ 목록 조회 |
| POST | /admin/products/reindex | ADMIN_KEY | 상품 1건 ChromaDB 재색인 |
| DELETE | /admin/products/{id} | ADMIN_KEY | 상품 1건 ChromaDB 삭제 |
| POST | /admin/products/reindex-all | ADMIN_KEY | 전체 재색인 |
| GET | /admin/metrics | 없음 | 성능/비용 대시보드 (HTML) |
| GET | /admin/metrics/summary | ADMIN_KEY | 경로별·모델별 집계 (JSON) |
| GET | /health | 없음 | 헬스체크 |
| GET | /health/deep | 없음 | 의존성(Oracle/Chroma/Redis) 점검 |

> 선택 인증: chat_token 있으면 로그인 회원 처리, 없으면 게스트 처리

