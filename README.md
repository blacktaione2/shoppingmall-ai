# 쇼핑몰 AI 챗봇 — FastAPI/LangGraph 백엔드

Oracle 기반 쇼핑몰의 AI 챗봇 파이프라인을 전담하는 FastAPI 서버입니다.
Spring Boot(회원/상품/장바구니/주문, Thymeleaf UI)와 이원화된 구조에서
**AI 파이프라인 전체(LangGraph 라우팅, RAG, 멀티모달, 멀티턴, 멀티에이전트,
음성, MCP 연동)**를 담당합니다.

- Spring Boot 저장소: [shoppingmall-spring](https://github.com/blacktaione2/shoppingmall-spring)
- 4인 팀 프로젝트 중 AI 백엔드(FastAPI) 파트를 담당했습니다.

## 아키텍처

```
브라우저 (챗봇 위젯 chatbot.js)
   │  직접 호출(CORS)
   ▼
FastAPI :8000  ──────────────►  ChromaDB :8001 (벡터 검색, client-server)
   │        LangGraph 라우터           Redis (멀티턴 체크포인터)
   │        6개 인텐트 분기             Oracle Autonomous DB (상품/주문/회원)
   │
   └──────► Spring Boot :8080 (회원/상품/장바구니/주문, CHAT_TOKEN 발급)
```

- 챗봇 위젯은 `/chat/stream`(LangGraph 라우터 경로)을 사용합니다.
- Spring Boot가 로그인 시 발급한 `CHAT_TOKEN`(UUID)으로 게스트/회원을 식별합니다.
- 상품 등록/수정 시 Spring Boot가 `AFTER_COMMIT` 이벤트로 `/admin/products/*`를
  호출해 ChromaDB 색인을 동기화합니다.

## 주요 기능

**라우팅 & RAG**
- LangGraph `StateGraph` 기반 6개 인텐트 라우팅(STRUCTURED_QUERY / SEMANTIC_SEARCH /
  FAQ / ORDER_INQUIRY / COMPLAINT / SMALL_TALK)
- BM25 + Dense 하이브리드 검색, RRF(Reciprocal Rank Fusion)로 융합
- Cohere 재랭킹, 출처(sources) 포함 응답
- 상품 가격/재고 등 정형 질의는 텍스트 매칭이 아닌 LLM 구조화 출력
  (`ProductAttributeQuery`)으로 처리
- Context-Rich 프리픽스 청킹: 긴 상품 설명을 `[상품명 | 카테고리]` 프리픽스와 함께
  분할 색인(`services/chunking_service.py`)

**멀티모달**
- OpenCLIP 이미지 임베딩 기반 이미지 유사도 검색(dual indexing)
- Vision LLM(gpt-4o-mini, Structured Outputs) 이미지 태깅 — 색상/소재/스타일을
  자동 추출해 텍스트 검색에도 반영(`services/vision_tagging_service.py`)

**대화**
- Redis 기반 멀티턴 체크포인터(24시간 TTL), 게스트/회원 모두 지원
- 다중 세션(채팅방) 지원 — 세션 생성/목록/이력 조회/삭제
- 구매 이력 기반 개인화(임베딩 가중치 조정)
- 신뢰도 기반 명확화 질문(clarify_node)

**에이전트 & 연동**
- 단일 Agent(Function Calling) / 멀티 Agent(Supervisor) 두 가지 경로
- MCP(Model Context Protocol) 연동 — Gmail·Slack으로 환불 알림 이중화
- STT/TTS 음성 대화
- 멀티 LLM 프로바이더 팩토리(OpenAI/Gemini/Claude/DeepSeek) + 동적 모델 선택,
  LangSmith 트레이싱

**운영**
- RAGAs 오프라인 평가 + CI 게이트
- 관리자 메트릭 대시보드(`/admin/metrics`)
- 프로바이더별 비용/응답시간 벤치마크(`PHASE3_BENCHMARK.md`)

## API 엔드포인트

| 경로 | 설명 |
|---|---|
| `POST /chat/stream` | 챗봇 위젯이 사용하는 LangGraph 라우터 경로(SSE 스트리밍) |
| `POST /chat/ask` | 비스트리밍 단건 응답 |
| `POST /chat/sessions`, `GET /chat/sessions` | 채팅방 생성/목록 |
| `GET /chat/sessions/{id}/messages`, `DELETE /chat/sessions/{id}` | 세션 이력 조회/삭제 |
| `POST /agent` , `POST /agent/resume` | 단일 Agent(Function Calling) 경로 |
| `POST /multi-agent` | 멀티 Agent(Supervisor) 경로 |
| `POST /mcp-agent` | MCP 도구 연동 Agent 경로 |
| `POST /voice/transcribe`, `POST /voice/tts`, `POST /voice/voice` | 음성 STT/TTS |
| `GET /faq` | FAQ 목록 |
| `POST /admin/products/reindex`, `DELETE /admin/products/{id}`, `POST /admin/products/reindex-all` | 상품 색인 동기화(Spring Boot → FastAPI, `X-ADMIN-KEY` 인증) |
| `GET /admin/metrics/summary`, `GET /admin/metrics` | 운영 메트릭 |
| `GET /health`, `GET /health/deep` | 헬스체크(얕음 / Oracle·ChromaDB·Redis 점검) |

## 기술 스택

- **API**: FastAPI, LangGraph, LangChain
- **LLM**: OpenAI / Gemini / Claude / DeepSeek (멀티 프로바이더), Cohere(재랭킹)
- **DB**: Oracle Autonomous DB, ChromaDB(client-server), Redis Stack(RediSearch)
- **멀티모달**: OpenCLIP(ViT-B-32), gpt-4o-mini Vision(Structured Outputs)
- **인프라**: Oracle Cloud VM(ARM), systemd, git 기반 배포(push → 서버 pull → restart)
- **평가**: RAGAs, LangSmith

## 로컬 실행

```bash
# Python 3.11+ 필요
pip install -r requirements.txt --break-system-packages

cp .env.example .env
# .env 에 실제 값 채우기(OPENAI_API_KEY, DB_DSN/DB_USER/DB_PASSWORD, CHROMA_HOST 등)

# ChromaDB 별도 기동 (client-server 모드)
python -m chromadb.cli.cli run --path ./chroma_data --port 8001

uvicorn main:app --reload --port 8000
```

`GET /health/deep`로 Oracle/ChromaDB/Redis 연결 상태를 확인할 수 있습니다.

### 상품 색인

```bash
python -m scripts.index_products          # 텍스트 임베딩(BM25+Dense)
python -m scripts.index_products_image     # CLIP 이미지 임베딩 + Vision 태깅
```

### 테스트

```bash
pytest
```

## 환경변수

`.env.example` 참고. 핵심 카테고리만 요약:

| 분류 | 주요 변수 |
|---|---|
| DB | `DB_DSN`, `DB_USER`, `DB_PASSWORD`, `WALLET_DIR`, `WALLET_PASSWORD`, `ORACLE_POOL_MIN/MAX/INCREMENT` |
| 벡터 검색 | `CHROMA_HOST`, `CHROMA_PORT`, `BM25_ENABLED`, `RRF_K`, `RERANK_ENABLED`, `RERANK_PROVIDER`, `COHERE_API_KEY` |
| 청킹 | `CHUNK_THRESHOLD_CHARS`, `CHUNK_OVERLAP_CHARS` |
| 멀티모달 | `CLIP_SERVING_ENABLED`, `CLIP_MODEL_NAME`, `CLIP_PRETRAINED`, `CLIP_IMAGE_COLLECTION` |
| LLM | `LLM_PROVIDER`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `DYNAMIC_MODEL_SELECTION` |
| 멀티턴 | `REDIS_URL`, `HISTORY_MAX_MESSAGES` |
| 개인화 | `PERSONALIZATION_ENABLED`, `PERSONALIZATION_ALPHA/BETA`, `PERSONALIZATION_*` |
| MCP/알림 | `MCP_ENABLED`, `MCP_CONFIG_PATH`, `SLACK_MCP_REFUND_CHANNEL_ID`, `ADMIN_ALERT_EMAIL` |
| 연동/보안 | `ADMIN_KEY`(Spring Boot 색인 동기화 인증), `ALLOWED_ORIGINS` |
| 관측성 | `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `METRICS_ENABLED` |

## 문서

- [`ROADMAP.md`](./ROADMAP.md) — 아키텍처/PHASE별 구현 내역, 테스트 현황
- [`IMPROVEMENTS.md`](./IMPROVEMENTS.md) — 번호별 개선 이력(설계 배경/트레이드오프)
- [`PHASE3_BENCHMARK.md`](./PHASE3_BENCHMARK.md) — 프로바이더별 비용/응답시간 실측
- [`MEASUREMENT_GUIDE.md`](./MEASUREMENT_GUIDE.md) — 성능 측정 방법론

## 배포

```
git push                                  # 로컬 → GitHub
ssh ubuntu@<server-ip>
cd ~/shoppingmall_ai && git pull origin main
sudo systemctl restart fastapi
```

Oracle Cloud VM(ARM) 위에서 `chromadb.service` → `fastapi.service` 순서로
systemd가 자동 기동합니다.
