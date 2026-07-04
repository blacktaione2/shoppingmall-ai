"""
FastAPI 엔트리포인트
실행: uvicorn main:app --reload --port 8000

포트 구성
- Spring Boot : 8080 (페이지 렌더링)
- FastAPI     : 8000 (본 서버, API 전담)
- ChromaDB    : 8001 (Client-Server 모드)
"""
import os
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# CORS 허용 오리진을 .env(ALLOWED_ORIGINS)에서 읽기 위해 환경변수 로드
load_dotenv()

from routers import chat, faq, admin, voice, agent_chat, multi_agent_chat, mcp_agent_chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 라이프스팬 핸들러.

    [시작 시]
    - [개선] Oracle 커넥션 풀을 생성한다(init_pool). 실패해도 직접연결로
      폴백하므로 기동을 막지 않는다.
    - 체크포인터(Redis/Memory)를 만들어 그래프에 주입한 뒤
      그래프를 미리 컴파일한다. 순서가 중요하다:
        set_checkpointer() → build_graph()
      (build_graph 가 compile 시점에 체크포인터를 바인딩하므로 주입이 먼저)
    - MCP_ENABLED=true 이면 MCP 서버 도구를 미리 로드해 캐시한다.
      첫 /chat/mcp-agent 요청의 지연을 방지하기 위한 prefetch.
      실패해도 예외를 삼키고 정상 기동 — 도구 없이 로컬 도구만으로 동작한다.

    [종료 시]
    - Redis 체크포인터 컨텍스트를 정리한다(연결 종료).
    - [개선] Oracle 커넥션 풀을 정리한다(close_pool).
    """
    # [개선] Oracle 커넥션 풀 생성 (실패 시 직접연결 폴백 — 회귀 0)
    from database.oracle_db import init_pool, close_pool
    init_pool()

    # 체크포인터 주입 + 그래프 사전 컴파일
    from graph.checkpointer import open_checkpointer
    from graph.builder import set_checkpointer, build_graph

    checkpointer, aclose_checkpointer = await open_checkpointer()
    set_checkpointer(checkpointer)
    build_graph()  # 주입된 체크포인터로 그래프를 미리 컴파일(첫 요청 지연 제거)

    # MCP 도구 prefetch (비활성이면 no-op)
    from graph.mcp_tools import prefetch_mcp_tools
    await prefetch_mcp_tools()

    # [하이브리드] BM25 인덱스 구축 (BM25_ENABLED=true 일 때만, 실패해도 기동 막지 않음)
    #   인메모리 인덱스라 재시작 때마다 1회 구축한다(상품 소규모 → 수십 ms).
    from services import bm25_service
    if bm25_service.is_enabled():
        try:
            import asyncio as _asyncio
            from database import oracle_db as _oracle_db
            _rows = await _asyncio.to_thread(_oracle_db.fetch_all_products)
            _n = bm25_service.build_index(_rows)
            logging.getLogger(__name__).info("BM25 인덱스 초기 구축: %d건", _n)
        except Exception:
            logging.getLogger(__name__).exception("BM25 인덱스 구축 실패 → Dense 단독 동작")

    yield

    # 종료 시 정리 — Redis 컨텍스트 닫기(MemorySaver 면 no-op)
    await aclose_checkpointer()
    # [개선] Oracle 커넥션 풀 정리(없으면 no-op)
    close_pool()


app = FastAPI(
    title="ShoppingMall AI Chatbot",
    description="하이브리드 AI 파이프라인 챗봇 서버",
    version="0.4.0",
    lifespan=lifespan,
)

# LangSmith 트레이싱 활성화 상태 로깅 (환경변수 기반, 키 미노출)
from graph.observability import init_observability
init_observability()

# 브라우저 JS → FastAPI 직접 호출을 위한 CORS 설정
# 배포 시 Spring Boot 서버의 실제 origin(도메인/IP:포트)을 추가해야 하므로
#        하드코딩 대신 .env 의 ALLOWED_ORIGINS(쉼표 구분)에서 읽는다.
#        예) ALLOWED_ORIGINS=http://localhost:8080,http://192.168.0.10:8080
#        값이 없으면 로컬 개발용 기본값(localhost/127.0.0.1:8080)을 사용한다.
_DEFAULT_ORIGINS = "http://localhost:8080,http://127.0.0.1:8080"
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(faq.router)
app.include_router(admin.router)   # 관리자 색인 동기화 라우터
app.include_router(voice.router)   # 음성 STT/TTS 라우터
app.include_router(agent_chat.router)   # 단일 Agent 라우터
app.include_router(multi_agent_chat.router)   # 멀티 Agent 라우터
app.include_router(mcp_agent_chat.router)   # MCP 연동 Agent 라우터


@app.get("/health")
def health_check():
    """서버 기동 확인용 헬스체크(얕음 — 프로세스 생존만 확인)"""
    return {"status": "ok", "version": app.version}


@app.get("/health/deep")
async def deep_health_check():
    """[개선] 의존 컴포넌트(Oracle/ChromaDB/Redis)까지 점검하는 깊은 헬스체크.

    배포 후 smoke test, 모니터링 프로브에 사용. 각 컴포넌트를 독립적으로
    점검해 어디가 죽었는지 components 에 표시한다. 하나라도 실패하면
    전체 status 는 'degraded'(HTTP 200 유지 — 프로브가 본문으로 판별).

    [주의] Redis 는 REDIS_URL 미설정(MemorySaver 폴백) 시 점검 대상이 아니므로
           'skipped' 로 표시한다(에러 아님).
    """
    from fastapi.concurrency import run_in_threadpool

    components: dict[str, str] = {}

    # 1) Oracle — 커넥션 획득 + SELECT 1 (풀/직접연결 무관 동일)
    def _ping_oracle() -> None:
        from database.oracle_db import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM DUAL")
                cur.fetchone()

    try:
        await run_in_threadpool(_ping_oracle)
        components["oracle"] = "ok"
    except Exception:
        logging.getLogger(__name__).exception("deep health: Oracle 점검 실패")
        components["oracle"] = "error"

    # 2) ChromaDB — heartbeat (이미 async + to_thread 래핑되어 있음)
    try:
        from services import chroma_service
        await chroma_service.heartbeat()
        components["chroma"] = "ok"
    except Exception:
        logging.getLogger(__name__).exception("deep health: ChromaDB 점검 실패")
        components["chroma"] = "error"

    # 3) Redis — REDIS_URL 설정 시에만 점검(미설정=MemorySaver 폴백 → skipped)
    from graph.checkpointer import is_redis_enabled
    if is_redis_enabled():
        try:
            import os as _os
            from redis.asyncio import from_url as _redis_from_url
            _r = _redis_from_url(_os.getenv("REDIS_URL"))
            await _r.ping()
            await _r.aclose()
            components["redis"] = "ok"
        except Exception:
            logging.getLogger(__name__).exception("deep health: Redis 점검 실패")
            components["redis"] = "error"
    else:
        components["redis"] = "skipped"

    healthy = all(v in ("ok", "skipped") for v in components.values())
    return {
        "status": "ok" if healthy else "degraded",
        "version": app.version,
        "components": components,
    }
