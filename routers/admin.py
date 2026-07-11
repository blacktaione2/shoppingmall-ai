"""
routers/admin.py
================
Spring Boot 관리자 화면의 상품 CRUD 와 ChromaDB 'products' 색인을 동기화하는 라우터.

[배경]
- PRODUCT 테이블은 이제 Spring Boot(JPA)가 직접 등록/수정/삭제한다.
- 그런데 SEMANTIC_SEARCH 는 ChromaDB 벡터 색인을 쓰므로, 상품이 바뀌어도 색인이
  자동으로 갱신되지 않으면 "DB엔 있는데 AI 검색엔 안 나오는" 불일치가 생긴다.
- 그래서 Spring Boot(FastApiSyncService)가 상품 변경 직후 아래 엔드포인트를 호출해
  ChromaDB 색인을 맞춘다. 동기화 실패는 Spring Boot 쪽에서 경고로만 처리하고,
  관리자가 '전체 재색인'으로 언제든 복구할 수 있다.

[인증]
- 브라우저가 아니라 Spring Boot 서버만 호출하는 내부 연동 엔드포인트다.
- X-ADMIN-KEY 헤더(.env 의 ADMIN_KEY 와 일치)를 요구해 외부 노출을 막는다.
  키가 비어 있거나 불일치하면 403.

[엔드포인트]
- POST   /admin/products/reindex       {product_id}  : 등록/수정된 상품 1건 재색인(upsert)
- DELETE /admin/products/{product_id}                : 삭제된 상품 1건 색인 제거
- POST   /admin/products/reindex-all                 : 전체 재색인(수동 복구)

[색인 로직 재사용]
- 메타데이터/임베딩 텍스트 구성은 scripts/index_products.py 의 헬퍼를 그대로 재사용해
  최초 인덱싱과 1건 재색인의 규칙(특히 id=str(product_id), price=float)이 어긋나지 않게 한다.
"""
import os
import hmac
import logging

from fastapi import APIRouter, Header, HTTPException, Depends, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from database.oracle_db import fetch_product_by_id, fetch_all_products
from services import embed_service, chroma_service, bm25_service
from services.chunking_service import build_chunk_documents
from graph.metrics import summarize_metrics
# 전체 재색인(POST /admin/products/reindex-all)은 scripts/index_products.main() 재사용
from scripts.index_products import main as reindex_all_main

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------- 인증 ----------------

def verify_admin_key(x_admin_key: str = Header(default=None)) -> None:
    """X-ADMIN-KEY 헤더를 .env 의 ADMIN_KEY 와 대조. 불일치/미설정 시 403.

    키 비교는 hmac.compare_digest 로 '상수 시간' 비교한다. 일반 != 비교는 문자열이
    앞에서부터 다르면 즉시 끝나, 응답 시간 차이로 키를 한 글자씩 추측당할 수 있다
    (타이밍 공격). 내부 연동 엔드포인트라 위험은 낮지만 인증 비교의 모범 사례를 따른다.
    """
    expected = os.getenv("ADMIN_KEY")
    if not expected or not x_admin_key:
        raise HTTPException(status_code=403, detail="관리자 인증에 실패했습니다.")
    if not hmac.compare_digest(x_admin_key, expected):
        raise HTTPException(status_code=403, detail="관리자 인증에 실패했습니다.")


# ---------------- 요청/응답 스키마 ----------------

class ReindexRequest(BaseModel):
    """POST /admin/products/reindex 요청 바디."""
    product_id: int = Field(..., description="재색인할 상품 PRODUCT_ID")


class SyncResponse(BaseModel):
    """동기화 결과 응답. total 은 작업 후 ChromaDB 컬렉션 총 건수."""
    ok: bool
    total: int
    message: str


# ---------------- 엔드포인트 ----------------

@router.post("/products/reindex", response_model=SyncResponse,
             dependencies=[Depends(verify_admin_key)])
async def reindex_product(req: ReindexRequest) -> SyncResponse:
    """등록/수정된 상품 1건을 Oracle 에서 읽어 ChromaDB 에 upsert(재색인)한다.

    [대규모 청크 처리] build_chunk_documents() 가 설명 길이에 따라 문서 1개
    (짧은 설명, 현재 카탈로그 전량) 또는 N개(긴 설명 — 프리픽스 삽입된 청크)를
    돌려준다. scripts/index_products.py(전체 재색인)와 동일 함수를 써서 두 경로의
    청킹 동작이 갈라지지 않게 한다.

    [delete-then-upsert] upsert 전에 chroma_service.delete_product() 로 이 상품의
    기존 문서를 '전부'(청크 개수 무관) 먼저 지운다. 상품 설명을 수정해 청크 개수가
    바뀌는 경우(예: 3개 → 2개, 혹은 임계값 아래로 짧아짐)에도 옛 청크가 orphan 으로
    남지 않도록 하기 위함이다.
    """
    # 1) Oracle 단건 조회 (동기 → 스레드풀)
    row = await run_in_threadpool(fetch_product_by_id, req.product_id)
    if row is None:
        # Spring Boot 가 이미 삭제한 상품을 재색인 요청한 비정상 케이스 → 404
        raise HTTPException(status_code=404, detail=f"PRODUCT_ID={req.product_id} 상품을 찾을 수 없습니다.")

    pid = row.get("product_id")

    # 2) 청크 문서 구성(짧은 설명이면 길이 1, 길면 프리픽스 삽입된 N개)
    docs = build_chunk_documents(row)

    # 3) 배치 임베딩(문서 개수만큼 — 청킹 안 됐으면 1개)
    embeddings = await embed_service.get_embeddings([d["document"] for d in docs])

    # 4) 기존 문서 전부 삭제 후 upsert(delete-then-upsert, 청크 수 변경 대응)
    await chroma_service.delete_product(pid)
    total = await chroma_service.upsert_products(
        ids=[d["id"] for d in docs],
        embeddings=embeddings,
        documents=[d["document"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
    )
    logger.info("상품 재색인 완료: product_id=%s (청크 %d개, 컬렉션 총 %d건)",
                pid, len(docs), total)

    # [하이브리드] BM25 인덱스도 증분 갱신(플래그 OFF 면 no-op, 실패해도 메인 경로 불방해)
    # BM25 는 설계상 청킹 대상에서 제외(키워드 빈도 기반이라 전체 문서로도 잘 동작) —
    # 원본 row 그대로 넘겨 기존 단일 문서 색인 규칙을 유지한다.
    try:
        bm25_service.upsert_one(row)
    except Exception:
        logger.exception("BM25 단건 갱신 실패(무시): product_id=%s", pid)

    return SyncResponse(ok=True, total=total, message="재색인 완료")


@router.delete("/products/{product_id}", response_model=SyncResponse,
               dependencies=[Depends(verify_admin_key)])
async def delete_product_index(product_id: int) -> SyncResponse:
    """삭제된 상품 1건을 ChromaDB 색인에서 제거한다(존재하지 않아도 안전).

    텍스트 컬렉션(products)과 이미지 컬렉션(products_image) 양쪽에서
    제거해 일관성을 맞춘다. 이미지 컬렉션에 해당 상품이 없어도(이미지 미등록) 무시된다.
    응답 total 은 텍스트 컬렉션 기준 건수다(기존 의미 유지).
    """
    total = await chroma_service.delete_product(product_id)
    image_total = await chroma_service.delete_image_product(product_id)
    logger.info(
        "상품 색인 삭제 완료: product_id=%s (텍스트 %d건 / 이미지 %d건)",
        product_id, total, image_total,
    )

    # [하이브리드] BM25 인덱스에서도 제거(플래그 OFF 면 no-op, 실패 무시)
    try:
        bm25_service.delete_one(product_id)
    except Exception:
        logger.exception("BM25 단건 삭제 실패(무시): product_id=%s", product_id)

    return SyncResponse(ok=True, total=total, message="색인 삭제 완료")


@router.post("/products/reindex-all", response_model=SyncResponse,
             dependencies=[Depends(verify_admin_key)])
async def reindex_all() -> SyncResponse:
    """전체 상품을 다시 색인한다(동기화가 깨졌을 때 수동 복구용).

    scripts/index_products.py 의 main() 을 그대로 호출해 최초 인덱싱과 동일한 경로로 처리한다.
    """
    await reindex_all_main()
    total = await chroma_service.count()
    logger.info("전체 재색인 완료 (총 %d건)", total)

    # [하이브리드] BM25 인덱스도 전체 재구축(플래그 OFF 면 no-op, 실패 무시)
    try:
        if bm25_service.is_enabled():
            rows = await run_in_threadpool(fetch_all_products)
            n = bm25_service.build_index(rows)
            logger.info("BM25 전체 재구축 완료 (%d건)", n)
    except Exception:
        logger.exception("BM25 전체 재구축 실패(무시)")

    return SyncResponse(ok=True, total=total, message="전체 재색인 완료")


# ════════════════════════════════════════════════════════════════════════
# 성능/비용 대시보드 (읽기 전용 — metrics.jsonl 집계 시각화)
# ════════════════════════════════════════════════════════════════════════
# 이미 record_metrics 가 매 요청을 metrics.jsonl 에 적재한다. 아래 두
# 엔드포인트는 그 파일을 '읽기만' 해서 경로별(라우터/단일/멀티 Agent)·모델별로
# 집계·시각화한다. 새 테이블/새 의존성/Spring Boot 변경 없이 FastAPI 안에서 끝난다.
#   - GET /admin/metrics/summary : 집계 JSON (X-ADMIN-KEY 필요)
#   - GET /admin/metrics         : 위 JSON 을 Chart.js 로 그리는 HTML 한 장
# HTML 페이지 자체는 인증 없이 열리되(브라우저 직접 접근), 데이터 조회 시 사용자가
# 입력한 키를 헤더에 실어 /summary 를 호출하는 구조라 데이터는 키로 보호된다.

@router.get("/metrics/summary", dependencies=[Depends(verify_admin_key)])
async def metrics_summary(
    limit: int | None = Query(
        default=None, ge=1,
        description="최근 N개 요청만 집계(미지정 시 전체)",
    ),
) -> dict:
    """metrics.jsonl 을 경로별/모델별로 집계해 반환한다(읽기 전용).

    파일 I/O 는 동기이므로 스레드풀로 넘겨 이벤트 루프를 막지 않는다.
    파일이 없거나 적재된 데이터가 없으면 available=False 로 응답한다(에러 아님).
    """
    return await run_in_threadpool(summarize_metrics, limit)


@router.get("/metrics", response_class=HTMLResponse)
async def metrics_dashboard() -> HTMLResponse:
    """성능/비용 비교 대시보드 HTML(한 장). Chart.js CDN 사용, 새 의존성 없음.

    페이지 로드는 인증 없이 열리고, 관리자 키는 상단 입력창에 넣으면 그 값으로
    /admin/metrics/summary 를 호출해 차트를 그린다(키는 브라우저 메모리에만 유지).
    """
    return HTMLResponse(content=_DASHBOARD_HTML)


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AI 챗봇 성능/비용 대시보드</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root { --bg:#0f1420; --card:#182034; --line:#2a3550; --fg:#e6ecf5; --muted:#8fa0bd; --accent:#5b8cff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family: system-ui, -apple-system, "Segoe UI", "Noto Sans KR", sans-serif; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 24px 20px 60px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
  .bar { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom:20px; }
  input, button, select { font-size:14px; padding:8px 10px; border-radius:8px;
         border:1px solid var(--line); background:var(--card); color:var(--fg); }
  button { background:var(--accent); border-color:var(--accent); color:#fff; cursor:pointer; }
  button:hover { opacity:.9; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-bottom:22px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
  .card .k { color:var(--muted); font-size:12px; margin-bottom:6px; }
  .card .v { font-size:22px; font-weight:700; }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin-bottom:18px; }
  .panel h2 { font-size:15px; margin:0 0 12px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:right; padding:8px 10px; border-bottom:1px solid var(--line); }
  th:first-child, td:first-child { text-align:left; }
  th { color:var(--muted); font-weight:600; }
  .msg { color:var(--muted); font-size:13px; padding:10px 0; }
  .err { color:#ff8080; }
  canvas { max-height:320px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>AI 챗봇 성능/비용 대시보드</h1>
  <div class="sub">경로별(라우터 / 단일 Agent / 멀티 Agent)·모델별 요청 수, 평균 레이턴시, 토큰, 비용 비교</div>

  <div class="bar">
    <input id="key" type="password" placeholder="X-ADMIN-KEY" style="min-width:220px" />
    <select id="limit">
      <option value="">전체 기간</option>
      <option value="100">최근 100건</option>
      <option value="500">최근 500건</option>
      <option value="1000">최근 1000건</option>
    </select>
    <button id="load">불러오기</button>
    <span id="status" class="msg"></span>
  </div>

  <div class="cards" id="cards"></div>

  <div class="panel"><h2>경로별 평균 레이턴시 (ms)</h2><canvas id="latencyChart"></canvas></div>
  <div class="panel"><h2>경로별 누적 토큰 (input / output)</h2><canvas id="tokenChart"></canvas></div>
  <div class="panel"><h2>경로별 누적 비용 (USD)</h2><canvas id="costChart"></canvas></div>

  <div class="panel">
    <h2>경로별 상세</h2>
    <table id="routeTable">
      <thead><tr>
        <th>경로</th><th>요청 수</th><th>평균 레이턴시(ms)</th>
        <th>누적 토큰</th><th>평균 토큰</th><th>도구 호출</th><th>누적 비용(USD)</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <div class="panel">
    <h2>모델(provider)별 상세</h2>
    <table id="providerTable">
      <thead><tr>
        <th>Provider</th><th>요청 수</th><th>평균 레이턴시(ms)</th>
        <th>누적 토큰</th><th>누적 비용(USD)</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<script>
const charts = {};
function fmt(n, d=0){ return (n ?? 0).toLocaleString(undefined,{maximumFractionDigits:d}); }
function setStatus(t, err=false){ const s=document.getElementById('status'); s.textContent=t; s.className='msg'+(err?' err':''); }

function renderCards(data){
  const el = document.getElementById('cards');
  el.innerHTML = '';
  const items = [
    ['총 요청 수', fmt(data.total_requests)],
    ['누적 비용(USD)', '$'+fmt(data.cost_total_usd, 6)],
    ['경로 수', fmt((data.by_route||[]).length)],
    ['Provider 수', fmt((data.by_provider||[]).length)],
  ];
  for(const [k,v] of items){
    const d=document.createElement('div'); d.className='card';
    d.innerHTML=`<div class="k">${k}</div><div class="v">${v}</div>`; el.appendChild(d);
  }
}
function fillTable(id, rows, cols){
  const tb = document.querySelector(`#${id} tbody`); tb.innerHTML='';
  for(const r of rows){
    const tr=document.createElement('tr');
    tr.innerHTML = cols.map(c=>`<td>${c(r)}</td>`).join('');
    tb.appendChild(tr);
  }
}
function drawBar(id, labels, datasets, stacked=false){
  if(charts[id]) charts[id].destroy();
  charts[id]=new Chart(document.getElementById(id),{
    type:'bar',
    data:{ labels, datasets },
    options:{
      responsive:true,
      plugins:{ legend:{ labels:{ color:'#e6ecf5' } } },
      scales:{
        x:{ stacked, ticks:{ color:'#8fa0bd' }, grid:{ color:'#2a3550' } },
        y:{ stacked, beginAtZero:true, ticks:{ color:'#8fa0bd' }, grid:{ color:'#2a3550' } }
      }
    }
  });
}

async function load(){
  const key = document.getElementById('key').value.trim();
  if(!key){ setStatus('X-ADMIN-KEY 를 입력하세요.', true); return; }
  const limit = document.getElementById('limit').value;
  const url = '/admin/metrics/summary' + (limit ? ('?limit='+limit) : '');
  setStatus('불러오는 중...');
  let res;
  try {
    res = await fetch(url, { headers: { 'X-ADMIN-KEY': key } });
  } catch(e){ setStatus('네트워크 오류: '+e.message, true); return; }
  if(res.status === 403){ setStatus('인증 실패: 키가 올바르지 않습니다.', true); return; }
  if(!res.ok){ setStatus('오류: HTTP '+res.status, true); return; }

  const data = await res.json();
  if(!data.available){
    setStatus('적재된 측정 데이터가 없습니다. (METRICS_ENABLED=true 인지, 요청이 있었는지 확인)', true);
    renderCards(data);
    fillTable('routeTable', [], []); fillTable('providerTable', [], []);
    ['latencyChart','tokenChart','costChart'].forEach(id=>{ if(charts[id]) charts[id].destroy(); });
    return;
  }
  setStatus('업데이트 완료 · 총 '+fmt(data.total_requests)+'건');
  renderCards(data);

  const routes = data.by_route || [];
  const rLabels = routes.map(r=>r.label);
  drawBar('latencyChart', rLabels, [
    { label:'평균 레이턴시(ms)', data:routes.map(r=>r.avg_latency_ms), backgroundColor:'#5b8cff' }
  ]);
  drawBar('tokenChart', rLabels, [
    { label:'input 토큰', data:routes.map(r=>r.prompt_tokens), backgroundColor:'#4bc0c0' },
    { label:'output 토큰', data:routes.map(r=>r.completion_tokens), backgroundColor:'#ffa14b' }
  ], true);
  drawBar('costChart', rLabels, [
    { label:'누적 비용(USD)', data:routes.map(r=>r.cost_usd), backgroundColor:'#c77dff' }
  ]);

  fillTable('routeTable', routes, [
    r=>r.label, r=>fmt(r.count), r=>fmt(r.avg_latency_ms,2),
    r=>fmt(r.total_tokens), r=>fmt(r.avg_tokens,1), r=>fmt(r.tool_calls), r=>'$'+fmt(r.cost_usd,6)
  ]);
  fillTable('providerTable', data.by_provider||[], [
    r=>r.label, r=>fmt(r.count), r=>fmt(r.avg_latency_ms,2),
    r=>fmt(r.total_tokens), r=>'$'+fmt(r.cost_usd,6)
  ]);
}
document.getElementById('load').addEventListener('click', load);
document.getElementById('key').addEventListener('keydown', e=>{ if(e.key==='Enter') load(); });
</script>
</body>
</html>
"""


# [임시 디버그 — Slack MCP 도구 미검출 원인 조사용]
# 확인 끝나면 제거할 것. force_reload=True로 캐시를 무시하고 실제 로드를 강제 실행해
# 어떤 도구가 잡히는지, 몇 개 서버가 잡히는지 그대로 노출한다.
@router.get("/debug/mcp-tools", dependencies=[Depends(verify_admin_key)])
async def debug_mcp_tools():
    from graph.mcp_tools import is_mcp_enabled, _load_config
    connections = _load_config()
    result = {
        "mcp_enabled": is_mcp_enabled(),
        "configured_servers": list(connections.keys()),
    }
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        client = MultiServerMCPClient(connections=connections)
        tools = await client.get_tools()
        result["tool_names"] = [getattr(t, "name", "?") for t in tools]
    except Exception as e:
        # [임시 디버그] get_mcp_tools()는 이 예외를 삼키고 빈 리스트로 폴백하는데,
        # 여기서는 원인 파악을 위해 그대로 노출한다.
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
    return result
