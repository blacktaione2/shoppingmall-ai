"""
graph/metrics.py
성능/비용 비교 측정 코어.

[역할]
1) collect_message_metrics(messages):
   Agent/멀티Agent 실행 결과 messages 에서 도구호출/토큰(입·출력 구분)/도구이름을
   집계한다. 기존 라우터의 중복 _collect_metrics 를 이 한 곳으로 통합한다.
2) RequestMetrics:
   한 요청의 측정 결과를 담는 데이터클래스(경로/모델/지연/토큰/비용/도구).
3) record_metrics(m):
   RequestMetrics 를 JSONL 파일에 1줄 append. 측정 실패가 본 응답을 깨지 않도록
   모든 예외를 삼키고 로깅만 한다.

[설계 메모]
- 기존 라우터(agent_chat/multi_agent_chat)의 _collect_metrics 는 total_tokens 만
  합산했다. 비용 정확도를 위해 여기서는 input/output 을 구분 집계한다.
  · 하위호환: collect_message_metrics 는 (tool_calls, total_tokens, tools_used) 를
    그대로 반환하되, 입·출력 구분값이 필요한 record 단계에서는 별도 함수
    collect_token_breakdown 을 쓴다. (라우터 응답 형식은 1도 안 바뀐다)
- 적재 경로(JSONL)는 .env METRICS_LOG_PATH 로 지정(기본 ./metrics.jsonl).
  METRICS_ENABLED=false(기본) 면 record 는 no-op → 로컬에서 의도적으로 켤 때만 적재.
- uvicorn --workers 1 전제라 파일 append 경합은 없다. 멀티워커 전환 시 라인
  단위 append 의 원자성에 의존(문서에 명시). 영속·정밀 집계가 필요해지면
  이후 DB/전용 수집기로 승격한다.
"""
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

load_dotenv()

logger = logging.getLogger(__name__)

# 비블로킹 기록용 백그라운드 태스크의 강참조 보관소.
# loop.create_task 반환을 어딘가 잡아두지 않으면 완료 전에 GC 될 수 있다(파이썬 공식 경고).
# 완료 시 done 콜백으로 스스로 discard 하므로 집합이 무한히 커지지 않는다.
_pending_writes: set = set()

_DEFAULT_LOG_PATH = "metrics.jsonl"


def _metrics_enabled() -> bool:
    """로컬 측정 적재 on/off (기본 off). 'true'/'1'/'yes' 만 활성."""
    return os.getenv("METRICS_ENABLED", "false").strip().lower() in (
        "true", "1", "yes",
    )


def _log_path() -> str:
    return (os.getenv("METRICS_LOG_PATH") or _DEFAULT_LOG_PATH).strip()


async def snapshot_prior_message_ids(app, config: dict) -> set:
    """invoke 직전, checkpointer 에 이미 누적된 메시지 id 집합을 반환한다.

    로그인 사용자는 result["messages"] 에 과거 턴 이력 전체가 함께 복원되므로,
    이 기준선 없이 전체를 합산하면 토큰/도구호출/비용이 턴 수에 비례해 부풀려진다
    (누적 이중 집계). invoke 전에 기존 id 를 떠두고, 집계 시 신규 메시지만 남긴다.

    실패하면 빈 집합을 반환한다 → filter_new_messages 가 전체 집계로 폴백
    (측정 실패가 응답을 깨지 않는 기존 정책과 동일).
    """
    try:
        snap = await app.aget_state(config)
        msgs = (snap.values or {}).get("messages", []) if snap else []
        return {m.id for m in msgs if getattr(m, "id", None) is not None}
    except Exception:
        logger.exception("사전 메시지 스냅샷 실패 → 전체 메시지 집계로 폴백")
        return set()


def filter_new_messages(messages: list, prior_ids: set) -> list:
    """스냅샷(prior_ids) 이후 추가된 신규 메시지만 남긴다.

    게스트(새 thread)는 prior_ids 가 비어 있어 전체가 그대로 반환된다(기존 동작).
    """
    if not prior_ids:
        return messages
    return [m for m in messages if getattr(m, "id", None) not in prior_ids]


def collect_message_metrics(messages: list) -> tuple[int, int, list[str]]:
    """messages 에서 (tool_calls, total_tokens, tools_used) 집계.

    기존 agent_chat/multi_agent_chat 의 _collect_metrics 와 '동일한 반환 계약'.
    두 라우터가 이 함수로 위임해 중복을 제거한다(동작 불변 → 무회귀).
    """
    tool_calls = 0
    total_tokens = 0
    tools_used: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls += len(msg.tool_calls or [])
            usage = getattr(msg, "usage_metadata", None)
            if usage and usage.get("total_tokens"):
                total_tokens += usage["total_tokens"]
        elif isinstance(msg, ToolMessage):
            if msg.name:
                tools_used.append(msg.name)
    return tool_calls, total_tokens, tools_used


def collect_token_breakdown(messages: list) -> tuple[int, int]:
    """messages 에서 (prompt_tokens, completion_tokens) 를 구분 집계한다.

    비용 환산 정확도를 위해 입·출력을 나눠 더한다. usage_metadata 가 없는
    메시지는 건너뛴다(0 으로 간주). total_tokens 만 있고 입·출력 구분이 없는
    경우는 input 으로 몰아 근사하지 않고 그대로 0/0 처리(과대청구 방지).
    """
    prompt_tokens = 0
    completion_tokens = 0
    for msg in messages:
        if isinstance(msg, AIMessage):
            usage = getattr(msg, "usage_metadata", None)
            if not usage:
                continue
            prompt_tokens += usage.get("input_tokens", 0) or 0
            completion_tokens += usage.get("output_tokens", 0) or 0
    return prompt_tokens, completion_tokens


@dataclass
class RequestMetrics:
    """한 요청의 측정 결과(경로 비교/비용 분석의 1행)."""
    route: str                     # 'router_pipeline' | 'single_agent' | 'multi_agent'
    provider: str                  # 'openai' | 'gemini' | 'anthropic' | 'deepseek'
    model_main: str                # 본응답 모델명(예: gpt-5.4)
    intent: str                    # 분류 인텐트 또는 경로 식별자(AGENT/MULTI_AGENT)
    is_guest: bool
    latency_ms: float              # 순수 처리시간(라우터가 측정)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    tools_used: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    ts: str = ""                   # ISO8601 UTC 타임스탬프

    def __post_init__(self):
        if not self.ts:
            self.ts = datetime.now(timezone.utc).isoformat()


def _write_line_sync(path: str, line: str) -> None:
    """JSONL 1줄 동기 append (실제 파일 I/O). 호출부에서 예외를 처리한다."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def record_metrics(m: RequestMetrics) -> None:
    """RequestMetrics 를 JSONL 1줄로 append. 실패해도 절대 예외를 올리지 않는다.

    METRICS_ENABLED=false 면 no-op. 측정은 부가 기능이라 본 응답 흐름과 분리한다.

    [비블로킹] 라우터의 async 핸들러 안에서 호출되므로, 실행 중인 이벤트 루프가
    있으면 파일 쓰기를 asyncio.to_thread 로 오프로드해 이벤트 루프 블로킹을 피한다
    (fire-and-forget). 루프가 없으면(동기 컨텍스트/테스트) 그 자리에서 동기로 쓴다.
    어느 경우든 호출부 시그니처(record_metrics(m))는 그대로라 회귀가 없다.
    """
    if not _metrics_enabled():
        return
    try:
        line = json.dumps(asdict(m), ensure_ascii=False)
        path = _log_path()
    except Exception:
        logger.exception("metrics 직렬화 실패(무시): route=%s", getattr(m, "route", "?"))
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # 이벤트 루프 안: 별도 스레드로 파일 쓰기를 넘겨 루프를 막지 않는다.
        async def _async_write():
            try:
                await asyncio.to_thread(_write_line_sync, path, line)
            except Exception:
                logger.exception("metrics 비동기 기록 실패(무시)")
        task = loop.create_task(_async_write())
        # 강참조 유지(GC 로 태스크 유실 방지) + 완료 시 자동 정리.
        _pending_writes.add(task)
        task.add_done_callback(_pending_writes.discard)
    else:
        # 동기 컨텍스트(테스트 등): 그 자리에서 동기 기록.
        try:
            _write_line_sync(path, line)
        except Exception:
            logger.exception("metrics 동기 기록 실패(무시)")


class LatencyTimer:
    """with 블록으로 순수 처리시간(ms)을 재는 헬퍼.

    [사용]
        with LatencyTimer() as t:
            result = await graph.ainvoke(...)
        t.elapsed_ms  # → float (ms)
    """
    def __init__(self):
        self.elapsed_ms = 0.0
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000.0, 2)
        return False  # 예외를 삼키지 않음


# ════════════════════════════════════════════════════════════════════════
# 집계 (대시보드용 — 읽기 전용)
# ════════════════════════════════════════════════════════════════════════
# metrics.jsonl 을 읽어 경로별/모델별로 집계한다. record_metrics 가 쓴 파일을
# '읽기만' 하므로 쓰기 경합이 없고, 파일이 없거나 깨진 줄은 건너뛴다(예외 없음).
_ROUTE_LABELS = {
    "router_pipeline": "라우터 파이프라인",
    "single_agent":    "단일 Agent",
    "multi_agent":     "멀티 Agent",
}


def _empty_group() -> dict:
    """집계 누적용 빈 그룹(요청 수/레이턴시 합/토큰 합/비용 합/도구호출 합)."""
    return {
        "count": 0,
        "latency_ms_sum": 0.0,
        "prompt_tokens_sum": 0,
        "completion_tokens_sum": 0,
        "total_tokens_sum": 0,
        "tool_calls_sum": 0,
        "cost_usd_sum": 0.0,
    }


def _finalize_group(key: str, g: dict, label: str | None = None) -> dict:
    """누적 그룹을 평균/합계가 담긴 표시용 dict 로 변환."""
    n = g["count"] or 1   # 0 나눗셈 방지(count=0 이면 평균은 0으로)
    has = g["count"] > 0
    return {
        "key": key,
        "label": label or key,
        "count": g["count"],
        "avg_latency_ms": round(g["latency_ms_sum"] / n, 2) if has else 0.0,
        "total_tokens": g["total_tokens_sum"],
        "prompt_tokens": g["prompt_tokens_sum"],
        "completion_tokens": g["completion_tokens_sum"],
        "avg_tokens": round(g["total_tokens_sum"] / n, 1) if has else 0.0,
        "tool_calls": g["tool_calls_sum"],
        "cost_usd": round(g["cost_usd_sum"], 6),
        "avg_cost_usd": round(g["cost_usd_sum"] / n, 6) if has else 0.0,
    }


def summarize_metrics(limit: int | None = None) -> dict:
    """metrics.jsonl 을 집계해 경로별/모델별 요약을 반환한다.

    Args:
        limit: 최근 N줄만 집계(None 이면 전체). 파일이 커졌을 때 최근 구간만 보기 위함.

    Returns:
        {
          "available": bool,        # 파일이 있고 파싱된 줄이 1개 이상인가
          "total_requests": int,
          "by_route":   [ {key,label,count,avg_latency_ms,total_tokens,...}, ... ],
          "by_provider":[ {key,label,count,...}, ... ],
          "cost_total_usd": float,
          "log_path": str,
        }
    실패(파일 없음/전부 파싱 실패)해도 예외 없이 available=False 로 반환한다.
    """
    path = _log_path()
    result = {
        "available": False,
        "total_requests": 0,
        "by_route": [],
        "by_provider": [],
        "cost_total_usd": 0.0,
        "log_path": path,
    }

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return result
    except Exception:
        logger.exception("metrics 파일 읽기 실패: %s", path)
        return result

    if limit is not None and limit > 0:
        lines = lines[-limit:]

    route_groups: dict[str, dict] = {}
    provider_groups: dict[str, dict] = {}
    total = 0
    cost_total = 0.0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue   # 깨진 줄은 건너뛴다

        route = str(row.get("route") or "unknown")
        provider = str(row.get("provider") or "unknown")
        latency = float(row.get("latency_ms") or 0.0)
        p_tok = int(row.get("prompt_tokens") or 0)
        c_tok = int(row.get("completion_tokens") or 0)
        t_tok = int(row.get("total_tokens") or 0)
        tools = int(row.get("tool_calls") or 0)
        cost = float(row.get("cost_usd") or 0.0)

        for groups, key in ((route_groups, route), (provider_groups, provider)):
            g = groups.setdefault(key, _empty_group())
            g["count"] += 1
            g["latency_ms_sum"] += latency
            g["prompt_tokens_sum"] += p_tok
            g["completion_tokens_sum"] += c_tok
            g["total_tokens_sum"] += t_tok
            g["tool_calls_sum"] += tools
            g["cost_usd_sum"] += cost

        total += 1
        cost_total += cost

    if total == 0:
        return result

    # 경로는 정해진 순서(라우터→단일→멀티)로, 그 외 키는 뒤에 붙인다.
    ordered_routes = list(_ROUTE_LABELS.keys())
    route_keys = ordered_routes + [k for k in route_groups if k not in ordered_routes]
    by_route = [
        _finalize_group(k, route_groups[k], _ROUTE_LABELS.get(k, k))
        for k in route_keys if k in route_groups
    ]
    by_provider = [
        _finalize_group(k, provider_groups[k])
        for k in sorted(provider_groups.keys())
    ]

    result.update({
        "available": True,
        "total_requests": total,
        "by_route": by_route,
        "by_provider": by_provider,
        "cost_total_usd": round(cost_total, 6),
    })
    return result
