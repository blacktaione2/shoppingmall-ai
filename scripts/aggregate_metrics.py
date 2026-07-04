"""
aggregate_metrics.py — metrics.jsonl 집계 (PHASE3_BENCHMARK.md Step 3 의 완성판).

(route, provider) 별로 평균·p50·p95 레이턴시, 평균 토큰, CPQ(평균 비용), QPD 를 출력한다.
비용/토큰은 Agent 계열 경로에서만 측정되고 router_pipeline 은 0 으로 기록되므로
(LangSmith 로 보완 — graph/metrics.py 참고), 0 인 그룹은 CPQ/QPD 를 '—' 로 표시한다.

사용 예:
    python scripts/aggregate_metrics.py                       # ./metrics.jsonl
    python scripts/aggregate_metrics.py --path /path/metrics.jsonl --csv out.csv
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
    return sorted_vals[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="metrics.jsonl 집계")
    parser.add_argument("--path", default="metrics.jsonl")
    parser.add_argument("--csv", default=None, help="지정 시 CSV 로도 저장")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"파일 없음: {path} (METRICS_ENABLED=true 로 측정을 먼저 실행)")

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # 손상 라인 무시(append 특성상 마지막 라인 절단 가능)

    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by[(r.get("route", "?"), r.get("provider", "?"))].append(r)

    header = (f"{'경로':<16} {'provider':<10} {'N':>4} {'avg':>8} {'p50':>8} {'p95':>8} "
              f"{'tokens':>8} {'CPQ':>11} {'QPD':>9}")
    print(header)
    print("-" * len(header))

    csv_rows = []
    for (route, provider), items in sorted(by.items()):
        lats = sorted(x.get("latency_ms", 0.0) for x in items)
        avg_lat = statistics.mean(lats)
        p50, p95 = pct(lats, 0.50), pct(lats, 0.95)
        avg_tok = statistics.mean(x.get("total_tokens", 0) for x in items)
        avg_cost = statistics.mean(x.get("cost_usd", 0.0) for x in items)
        has_cost = avg_cost > 0
        cpq = f"${avg_cost:.6f}" if has_cost else "—"
        qpd = f"{round(1 / avg_cost):,}건/$" if has_cost else "—"
        print(f"{route:<16} {provider:<10} {len(items):>4} {avg_lat:>7.0f}ms {p50:>7.0f}ms "
              f"{p95:>7.0f}ms {avg_tok:>8.0f} {cpq:>11} {qpd:>9}")
        csv_rows.append({
            "route": route, "provider": provider, "n": len(items),
            "avg_latency_ms": round(avg_lat, 1), "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1), "avg_total_tokens": round(avg_tok, 1),
            "avg_cost_usd": round(avg_cost, 6),
            "qpd": round(1 / avg_cost) if has_cost else "",
        })

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCSV 저장: {args.csv}")


if __name__ == "__main__":
    main()
