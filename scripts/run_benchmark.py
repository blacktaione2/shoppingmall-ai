"""
run_benchmark.py — PHASE3 provider 비교 측정 러너.

benchmark_questions.json 의 질문 50개를 지정 엔드포인트로 순회 전송하고,
질문별 결과(응답/인텐트/클라이언트 레이턴시/인텐트 일치 여부)를 JSONL 로 남긴다.
서버 측 상세 지표(토큰/비용/순수 처리시간)는 METRICS_ENABLED=true 일 때
metrics.jsonl 에 별도로 쌓이며 aggregate_metrics.py 로 집계한다.

사용 예 (서버에서, provider 를 .env 로 바꿔 재기동한 뒤):
    python scripts/run_benchmark.py --label openai_gpt54
    python scripts/run_benchmark.py --label gemini_flash --endpoint /chat/agent
    python scripts/run_benchmark.py --label openai_gpt54 --chat-token <UUID>  # 주문조회 포함 측정

의존성: requests (pip install requests)
"""
import argparse
import json
import logging
import time
import uuid
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_QUESTIONS = Path(__file__).parent / "benchmark_questions.json"


def load_questions(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["questions"]


def run_one(base_url: str, endpoint: str, question: str, chat_token: str | None,
            timeout: float) -> tuple[dict | None, float, str | None]:
    """1개 질문 전송. (응답 JSON, 왕복 레이턴시 ms, 오류메시지) 반환."""
    payload = {"question": question, "request_id": str(uuid.uuid4())}
    if chat_token:
        payload["chat_token"] = chat_token
    started = time.perf_counter()
    try:
        res = requests.post(base_url + endpoint, json=payload, timeout=timeout)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if res.status_code != 200:
            return None, elapsed_ms, f"HTTP {res.status_code}: {res.text[:120]}"
        return res.json(), elapsed_ms, None
    except requests.RequestException as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return None, elapsed_ms, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="PHASE3 provider 비교 벤치마크 러너")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="FastAPI 주소 (기본: 서버 로컬 — 네트워크 노이즈 제거)")
    parser.add_argument("--endpoint", default="/chat/ask",
                        help="/chat/ask(라우터) 또는 /chat/agent(토큰 측정 경로)")
    parser.add_argument("--questions", default=str(_DEFAULT_QUESTIONS))
    parser.add_argument("--label", required=True,
                        help="결과 구분 라벨 (예: openai_gpt54, gemini_flash)")
    parser.add_argument("--chat-token", default=None,
                        help="로그인 측정 시 CHAT_TOKEN. 미지정이면 게스트(ORDER_INQUIRY 는 로그인 안내 응답)")
    parser.add_argument("--out", default=None,
                        help="결과 JSONL 경로 (기본: benchmark_results_{label}.jsonl)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="요청 간 대기(초) — rate limit 방지")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    out_path = Path(args.out or f"benchmark_results_{args.label}.jsonl")

    n_ok = n_err = n_intent_match = 0
    latencies: list[float] = []

    with out_path.open("w", encoding="utf-8") as f:
        for q in questions:
            body, elapsed_ms, err = run_one(
                args.base_url, args.endpoint, q["question"], args.chat_token, args.timeout)
            row = {
                "label": args.label,
                "endpoint": args.endpoint,
                "id": q["id"],
                "question": q["question"],
                "expected_intent": q["expected_intent"],
                "tag": q.get("tag"),
                "client_latency_ms": round(elapsed_ms, 1),
            }
            if err:
                n_err += 1
                row["error"] = err
                logger.info("%s  ERROR  %s", q["id"], err)
            else:
                n_ok += 1
                latencies.append(elapsed_ms)
                got_intent = body.get("intent")
                match = got_intent == q["expected_intent"]
                n_intent_match += int(match)
                row.update({
                    "intent": got_intent,
                    "intent_match": match,
                    "confidence": body.get("confidence"),
                    "answer": body.get("answer"),
                    "sources": body.get("sources"),
                })
                logger.info("%s  %-17s %s  %6.0fms  %s", q["id"], got_intent,
                            "○" if match else "✗", elapsed_ms, q["question"][:24])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            time.sleep(args.sleep)

    logger.info("-" * 60)
    logger.info("성공 %d / 오류 %d", n_ok, n_err)
    if n_ok:
        lat_sorted = sorted(latencies)
        p50 = lat_sorted[len(lat_sorted) // 2]
        p95 = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))]
        logger.info("인텐트 분류 정확도: %d/%d (%.1f%%)",
                    n_intent_match, n_ok, n_intent_match / n_ok * 100)
        logger.info("클라이언트 레이턴시: 평균 %.0fms / p50 %.0fms / p95 %.0fms",
                    sum(latencies) / len(latencies), p50, p95)
    logger.info("결과 저장: %s (품질 채점은 judge_quality.py, 서버 지표는 aggregate_metrics.py)",
                out_path)


if __name__ == "__main__":
    main()
