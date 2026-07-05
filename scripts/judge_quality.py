"""
judge_quality.py — LLM-as-a-Judge 품질 채점 (PHASE3_BENCHMARK.md 2-1 의 구현).

run_benchmark.py 가 만든 benchmark_results_{label}.jsonl 을 읽어
gpt-5.4-mini 심판으로 정확성/자연스러움/인텐트반영(1~5) + 환각없음(0/1)을 채점한다.

출력:
- 라벨별 평균 품질점수 = (accuracy + fluency + intent) / 3 × no_hallucination
- 환각률 = 1 - mean(no_hallucination)
- --qpd 지정 시 CX-ROI(품질×QPD), RCI((1-환각률)×QPD) 까지 계산
- tag=halluc_bait 부분집합의 환각률 별도 표기 (환각 가드 실측 ⑥ 용)
- tag=scope_test 질문의 응답을 원문 출력 (SMALL_TALK 스코프 제한 육안 확인용)

사용 예 (.env 의 OPENAI_API_KEY 를 자동으로 읽음):
    python scripts/judge_quality.py benchmark_results_openai_gpt54.jsonl --qpd 56
"""
import argparse
import json
import logging
import statistics
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

JUDGE_MODEL = "gpt-5.4-mini"   # 비용 절감 + 일관성 (PHASE3_BENCHMARK.md 2-1)

_JUDGE_PROMPT = """당신은 쇼핑몰 AI 챗봇 응답의 품질 평가자입니다.
아래 사용자 질문과 AI 응답을 보고 각 항목을 1~5점으로 평가하세요.

[질문]: {question}
[응답]: {answer}
[참고 상품 정보]: {context}

평가 항목:
1. 정확성 (1-5): 제공된 상품 정보와 일치하는가? (참고 정보가 '(없음)'이면 일반 상식·정책 안내의 타당성으로 평가)
2. 자연스러움 (1-5): 한국어가 자연스럽고 친절한가?
3. 인텐트 반영 (1-5): 사용자 의도에 맞는 답변인가?
4. 환각 없음 (0/1): 근거 없는 정보 생성이 없는가?

JSON 형식으로만 응답: {{"accuracy": N, "fluency": N, "intent": N, "no_hallucination": N}}"""


def build_context(row: dict) -> str:
    """SEMANTIC 응답의 sources 를 심판 참고 정보로 변환. 없으면 '(없음)'."""
    sources = row.get("sources")
    if not sources:
        return "(없음)"
    lines = []
    for s in sources[:5]:
        if isinstance(s, dict):
            lines.append(" / ".join(f"{k}={v}" for k, v in s.items() if v is not None))
        else:
            lines.append(str(s))
    return "\n".join(lines) or "(없음)"


def judge_one(client: OpenAI, question: str, answer: str, context: str) -> dict | None:
    prompt = _JUDGE_PROMPT.format(question=question, answer=answer, context=context)
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:  # noqa: BLE001 — 채점 실패 행은 집계에서 제외하고 계속
        logger.warning("채점 실패(건너뜀): %s", exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge 품질 채점")
    parser.add_argument("results", help="run_benchmark.py 결과 JSONL")
    parser.add_argument("--qpd", type=float, default=None,
                        help="해당 모델의 QPD (PHASE3_BENCHMARK.md 1-2 표) — CX-ROI/RCI 계산용")
    parser.add_argument("--out", default=None, help="채점 상세 JSONL 저장 경로")
    args = parser.parse_args()

    client = OpenAI()
    rows = [json.loads(l) for l in Path(args.results).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("answer")]

    scored, bait_flags = [], []
    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    for r in rows:
        s = judge_one(client, r["question"], r["answer"], build_context(r))
        if s is None:
            continue
        quality = (s["accuracy"] + s["fluency"] + s["intent"]) / 3 * s["no_hallucination"]
        scored.append({"quality": quality, "no_hallucination": s["no_hallucination"]})
        if r.get("tag") == "halluc_bait":
            bait_flags.append(s["no_hallucination"])
        if r.get("tag") == "scope_test":
            logger.info("[스코프 검증] %s\n  Q: %s\n  A: %s\n", r["id"], r["question"], r["answer"][:200])
        if out_f:
            out_f.write(json.dumps({**r, "judge": s, "quality": round(quality, 2)},
                                   ensure_ascii=False) + "\n")
        logger.info("%s  품질 %.2f  환각없음 %d", r["id"], quality, s["no_hallucination"])
    if out_f:
        out_f.close()

    if not scored:
        raise SystemExit("채점된 행이 없습니다.")

    avg_quality = statistics.mean(x["quality"] for x in scored)
    halluc_rate = 1 - statistics.mean(x["no_hallucination"] for x in scored)
    logger.info("=" * 50)
    logger.info("채점 %d건 | 평균 품질점수 %.2f/5 | 환각률 %.1f%%",
                len(scored), avg_quality, halluc_rate * 100)
    if bait_flags:
        logger.info("halluc_bait 부분집합(%d건) 환각률: %.1f%%  ← 환각 가드 실측(⑥)",
                    len(bait_flags), (1 - statistics.mean(bait_flags)) * 100)
    if args.qpd:
        logger.info("CX-ROI = %.2f × %.0f = %.0f", avg_quality, args.qpd, avg_quality * args.qpd)
        logger.info("RCI    = %.3f × %.0f = %.0f", 1 - halluc_rate, args.qpd,
                    (1 - halluc_rate) * args.qpd)


if __name__ == "__main__":
    main()