"""
scripts/evaluate_rag.py
[추가] RAGAs 기반 RAG 파이프라인 정량 평가 (오프라인 전용).

[목적]
- 성능 대시보드의 metrics.jsonl(레이턴시/비용)은 '얼마나 빠르고 싼가'를 측정하지만,
  '응답이 얼마나 정확한가(환각은 없는가)'는 수치가 없다. 이 스크립트가 그 빈자리를
  채운다: 실제 검색(search_and_rerank) + 실제 생성(generate_rag_response)을 그대로
  거친 뒤, RAGAs 로 아래 지표를 산출한다.
    · Faithfulness      : 답변이 검색 컨텍스트에 충실한가(=환각 방어 효과의 정량 지표)
    · Answer Relevancy  : 답변이 질문에 적절한가
    · Context Precision : 검색된 컨텍스트가 질문에 관련 있는가(=재랭킹 효과)

[운영 코드 영향 0]
- 이 파일은 '평가용 스크립트'다. FastAPI 엔드포인트/그래프/스키마를 일절 수정하지 않고,
  운영 모듈(graph.rag_pipeline, services.rag_service)을 import 해 '읽기'만 한다.
- 따라서 Spring Boot 는 물론, 실행 중인 FastAPI 서버에도 영향이 없다.

[재랭킹 효과 비교(권장)]
- .env 의 RERANK_ENABLED=true/false 를 바꿔 두 번 실행하면
  "Cohere 재랭킹 적용 전후 Context Precision/Faithfulness 변화"를 표로 만들 수 있다.

[사용법]
    # 1) 의존성(평가 전용, 서버에는 불필요)
    pip install ragas datasets

    # 2) 평가셋 준비: 없으면 아래 명령이 샘플 템플릿을 생성한다.
    python -m scripts.evaluate_rag --make-sample

    # 3) eval_dataset.json 의 question/ground_truth 를 도메인에 맞게 채운 뒤 실행
    python -m scripts.evaluate_rag --dataset eval_dataset.json

[평가셋 형식] (eval_dataset.json)
    [
      {"question": "겨울에 따뜻한 외투 추천", "ground_truth": "패딩/코트 등 보온 아우터"},
      {"question": "5만원 이하 운동화 있어?",   "ground_truth": "5만원 이하 운동화 목록"}
    ]
    · ground_truth 는 Answer Relevancy/Context Precision 계산에 쓰인다(없어도 일부 지표는 동작).

[주의]
- RAGAs 는 내부적으로 LLM(평가자) + 임베딩을 호출하므로 OPENAI_API_KEY 가 필요하고,
  평가 항목 수만큼 토큰 비용이 발생한다(소규모 평가셋 권장).
- ChromaDB 서버가 떠 있어야 검색이 동작한다(운영과 동일 전제).
"""
import argparse
import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv

# 운영 파이프라인 재사용 (수정 없이 import 만)
from graph.rag_pipeline import search_and_rerank
from services.rag_service import generate_rag_response, build_product_context

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate_rag")

_SEMANTIC_TOP_K = 4
_SAMPLE_PATH = "eval_dataset.json"

_SAMPLE_DATASET = [
    {"question": "겨울에 따뜻하게 입을 외투 추천해줘", "ground_truth": "패딩이나 코트 등 보온성 좋은 아우터"},
    {"question": "5만원 이하 운동화 있어?", "ground_truth": "5만원 이하 가격대의 운동화"},
    {"question": "데이트할 때 입기 좋은 옷 추천", "ground_truth": "깔끔하고 단정한 데이트룩 상의/하의"},
    {"question": "선물하기 좋은 상품 알려줘", "ground_truth": "선물용으로 적합한 상품"},
    {"question": "캠핑 갈 때 필요한 용품", "ground_truth": "캠핑/아웃도어 관련 상품"},
]


def _make_sample() -> None:
    """평가셋 템플릿 파일을 생성한다(이미 있으면 덮어쓰지 않음)."""
    if os.path.exists(_SAMPLE_PATH):
        logger.info("이미 존재함 → 덮어쓰지 않음: %s", _SAMPLE_PATH)
        return
    with open(_SAMPLE_PATH, "w", encoding="utf-8") as f:
        json.dump(_SAMPLE_DATASET, f, ensure_ascii=False, indent=2)
    logger.info("샘플 평가셋 생성: %s (question/ground_truth 를 도메인에 맞게 수정하세요)",
                _SAMPLE_PATH)


def _load_dataset(path: str) -> list[dict]:
    if not os.path.exists(path):
        logger.error("평가셋 파일이 없습니다: %s  (--make-sample 로 템플릿 생성)", path)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        logger.error("평가셋은 비어 있지 않은 리스트여야 합니다: %s", path)
        sys.exit(1)
    return data


async def _build_eval_rows(dataset: list[dict]) -> list[dict]:
    """각 질문을 실제 RAG 파이프라인에 통과시켜 RAGAs 입력 행을 만든다.

    반환 행 형식(RAGAs SingleTurnSample 호환):
        {question, answer, contexts(list[str]), ground_truth}
    """
    rows: list[dict] = []
    for i, item in enumerate(dataset, start=1):
        question = (item.get("question") or "").strip()
        if not question:
            continue
        ground_truth = (item.get("ground_truth") or "").strip()

        # 1) 실제 검색(+재랭킹) — 운영 semantic 경로와 동일
        hits = await search_and_rerank(question, top_n=_SEMANTIC_TOP_K)
        # contexts: RAGAs 가 문자열 리스트를 기대 → hit 별 컨텍스트 문자열
        contexts = [build_product_context([h]) for h in hits] or ["(검색 결과 없음)"]

        # 2) 실제 생성 — 운영 RAG 응답과 동일
        answer = await generate_rag_response(question, hits, history=[])

        rows.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": ground_truth or answer,  # reference 없으면 답변으로 대체(일부 지표 한정)
        })
        logger.info("[%d/%d] 평가행 생성 완료: %r (hits=%d)", i, len(dataset), question, len(hits))
    return rows


def _run_ragas(rows: list[dict]):
    """RAGAs 평가 실행. 미설치 시 친절히 안내하고 종료."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
    except ImportError as e:
        logger.error(
            "RAGAs 미설치. 평가 전용 의존성을 설치하세요:\n"
            "    pip install ragas datasets\n"
            "(서버 런타임에는 불필요 — 오프라인 평가 전용)\n"
            f"원본 에러: {e}"
        )
        sys.exit(1)

    # RAGAs 버전별 컬럼명 차이를 흡수: 신버전은 user_input/response/...,
    # 구버전은 question/answer/contexts/ground_truth 를 쓴다. 양쪽 키를 모두 채운다.
    enriched = []
    for r in rows:
        enriched.append({
            **r,
            "question": r["user_input"],
            "answer": r["response"],
            "contexts": r["retrieved_contexts"],
            "ground_truth": r["reference"],
        })

    ds = Dataset.from_list(enriched)
    logger.info("RAGAs 평가 시작 (%d 행)...", len(enriched))
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAs 기반 RAG 정량 평가(오프라인)")
    parser.add_argument("--dataset", default=_SAMPLE_PATH,
                        help=f"평가셋 JSON 경로 (기본 {_SAMPLE_PATH})")
    parser.add_argument("--make-sample", action="store_true",
                        help="샘플 평가셋 템플릿을 생성하고 종료")
    parser.add_argument("--out", default="ragas_result.json",
                        help="평가 결과 저장 경로(JSON)")
    # ── [CI 게이트] ────────────────────────────────────────────────────
    parser.add_argument("--ci", action="store_true",
                        help="CI 모드: 임계값 미달 시 exit(1) 로 빌드를 실패시킨다.")
    parser.add_argument("--threshold-faithfulness", type=float, default=0.0,
                        help="Faithfulness 최소 기준(미달 시 CI 실패). 0이면 미검사.")
    parser.add_argument("--threshold-answer-relevancy", type=float, default=0.0,
                        help="Answer Relevancy 최소 기준. 0이면 미검사.")
    parser.add_argument("--threshold-context-precision", type=float, default=0.0,
                        help="Context Precision 최소 기준. 0이면 미검사.")
    args = parser.parse_args()

    if args.make_sample:
        _make_sample()
        return

    dataset = _load_dataset(args.dataset)
    rows = asyncio.run(_build_eval_rows(dataset))
    if not rows:
        logger.error("유효한 평가행이 없습니다(질문이 모두 비어 있음).")
        sys.exit(1)

    result = _run_ragas(rows)

    # 결과 출력 + 저장
    print("\n================ RAGAs 평가 결과 ================")
    print(result)
    try:
        df = result.to_pandas()
        df.to_json(args.out, orient="records", force_ascii=False, indent=2)
        logger.info("행별 결과 저장: %s", args.out)
    except Exception:
        logger.info("행별 결과 저장 생략(요약 점수는 위 출력 참조)")

    # ── [CI 게이트] 임계값 검사 ─────────────────────────────────────────
    if args.ci:
        _enforce_thresholds(result, {
            "faithfulness": args.threshold_faithfulness,
            "answer_relevancy": args.threshold_answer_relevancy,
            "context_precision": args.threshold_context_precision,
        })


def _enforce_thresholds(result, thresholds: dict) -> None:
    """RAGAs 결과 요약 점수를 임계값과 비교해 미달 시 exit(1).

    result 에서 지표별 평균 점수를 안전하게 추출한다(RAGAs 버전별 접근 차이 흡수).
    threshold 가 0 이하인 지표는 검사하지 않는다(미설정으로 간주).
    """
    scores = _extract_summary_scores(result)
    logger.info("CI 게이트 요약 점수: %s", scores)

    failures = []
    for metric, minimum in thresholds.items():
        if minimum and minimum > 0:
            actual = scores.get(metric)
            if actual is None:
                logger.warning("지표 '%s' 점수를 찾지 못해 검사 생략", metric)
                continue
            if actual < minimum:
                failures.append(f"{metric}={actual:.3f} < 기준 {minimum:.3f}")

    if failures:
        print("\n❌ RAG 품질 게이트 실패:")
        for f in failures:
            print(f"   - {f}")
        sys.exit(1)
    print("\n✅ RAG 품질 게이트 통과")
    sys.exit(0)


def _extract_summary_scores(result) -> dict:
    """RAGAs 결과 객체에서 지표별 평균 점수 dict 를 안전 추출.

    버전에 따라 result 가 dict 처럼 동작하거나 to_pandas() 로 접근해야 한다.
    두 경로를 모두 시도해 {metric: mean_score} 를 만든다.
    """
    metrics = ("faithfulness", "answer_relevancy", "context_precision")
    scores: dict = {}

    # 경로 1: dict 스타일 접근(result["faithfulness"])
    for m in metrics:
        try:
            val = result[m]
            if isinstance(val, (int, float)):
                scores[m] = float(val)
        except Exception:
            pass

    # 경로 2: to_pandas() 평균(경로 1이 비었을 때 보완)
    if not scores:
        try:
            df = result.to_pandas()
            for m in metrics:
                if m in df.columns:
                    scores[m] = float(df[m].mean())
        except Exception:
            pass
    return scores


if __name__ == "__main__":
    main()