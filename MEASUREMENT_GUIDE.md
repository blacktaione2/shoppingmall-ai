# 측정 가이드 — 배포 후 실측 ①~⑨

> 대상: PHASE3_BENCHMARK.md 의 지표 체계 + 배포 직전 보강분(동시성/멱등성).
> 모든 스크립트는 서버에서 실행한다(localhost 호출 — 네트워크 노이즈 제거).
> 사전 준비: `pip install requests` (스크립트 의존성, 1회)

## 공통 .env 설정 (측정 시작 전 1회)

```bash
METRICS_ENABLED=true
METRICS_LOG_PATH=metrics.jsonl
LANGSMITH_TRACING=true
# pricing 실단가 (PHASE3_BENCHMARK.md 0장 표 참고)
PRICE_OPENAI_GPT_5_4_INPUT=0.005
PRICE_OPENAI_GPT_5_4_OUTPUT=0.015
# ... provider 별로 입력 (pricing.py 의 env 키 규칙 참고)
```

---

## ① Provider 5종 비교 (CPQ/QPD/CX-ROI/RCI/SEI + 품질/정확도)

```bash
# provider 마다 반복: .env 의 LLM_PROVIDER 변경 → 재기동 → 러너 실행
sed -i "s/^LLM_PROVIDER=.*/LLM_PROVIDER=openai/" .env
pkill -f "uvicorn main:app"; nohup uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 > uvicorn.log 2>&1 &
sleep 8

# 라우터 경로(인텐트 정확도 + 레이턴시) — 게스트
python scripts/run_benchmark.py --label openai_router
# 토큰/비용 측정 경로 — Agent
python scripts/run_benchmark.py --label openai_agent --endpoint /chat/agent
# 로그인 경로 포함(ORDER_INQUIRY 실동작)이 필요하면: --chat-token <로그인 후 발급 UUID>
```

- 서버 지표 집계: `python scripts/aggregate_metrics.py --csv provider_metrics.csv`
  → 경로×provider 별 avg/p50/p95 레이턴시, 평균 토큰, CPQ, QPD (⑩ p95 도 여기서 나옴)
- 품질 채점: `python scripts/judge_quality.py benchmark_results_openai_agent.jsonl --qpd 56 --out judged_openai.jsonl`
  → 평균 품질점수, 환각률, CX-ROI, RCI (QPD 값은 PHASE3_BENCHMARK.md 1-2 표)
- 결과 기입처: PHASE3_BENCHMARK.md 의 `—` 칸들

## ② RAGAs CI (GitHub Actions 품질 게이트)

1. 확장된 평가셋(18개)이 이 커밋에 포함 — push 후 RAG 파일을 건드리는 PR 생성
2. Actions 로그에서 faithfulness / answer_relevancy / context_precision 점수 캡처
3. **차단 증명 컷**: rag_pipeline.py 를 일부러 열화(예: 검색 결과를 1건으로 제한)한 PR
   → 게이트 FAIL 스크린샷 → PR 닫기
4. threshold 는 정상 점수 실측 후 `-0.05` 여유로 rag_quality.yml 에 설정

## ③ LangSmith 노드별 분해

측정 ① 실행이 곧 데이터 수집. LangSmith 콘솔에서 대표 트레이스 1건을 열어
"인텐트 분류 N토큰 → RAG 응답 M토큰" 노드별 분해 스크린샷 확보.

## ④ 하이브리드 검색 Ablation

동일 평가셋으로 4개 구성 반복 (구성 변경마다 재기동):

| 구성 | BM25_ENABLED | RERANK_ENABLED |
|---|---|---|
| A. Dense only | false | false |
| B. +BM25(RRF) | true | false |
| C. +Cohere 재랭킹 | false | true |
| D. 풀 구성 | true | true |

```bash
python scripts/evaluate_rag.py --dataset scripts/eval_dataset_prod.json --out ragas_A.json
```
→ 구성별 faithfulness/answer_relevancy/context_precision 표로 정리.
"RRF 채택·재랭킹 도입이 각각 몇 %p 기여했는가"가 산출물.

## ⑤ 개인화 A/B

1. 테스트 계정으로 특정 카테고리(예: 아우터) 상품 2~3건 주문
2. `PERSONALIZATION_ENABLED=false` 재기동 → "추천 상품 보여줘" 등 5개 질문 결과 저장
3. `true` 재기동 → 동일 질문 → 순위 변화 비교 (구매 카테고리가 상위로 오는지)
4. 보너스: 주문 1건을 관리자에서 CANCELLED 로 바꾼 뒤 재측정
   → "취소 주문 제외 수정(#3)"의 효과 실증

## ⑥ 환각 가드 실측

- 1차 데이터: judge_quality.py 출력의 **halluc_bait 부분집합 환각률**(가격 민감 6문항)
- 2차 데이터: LangSmith 에서 halluc_bait 트레이스를 열어
  가드 통과 / mini 재시도 / 안전문구 대체 각 단계 건수 집계
- 문서화: "가격 민감 질문 N건 중 최종 환각 노출 M건, 가드 재시도 성공 K건"

## ⑦ 재고 락 부하테스트

```bash
# 관리자 화면에서 테스트 상품 재고를 10으로 설정 후
python scripts/load_test_orders.py --login-id <계정> --password <비번> \
    --product-id <ID> --stock 10 --concurrency 50
```
- 기대: `성공 10 / 거절 40 / 오류 0` + "초과판매 0건" 판정 + 락 대기 p95
- 사후: 관리자 화면에서 재고 0 + SOLD_OUT 전환, 챗봇에서 해당 상품 품절 안내 확인
  (색인 동기화까지 한 번에 검증됨)
- 대조군(선택): v13 jar 로 같은 테스트 → 초과판매 재현 수치가 있으면 발표 임팩트 최대

## ⑧ 멱등성 실측

```bash
# 같은 request_id 2회 → 두 번째가 409 여야 함
RID=$(python3 -c "import uuid; print(uuid.uuid4())")
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8000/chat/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"안녕\",\"request_id\":\"$RID\"}"        # 200
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8000/chat/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"안녕\",\"request_id\":\"$RID\"}"        # 409

# 폴백 검증: redis 중지 → 같은 실험 → 둘 다 200 (통과 모드) → redis 재기동
sudo systemctl stop redis && (위 2회 반복) && sudo systemctl start redis
```
- 문서화: "중복 차단율 100%, Redis 장애 시 가용성 우선 폴백 동작 확인"

## ⑨ Redis 체크포인터 생존성

1. 로그인 상태로 챗봇에서 "겨울 코트 추천해줘" → "그 중에 제일 싼 건?" (멀티턴 성립 확인)
2. `pkill -f uvicorn` → 재기동
3. 같은 창에서 "방금 말한 거 얼마라고?" → **맥락 유지되면 성공**
4. 대조군: `.env` 에서 REDIS_URL 제거(MemorySaver) 후 동일 절차 → 맥락 소실 확인
5. 오버헤드: aggregate_metrics 의 레이턴시를 Redis on/off 로 비교 (체크포인트 ms)

---

## 결과 기입처 요약

| 측정 | 기입 문서 |
|---|---|
| ①③⑩ | PHASE3_BENCHMARK.md 의 `—` 칸 + /admin/metrics 대시보드 스크린샷 |
| ② | PRESENTATION_SCRIPT.md 10장 + Actions 스크린샷 |
| ④⑤⑥ | PHASE3_BENCHMARK.md 신규 절 또는 발표 자료 |
| ⑦⑧⑨ | PRESENTATION_SCRIPT.md 8-1장 (설계 서사에 실측 수치 부착) |
