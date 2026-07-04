# PHASE 3-② 성능/비용 비교 분석

> 상태: **측정 인프라 완료 / 실측 데이터는 배포 후 채움**
> 단순 기술 벤치마크가 아닌 **"비즈니스 관점 지표"** 를 1차 기준으로 설계.

---

## 0. 비교군 (5개 provider × 모델)

| # | Provider | 모델 | 포지션 | 비고 |
|---|----------|------|--------|------|
| A | OpenAI | gpt-5.4 | 고성능 기준점 | 품질 상한선 |
| B | OpenAI | gpt-5.4-mini | 저비용 기준점 | 비용 하한선 (OpenAI) |
| C | Google | gemini-3.1-flash | 초저비용 | 쇼핑몰 CS 챗봇 추천 모델 |
| D | Anthropic | claude-sonnet-4-6 | 고성능 · 지시 준수 | 문체·정확도 비교 |
| E | DeepSeek | deepseek-v4-flash | 와일드카드 | 비용 극단 비교 (중국 오픈소스) |

---

## 1. 핵심 비즈니스 지표 정의

> 단순 품질 점수(BLEU, ROUGE)가 아닌 **"이 모델을 운영하면 실제로 어떤 비즈니스 가치가 나오는가"**
> 를 수치화한다.

### 1-1. CPQ (Cost Per Query) — 문의 1건당 비용 (USD)

```
기준 요청 토큰 가정 (SEMANTIC_SEARCH 기준):
  - Intent 분류:  500 input + 50 output  tokens
  - RAG 응답:   2,000 input + 300 output tokens
  - 합계:       2,500 input + 350 output tokens

CPQ = (2500 × input_rate/1000) + (350 × output_rate/1000)
```

| 모델 | Input ($/1K) | Output ($/1K) | CPQ (USD) |
|------|-------------|--------------|-----------|
| gpt-5.4 | 0.005 | 0.015 | **$0.01775** |
| gpt-5.4-mini | 0.0006 | 0.0024 | **$0.00234** |
| gemini-3.1-flash | 0.000075 | 0.0003 | **$0.000293** |
| claude-sonnet-4-6 | 0.003 | 0.015 | **$0.01275** |
| deepseek-v4-flash | 0.00014 | 0.00028 | **$0.000448** |

---

### 1-2. QPD (Queries Per Dollar) — $1당 처리 가능 문의 건수

```
QPD = 1 / CPQ
```

| 모델 | QPD (건/$) | gpt-5.4 대비 |
|------|-----------|------------|
| gpt-5.4 | **56건** | 기준 (1×) |
| gpt-5.4-mini | **427건** | **7.6×** |
| gemini-3.1-flash | **3,413건** | **60.9×** |
| claude-sonnet-4-6 | **78건** | 1.4× |
| deepseek-v4-flash | **2,232건** | **39.9×** |

> **발표 포인트**: "gemini-3.1-flash 는 동일 예산으로 gpt-5.4 대비 60배 이상의 문의를
> 처리할 수 있습니다. 단, 이 수치만으로는 품질 차이를 알 수 없으므로
> 아래 CX-ROI 와 함께 봐야 합니다."

---

### 1-3. 월 예상 운영비 — 규모별 시뮬레이션

```
월 운영비 = 월 문의 건수 × CPQ
```

| 규모 | 월 문의 | gpt-5.4 | gpt-5.4-mini | gemini-3.1-flash | claude-sonnet | deepseek |
|------|--------|---------|-------------|-----------------|---------------|---------|
| 소형 | 1천건 | $17.75 | $2.34 | $0.29 | $12.75 | $0.45 |
| 중형 | 1만건 | $177.5 | $23.4 | **$2.93** | $127.5 | **$4.48** |
| 대형 | 10만건 | $1,775 | $234 | **$29.3** | $1,275 | **$44.8** |

> **발표 포인트**: "월 1만 건 기준, gpt-5.4 는 약 $178/월이지만
> gemini-3.1-flash 는 단 $3/월입니다. 품질이 충분하다면
> 운영비 차이가 60배에 달합니다."

---

### 1-4. CX-ROI (Customer Experience Return on Investment)

```
CX-ROI = 평균 품질점수 (1~5) × QPD

"1달러를 썼을 때 고객이 느끼는 만족도의 총합"
→ 비용 효율성과 품질을 동시에 고려한 복합 지표
```

| 모델 | 품질점수 (실측 예정) | QPD | CX-ROI | 해석 |
|------|------------------|----|--------|------|
| gpt-5.4 | — | 56 | — | 기준 |
| gpt-5.4-mini | — | 427 | — | |
| gemini-3.1-flash | — | 3,413 | — | |
| claude-sonnet-4-6 | — | 78 | — | |
| deepseek-v4-flash | — | 2,232 | — | |

> CX-ROI 가 높을수록 "품질 대비 가성비가 좋은 모델".
> 품질점수가 gpt-5.4 의 80%여도 QPD 가 60배면 CX-ROI 가 48배 높을 수 있음.

---

### 1-5. RCI (Reliability-Cost Index) — 신뢰성 × 비용 효율

```
RCI = (1 - 환각률) × QPD

"$1당 신뢰할 수 있는 정확한 답변 건수"
→ 환각이 잦은 저비용 모델의 실질 가치를 보정하는 지표
```

| 모델 | 환각률 (실측 예정) | QPD | RCI |
|------|-----------------|-----|-----|
| gpt-5.4 | — | 56 | — |
| gpt-5.4-mini | — | 427 | — |
| gemini-3.1-flash | — | 3,413 | — |
| claude-sonnet-4-6 | — | 78 | — |
| deepseek-v4-flash | — | 2,232 | — |

> **발표 포인트**: "DeepSeek 는 QPD 가 높지만 복잡한 Agent 작업에서
> 환각·tool-calling 불안정 이슈가 보고됩니다.
> RCI 가 이 차이를 정량화해줍니다."

---

### 1-6. SEI (Speed-Economy Index) — 속도 × 비용 효율

```
SEI = QPD / 평균_레이턴시(ms)

"$1당, 1ms당 처리 가능 건수" — 빠르고 저렴할수록 높음
```

| 모델 | 평균 레이턴시(ms) | QPD | SEI |
|------|----------------|-----|-----|
| gpt-5.4 | — | 56 | — |
| gpt-5.4-mini | — | 427 | — |
| gemini-3.1-flash | — | 3,413 | — |
| claude-sonnet-4-6 | — | 78 | — |
| deepseek-v4-flash | — | 2,232 | — |

---

## 2. 기술 지표 (2차)

비즈니스 지표의 근거가 되는 기술 수치.

### 2-1. 품질 측정 방법 — LLM-as-a-Judge

**Judge 모델**: gpt-5.4-mini (비용 절감 + 일관성)

**평가 프롬프트**:
```
당신은 쇼핑몰 AI 챗봇 응답의 품질 평가자입니다.
아래 사용자 질문과 AI 응답을 보고 각 항목을 1~5점으로 평가하세요.

[질문]: {question}
[응답]: {answer}
[참고 상품 정보]: {context}

평가 항목:
1. 정확성 (1-5): 제공된 상품 정보와 일치하는가?
2. 자연스러움 (1-5): 한국어가 자연스럽고 친절한가?
3. 인텐트 반영 (1-5): 사용자 의도에 맞는 답변인가?
4. 환각 없음 (0/1): 근거 없는 정보 생성이 없는가?

JSON 형식으로만 응답: {"accuracy": N, "fluency": N, "intent": N, "no_hallucination": N}
```

**최종 품질점수** = (accuracy + fluency + intent) / 3 × no_hallucination

---

### 2-2. 인텐트 분류 정확도

동일 테스트셋 50개 질문에 대한 인텐트 분류 결과를 정답과 비교.

```
분류 정확도 = 올바르게 분류된 건수 / 50
```

| 모델 | 정확도 | 오분류 패턴 |
|------|--------|-----------|
| gpt-5.4 | — | — |
| gpt-5.4-mini | — | — |
| gemini-3.1-flash | — | — (temperature override 주의) |
| claude-sonnet-4-6 | — | — |
| deepseek-v4-flash | — | — |

---

### 2-3. 경로별 레이턴시 & 토큰 소모

| 경로 | 모델 | 평균 레이턴시(ms) | prompt_tokens | completion_tokens | total_tokens |
|------|------|----------------|-------------|-----------------|-------------|
| router | gpt-5.4 | — | (LangSmith) | (LangSmith) | (LangSmith) |
| single_agent | gpt-5.4 | — | — | — | — |
| single_agent | gemini-3.1-flash | — | — | — | — |
| single_agent | claude-sonnet-4-6 | — | — | — | — |
| single_agent | deepseek-v4-flash | — | — | — | — |
| multi_agent | gpt-5.4 | — | — | — | — |

---

## 3. 테스트 질문셋 (50개)

| # | 질문 | 기대 인텐트 | 기대 답변 핵심 |
|---|------|-----------|--------------|
| Q01 | "5만원 이하 청바지 추천해줘" | STRUCTURED_QUERY | 해당 조건 상품 목록 |
| Q02 | "여름에 시원하게 입을 수 있는 옷" | SEMANTIC_SEARCH | 린넨/통기성 상품 |
| Q03 | "교환 어떻게 해?" | FAQ | 교환 절차 안내 |
| Q04 | "내 주문 어디까지 왔어?" | ORDER_INQUIRY | 주문 현황 |
| Q05 | "사이즈가 안 맞아서 너무 화가 나" | COMPLAINT | 공감 응대 |
| Q06 | "안녕!" | SMALL_TALK | 인사 응대 |
| Q07 | "30만원 이하 전자기기 있어?" | STRUCTURED_QUERY | 조건 검색 |
| Q08 | "피부에 좋은 크림 뭐 있어?" | SEMANTIC_SEARCH | 뷰티 상품 RAG |
| Q09 | "배송비 얼마야?" | FAQ | 배송비 정책 |
| Q10 | "주문번호 12345 취소하고 싶어" | ORDER_INQUIRY | 취소 안내 |
| … | (50개 전체는 측정 시 작성) | | |

---

## 4. 측정 절차 (배포 후)

### Step 1. 환경 설정

```bash
# .env 수정
METRICS_ENABLED=true
METRICS_LOG_PATH=metrics.jsonl
LANGSMITH_TRACING=true

# 단가 실측값으로 갱신 (pricing.py 기본값이 자리표시라 실단가 입력 권장)
PRICE_OPENAI_GPT_5_4_INPUT=0.005
PRICE_OPENAI_GPT_5_4_OUTPUT=0.015
PRICE_DEEPSEEK_DEEPSEEK_V4_FLASH_INPUT=0.00014
PRICE_DEEPSEEK_DEEPSEEK_V4_FLASH_OUTPUT=0.00028
```

### Step 2. 모델별 측정

```bash
# provider 를 하나씩 바꿔가며 동일 질문셋 50개 실행
# /chat/agent 엔드포인트로 측정 (토큰 측정 가능한 경로)

for provider in openai gemini anthropic deepseek; do
  # .env 의 LLM_PROVIDER 변경 후 FastAPI 재시작
  sed -i "s/LLM_PROVIDER=.*/LLM_PROVIDER=$provider/" .env
  pkill -f uvicorn && nohup uvicorn main:app ... &
  sleep 5
  python scripts/run_benchmark.py --provider $provider
done
```

### Step 3. 집계 스크립트

```python
import json, statistics
from collections import defaultdict

rows = [json.loads(l) for l in open("metrics.jsonl", encoding="utf-8")]
by = defaultdict(list)
for r in rows:
    by[(r["route"], r["provider"])].append(r)

print(f"{'경로':15} {'provider':10} {'N':>3} {'latency':>10} {'tokens':>8} {'CPQ':>10} {'QPD':>8}")
print("-" * 75)
for (route, provider), items in sorted(by.items()):
    lat  = statistics.mean(x["latency_ms"]  for x in items)
    tok  = statistics.mean(x["total_tokens"] for x in items)
    cost = statistics.mean(x["cost_usd"]     for x in items)
    qpd  = round(1 / cost) if cost > 0 else 0
    print(f"{route:15} {provider:10} {len(items):>3} {lat:>8.0f}ms {tok:>8.0f} ${cost:>8.5f} {qpd:>8}건/$")
```

### Step 4. 비즈니스 지표 계산

```python
# LLM-as-a-Judge 품질 평가 후 CX-ROI, RCI 계산
import openai

def judge_response(question, answer, context, judge_model="gpt-5.4-mini"):
    prompt = f"""...(위 평가 프롬프트)..."""
    resp = openai.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    scores = json.loads(resp.choices[0].message.content)
    quality = (scores["accuracy"] + scores["fluency"] + scores["intent"]) / 3
    return quality * scores["no_hallucination"]  # 환각 있으면 0점

# 모델별 평균 품질점수 × QPD = CX-ROI
for provider, score in quality_scores.items():
    qpd = 1 / cpq[provider]
    print(f"{provider}: CX-ROI = {score:.2f} × {qpd:.0f} = {score * qpd:.0f}")
```

---

## 5. 최종 결과 리포트 템플릿

### 5-1. 비즈니스 지표 종합

| 지표 | gpt-5.4 | gpt-5.4-mini | gemini-3.1-flash | claude-sonnet | deepseek-v4-flash |
|------|---------|-------------|-----------------|--------------|-----------------|
| CPQ (USD) | $0.01775 | $0.00234 | **$0.000293** | $0.01275 | **$0.000448** |
| QPD (건/$) | 56 | 427 | **3,413** | 78 | **2,232** |
| 품질점수 (1-5) | — | — | — | — | — |
| CX-ROI | — | — | — | — | — |
| 환각률 (%) | — | — | — | — | — |
| RCI | — | — | — | — | — |
| 평균 레이턴시 | — | — | — | — | — |
| SEI | — | — | — | — | — |
| **월비용 (1만건)** | $177.5 | $23.4 | **$2.93** | $127.5 | **$4.48** |

### 5-2. 추천 시나리오

| 시나리오 | 추천 모델 | 근거 |
|---------|---------|------|
| 품질 최우선 (B2B·고가 상품) | gpt-5.4 | 품질점수 최고, 환각 최소 |
| 비용·품질 균형 | gpt-5.4-mini or deepseek-v4-flash | CX-ROI 분석 후 결정 |
| 비용 극단 절감 | gemini-3.1-flash | QPD 3,413건/$, 월 $2.93 |
| 지시 준수 중요 | claude-sonnet-4-6 | 시스템 프롬프트 준수율 최고 |
| 와일드카드 | deepseek-v4-flash | Agent 안정성 확인 후 판단 |

---

## 6. 알려진 한계 / 주의

- **CPQ 계산의 가정**: 실제 요청의 토큰 분포는 인텐트에 따라 다름.
  STRUCTURED_QUERY(토큰 적음) vs SEMANTIC_SEARCH(RAG 컨텍스트로 토큰 많음).
  인텐트별 실측 분포를 metrics.jsonl 에서 뽑아 재보정 필요.

- **Gemini 3+ temperature override**: 인텐트 분류(temperature=0.0 의도)에서
  langchain 이 1.0 으로 override → 분류 일관성에 영향 가능. 별도 확인 필요.

- **DeepSeek Agent 안정성**: multi_agent 경로에서 tool-calling 불안정 보고 있음.
  단일 RAG 응답(router_pipeline, single_agent)에서는 문제없지만
  multi_agent 경로 비교 시 불안정 건수를 별도 집계할 것.

- **단가표는 2026-06 기준 자리표시값**: 배포 시 각 provider 공식 가격표로
  `.env PRICE_*` 오버라이드 갱신 필수.

- **라우터 그래프 토큰 미측정**: `/chat/ask` 경로는 토큰이 0으로 기록됨.
  토큰/비용 비교는 `/chat/agent` 경로 기준으로 진행하고
  라우터 경로는 LangSmith 로 보완.
