"""
scripts/test_semantic_offline.py
외부 서비스(OpenAI/Oracle/Chroma 서버) 없이 SEMANTIC 파이프라인 배선을 검증하는 오프라인 테스트.

무엇을 검증하나:
  - search_similar → build_product_context → handle_semantic 배선
  - 임베딩 차원 일관성, cosine 최근접 정렬, 메타데이터 컨텍스트 생성
  - 빈 컬렉션 시 고정 안내 문구 경로
  - "[추천]" 프리픽스 반환 계약

무엇을 검증하지 못하나(실 통합 테스트 test_semantic.py 로 확인):
  - 실제 GPT 추천 품질 / 무관 질문 거절 동작
  - 실제 OpenAI 임베딩 의미 정확도

실행(프로젝트 루트):  python -m scripts.test_semantic_offline
"""
import asyncio
import math
import chromadb

import services.chroma_service as chroma_service
import services.embed_service as embed_service
import services.rag_service as rag_service
import pipeline.semantic_handler as semantic_handler

# ---------------- 1) 결정적 가짜 임베더 (키워드 기반) ----------------
# 의미적으로 겹치는 텍스트가 더 가까운 벡터를 갖도록 키워드를 차원에 매핑한다.
_VOCAB = ["겨울", "따뜻", "코트", "외투", "패딩",
          "운동", "신발", "가벼운", "러닝",
          "향수", "선물", "향기",
          "비", "방수", "우산"]
_BASELINE_DIM = len(_VOCAB)   # 마지막에 항상 0.1 베이스라인 차원 추가(영벡터 방지)


def _fake_vector(text: str) -> list[float]:
    vec = [0.0] * (len(_VOCAB) + 1)
    t = text or ""
    for i, kw in enumerate(_VOCAB):
        if kw in t:
            vec[i] = 1.0
    vec[_BASELINE_DIM] = 0.1                      # 영벡터 방지(키워드 0개여도 cosine 계산 가능)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]               # L2 정규화(실제 OpenAI 임베딩과 동일 성격)


async def _fake_get_embedding(text: str) -> list[float]:
    return _fake_vector(text)


async def _fake_get_embeddings(texts: list[str]) -> list[list[float]]:
    return [_fake_vector(t) for t in texts]


# GPT 호출만 가짜로(컨텍스트 빌드는 실제 함수 사용 → 메타데이터 매핑까지 검증)
async def _fake_generate_rag_response(question: str, hits: list[dict]) -> str:
    context = rag_service.build_product_context(hits)   # 실제 컨텍스트 빌더 호출
    top_name = (hits[0].get("metadata", {}) or {}).get("product_name", "?")
    # context 를 실제로 소비해 빌더의 메타데이터 매핑까지 함께 검증한다.
    return f"(가짜GPT) 최우선 추천: {top_name} | 컨텍스트길이={len(context)} | 컨텍스트상품수={len(hits)}"


def _patch():
    # 임베딩: 가짜로 교체
    embed_service.get_embedding = _fake_get_embedding
    embed_service.get_embeddings = _fake_get_embeddings
    # RAG GPT 호출: 가짜로 교체
    rag_service.generate_rag_response = _fake_generate_rag_response
    # ChromaDB: 인메모리 EphemeralClient 로 교체 + 싱글톤 초기화
    ephemeral = chromadb.EphemeralClient()
    chroma_service._client = ephemeral
    chroma_service._collection = None   # cosine 컬렉션을 인메모리에 새로 생성하도록 리셋
    chroma_service._get_client = lambda: ephemeral


# 가짜 상품(샘플을 모사)
_FAKE_PRODUCTS = [
    {"id": "1", "name": "구스다운 롱패딩 코트", "category": "아우터",
     "price": 189000.0, "desc": "한겨울에도 따뜻한 구스다운 충전재의 롱 패딩 외투"},
    {"id": "2", "name": "초경량 러닝화", "category": "신발",
     "price": 89000.0, "desc": "장거리 운동에 좋은 가벼운 러닝 신발"},
    {"id": "3", "name": "플로럴 오 드 퍼퓸", "category": "뷰티",
     "price": 75000.0, "desc": "선물용으로 좋은 은은한 향기의 향수"},
    {"id": "4", "name": "3단 자동 우산", "category": "잡화",
     "price": 22000.0, "desc": "비 오는 날 강한 방수 기능의 우산"},
]


async def _seed():
    ids, embeds, docs, metas = [], [], [], []
    for p in _FAKE_PRODUCTS:
        ids.append(p["id"])
        embeds.append(_fake_vector(p["desc"]))
        docs.append(p["desc"])
        metas.append({
            "product_id": int(p["id"]),
            "product_name": p["name"],
            "category": p["category"],
            "price": p["price"],
            "description": p["desc"],
            "stock": 10,
        })
    total = await chroma_service.upsert_products(ids, embeds, docs, metas)
    return total


async def main():
    _patch()

    # A) 빈 컬렉션 경로 먼저 확인 (seed 전)
    empty_answer = await semantic_handler.handle_semantic("아무거나", {"intent": "SEMANTIC_SEARCH"})
    assert empty_answer.startswith("[추천]"), "프리픽스 누락"
    assert "찾지 못했" in empty_answer, "빈 컬렉션 안내 문구 불일치"
    print("[PASS] 빈 컬렉션 → 고정 안내:", empty_answer)

    # B) seed 후 검색 정확도/배선 확인
    total = await _seed()
    cnt = await chroma_service.count()
    assert total == cnt == len(_FAKE_PRODUCTS), f"건수 불일치 total={total} cnt={cnt}"
    print(f"[PASS] 인덱싱 건수 = {cnt}")

    cases = [
        ("겨울에 따뜻한 코트 외투 추천", "구스다운 롱패딩 코트"),
        ("운동용 가벼운 러닝 신발", "초경량 러닝화"),
        ("선물용 향수 향기", "플로럴 오 드 퍼퓸"),
        ("비 오는 날 방수 우산", "3단 자동 우산"),
    ]
    for q, expected_top in cases:
        # 핸들러 전체 경로
        ans = await semantic_handler.handle_semantic(q, {"intent": "SEMANTIC_SEARCH"})
        assert ans.startswith("[추천]"), f"프리픽스 누락: {ans}"
        # 검색 단독 최근접 검증
        qvec = _fake_vector(q)
        hits = await chroma_service.search_similar(qvec, n_results=4)
        assert len(hits) == 4, f"top_k 결과 수 불일치: {len(hits)}"
        # distance 오름차순(가까운 순) 정렬 확인
        dists = [h["distance"] for h in hits]
        assert dists == sorted(dists), f"distance 정렬 안됨: {dists}"
        top_name = hits[0]["metadata"]["product_name"]
        assert top_name == expected_top, f"최근접 기대={expected_top} 실제={top_name}"
        # 컨텍스트 빌드 검증(가격 포맷/필드 매핑)
        ctx = rag_service.build_product_context(hits)
        assert "원" in ctx and "상품명" in ctx and "카테고리" in ctx, "컨텍스트 필드 누락"
        print(f"[PASS] q={q!r} → top={top_name!r}, dist={dists[0]:.4f} | {ans}")

    # C) 가격 포맷 단독 검증 (Decimal→float 가정, 천단위 콤마)
    sample_ctx = rag_service.build_product_context([{
        "document": "샘플 설명",
        "metadata": {"product_name": "테스트", "category": "테스트", "price": 189000.0},
    }])
    assert "189,000원" in sample_ctx, f"가격 포맷 오류:\n{sample_ctx}"
    print("[PASS] 가격 천단위 포맷: 189,000원")

    print("\n✅ 오프라인 테스트 전부 통과")


if __name__ == "__main__":
    asyncio.run(main())
