"""
tests/test_chunking_service.py
services/chunking_service.py 단위 테스트 (오프라인, DB/외부 API 호출 없음).

[검증 시나리오]
1. 임계값 이하 설명 → 청킹 없이 문서 1개(기존 인덱싱과 100% 동일한 데이터)
2. 임계값 초과 설명 → 문서 N개, id/프리픽스/청크 메타 형식 확인
3. 청크 메타(chunk_index/chunk_count/chunk_text_raw)가 정확한지
4. 프리픽스에 상품명/카테고리 누락 시 "(이름 없음)"/"(카테고리 없음)" 폴백
5. _embed_text_for(): description 우선 → product_name+category 폴백 →
   image_caption 이어붙이기
6. _chunk_threshold_chars()/_chunk_overlap_chars(): .env 파싱, 잘못된 값/0 이하 폴백
7. [회귀] threshold < overlap(기본 50) 조합에서도 예외 없이 청킹돼야 함
"""
import pytest

from services import chunking_service


def _row(**overrides) -> dict:
    base = {
        "product_id": 1,
        "product_name": "테스트 상품",
        "category": "상의",
        "price": 29000,
        "description": "짧은 설명입니다.",
        "stock": 10,
        "status": "ACTIVE",
        "image_url": "",
    }
    base.update(overrides)
    return base


# ────────────────────────────────────────────────────────────────────────
# 1) 임계값 이하 → 청킹 없이 문서 1개, 기존 형식과 동일
# ────────────────────────────────────────────────────────────────────────
def test_short_description_no_chunking():
    row = _row(description="짧은 설명입니다.")
    docs = chunking_service.build_chunk_documents(row)

    assert len(docs) == 1
    doc = docs[0]
    assert doc["id"] == "1"                      # 청크 접미사 없음
    assert doc["document"] == "짧은 설명입니다."   # 프리픽스 없음
    # 청크 부가 필드가 없어야 기존 인덱싱과 100% 동일
    assert "chunk_index" not in doc["metadata"]
    assert "chunk_count" not in doc["metadata"]
    assert "chunk_text_raw" not in doc["metadata"]
    assert doc["metadata"]["product_id"] == 1
    assert doc["metadata"]["price"] == 29000.0


# ────────────────────────────────────────────────────────────────────────
# 2) 임계값 초과 → 문서 N개, id 형식, 프리픽스 형식
# ────────────────────────────────────────────────────────────────────────
def test_long_description_chunks_with_prefix(monkeypatch):
    monkeypatch.setenv("CHUNK_THRESHOLD_CHARS", "50")
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", "10")

    long_desc = "가나다라마바사아자차카타파하. " * 20  # 50자 훨씬 초과
    row = _row(description=long_desc)

    docs = chunking_service.build_chunk_documents(row)

    assert len(docs) > 1
    prefix = "[테스트 상품 | 상의] "
    for idx, doc in enumerate(docs):
        assert doc["id"] == f"1_chunk_{idx}"
        assert doc["document"].startswith(prefix)
        # 프리픽스 없는 순수 원문은 metadata.chunk_text_raw 에만 있음
        assert prefix not in doc["metadata"]["chunk_text_raw"]
        assert doc["metadata"]["chunk_index"] == idx
        assert doc["metadata"]["chunk_count"] == len(docs)
        # 원본 상품 메타는 청크 전부에 동일하게 복사됨
        assert doc["metadata"]["product_id"] == 1
        assert doc["metadata"]["category"] == "상의"


# ────────────────────────────────────────────────────────────────────────
# 3) 상품명/카테고리 누락 시 프리픽스 폴백 문구
# ────────────────────────────────────────────────────────────────────────
def test_prefix_fallback_when_name_category_missing(monkeypatch):
    monkeypatch.setenv("CHUNK_THRESHOLD_CHARS", "100")
    row = _row(product_name="", category="", description="긴 설명 " * 30)

    docs = chunking_service.build_chunk_documents(row)

    assert len(docs) > 1
    assert docs[0]["document"].startswith("[(이름 없음) | (카테고리 없음)] ")


# ────────────────────────────────────────────────────────────────────────
# 4) _embed_text_for(): description 우선
# ────────────────────────────────────────────────────────────────────────
def test_embed_text_prefers_description():
    row = _row(description="상세 설명 텍스트")
    assert chunking_service._embed_text_for(row) == "상세 설명 텍스트"


# ────────────────────────────────────────────────────────────────────────
# 5) _embed_text_for(): description 없으면 product_name+category 폴백
# ────────────────────────────────────────────────────────────────────────
def test_embed_text_falls_back_to_name_category():
    row = _row(description=None, product_name="반팔 티셔츠", category="상의")
    assert chunking_service._embed_text_for(row) == "반팔 티셔츠 상의"


def test_embed_text_falls_back_to_default_when_all_missing():
    row = _row(description="", product_name="", category="")
    assert chunking_service._embed_text_for(row) == "상품"


# ────────────────────────────────────────────────────────────────────────
# 6) _embed_text_for(): image_caption 이 있으면 뒤에 이어붙임
# ────────────────────────────────────────────────────────────────────────
def test_embed_text_appends_image_caption():
    row = _row(description="상세 설명", image_caption="화이트 코튼 캐주얼 여름용")
    assert chunking_service._embed_text_for(row) == "상세 설명 화이트 코튼 캐주얼 여름용"


def test_embed_text_no_caption_unaffected():
    row = _row(description="상세 설명", image_caption="")
    assert chunking_service._embed_text_for(row) == "상세 설명"


# ────────────────────────────────────────────────────────────────────────
# 7) _chunk_threshold_chars()/_chunk_overlap_chars(): .env 파싱 및 폴백
# ────────────────────────────────────────────────────────────────────────
def test_chunk_threshold_default(monkeypatch):
    monkeypatch.delenv("CHUNK_THRESHOLD_CHARS", raising=False)
    assert chunking_service._chunk_threshold_chars() == 500


def test_chunk_threshold_custom(monkeypatch):
    monkeypatch.setenv("CHUNK_THRESHOLD_CHARS", "800")
    assert chunking_service._chunk_threshold_chars() == 800


@pytest.mark.parametrize("bad_value", ["not-a-number", "0", "-10"])
def test_chunk_threshold_invalid_falls_back_to_500(monkeypatch, bad_value):
    monkeypatch.setenv("CHUNK_THRESHOLD_CHARS", bad_value)
    assert chunking_service._chunk_threshold_chars() == 500


def test_chunk_overlap_default(monkeypatch):
    monkeypatch.delenv("CHUNK_OVERLAP_CHARS", raising=False)
    assert chunking_service._chunk_overlap_chars() == 50


@pytest.mark.parametrize("bad_value", ["not-a-number", "-5"])
def test_chunk_overlap_invalid_falls_back_to_50(monkeypatch, bad_value):
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", bad_value)
    assert chunking_service._chunk_overlap_chars() == 50


# ────────────────────────────────────────────────────────────────────────
# 8) [회귀] threshold < overlap(기본 50) 조합에서도 예외 없이 청킹돼야 함
#    ("Got a larger chunk overlap than chunk size" ValueError 재현 방지)
# ────────────────────────────────────────────────────────────────────────
def test_threshold_smaller_than_default_overlap_does_not_raise(monkeypatch):
    monkeypatch.setenv("CHUNK_THRESHOLD_CHARS", "30")
    monkeypatch.delenv("CHUNK_OVERLAP_CHARS", raising=False)  # 기본값 50 그대로
    row = _row(description="긴 설명 " * 20)  # 100자, threshold(30) 초과

    docs = chunking_service.build_chunk_documents(row)  # 예외 없이 성공해야 함

    assert len(docs) > 1
    for idx, doc in enumerate(docs):
        assert doc["id"] == f"1_chunk_{idx}"