"""
scripts/index_products_image.py  [멀티모달]
Oracle PRODUCT.IMAGE_URL → OpenCLIP 이미지 임베딩 → ChromaDB 'products_image' 컬렉션.

실행(프로젝트 루트에서):  python -m scripts.index_products_image
사전조건:
  - ALTER TABLE PRODUCT ADD (IMAGE_URL VARCHAR2(1000)) 선행 + 상품 이미지 URL 등록
  - .env 에 DB_* / CHROMA_* 설정
  - CLIP 의존성 설치:  pip install open_clip_torch torch pillow
  - (서빙 인스턴스의 CLIP_SERVING_ENABLED 값과 무관하게) 이 스크립트는 force=True 로
    CLIP 모델을 '강제 로드'한다. 메모리 부족 인스턴스라면 로컬/여유 머신에서 실행 권장.

처리 흐름:
  1) oracle_db.fetch_all_products() 로 전체 상품 로드(이제 image_url 포함)
  2) IMAGE_URL 이 있는 상품만 대상으로:
       이미지 다운로드(http/https/로컬경로) → PIL 디코딩 → CLIP 이미지 임베딩(512차원)
  3) ChromaDB 'products_image' 컬렉션에 upsert
       - id = str(product_id)  (텍스트 컬렉션과 동일 규칙 → 병합/중복제거 키 일치)
       - document/metadata = 텍스트 컬렉션과 동일하게 채움(재랭킹/RAG 일관성)
         · 단 임베딩만 CLIP. 메타데이터에 image_url 포함.

견고성(장애 격리):
  - IMAGE_URL NULL/공백 → 스킵(이미지 미등록 상품)
  - 다운로드 실패/디코딩 실패 → 해당 1건만 스킵하고 경고 로그(전체 중단 금지)
  - 한 건도 성공 못하면 컬렉션 변경 없이 종료
"""
import asyncio
import io
import logging
from urllib.request import urlopen, Request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from database import oracle_db
from services import chroma_service, clip_service, vision_tagging_service
# 텍스트 컬렉션과 동일한 임베딩 텍스트 규칙 재사용(문서/메타 일관성)
from scripts.index_products import _embed_text_for, _to_float

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("index_products_image")

# 이미지 다운로드 타임아웃(초). 느린 URL 하나가 전체 인덱싱을 막지 않도록 제한.
_DOWNLOAD_TIMEOUT = 10
# 다운로드 시 일부 CDN 이 기본 User-Agent 를 막는 경우가 있어 명시.
_HTTP_HEADERS = {"User-Agent": "shoppingmall-ai-indexer/1.0"}


def _load_image(image_url: str):
    """IMAGE_URL → PIL.Image(RGB). http/https URL 과 로컬 파일 경로를 모두 지원.

    실패(네트워크/디코딩/포맷) 시 예외를 그대로 올린다 → 호출 측이 1건 스킵 처리.
    """
    from PIL import Image  # lazy import (pillow 미설치 환경 보호)

    url = image_url.strip()
    if url.startswith("http://") or url.startswith("https://"):
        req = Request(url, headers=_HTTP_HEADERS)
        with urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as resp:
            raw = resp.read()
        img = Image.open(io.BytesIO(raw))
    else:
        # 로컬 파일 경로(예: /home/ubuntu/images/1.jpg)
        img = Image.open(url)
    # CLIP 전처리는 RGB 3채널을 가정 → 흑백/투명 PNG 등도 RGB 로 통일
    return img.convert("RGB")


def _build_metadata(row: dict) -> dict:
    """텍스트 컬렉션(index_products.py)과 동일 규칙의 메타데이터 구성."""
    pid = row.get("product_id")
    return {
        "product_id":   int(pid),
        "product_name": row.get("product_name") or "",
        "category":     row.get("category") or "",
        "price":        _to_float(row.get("price")),
        "description":  str(row.get("description")) if row.get("description") else "",
        "stock":        int(row.get("stock")) if row.get("stock") is not None else 0,
        "image_url":    row.get("image_url") or "",
    }


async def main():
    # 0) ChromaDB 연결 확인
    try:
        hb = await chroma_service.heartbeat()
        logger.info("ChromaDB 연결 OK (heartbeat=%s)", hb)
    except Exception as e:
        logger.error("ChromaDB 연결 실패 — 서버 기동/포트(CHROMA_PORT) 확인 필요: %s", e)
        raise

    # 1) Oracle 상품 로드(동기 → to_thread)
    rows: list[dict] = await asyncio.to_thread(oracle_db.fetch_all_products)
    if not rows:
        logger.warning("PRODUCT 테이블에 상품이 없습니다. 이미지 인덱싱 중단.")
        return
    logger.info("Oracle 상품 %d건 로드", len(rows))

    # 2) IMAGE_URL 있는 상품만 추려 CLIP 이미지 임베딩 생성
    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    skipped_no_url = 0
    skipped_failed = 0

    for row in rows:
        pid = row.get("product_id")
        if pid is None:
            logger.warning("product_id 없는 행 스킵: %r", row)
            continue

        image_url = (row.get("image_url") or "").strip()
        if not image_url:
            skipped_no_url += 1
            continue

        try:
            pil_image = await asyncio.to_thread(_load_image, image_url)
            # force=True: 서빙 플래그와 무관하게 모델 강제 로드(인덱싱 전용)
            embedding = await asyncio.to_thread(clip_service.encode_image, pil_image, True)
        except Exception as e:
            skipped_failed += 1
            logger.warning("이미지 처리 실패 → 스킵: product_id=%s url=%s (%s)", pid, image_url, e)
            continue

        # [Vision 태깅] 이미 태깅된 상품은 재호출하지 않는다(비용 통제 — 재실행마다
        # 매번 Vision API를 다시 부르지 않도록). 실패해도 CLIP 임베딩/인덱싱은 계속.
        if not (row.get("image_caption") or "").strip():
            caption = await vision_tagging_service.generate_image_caption(pil_image)
            if caption:
                await asyncio.to_thread(oracle_db.update_image_caption, pid, caption)
                row = {**row, "image_caption": caption}

        text = _embed_text_for(row)          # 문서/재랭킹용 텍스트(텍스트 컬렉션과 동일)
        ids.append(str(pid))
        embeddings.append(embedding)
        documents.append(text)
        metadatas.append(_build_metadata(row))

    logger.info(
        "이미지 임베딩 대상: 성공 %d건 / URL없음 %d건 / 실패 %d건",
        len(ids), skipped_no_url, skipped_failed,
    )

    if not ids:
        logger.warning("인덱싱할 유효 이미지가 없습니다. 컬렉션 변경 없이 종료.")
        return

    # 3) ChromaDB 'products_image' upsert (idempotent)
    total = await chroma_service.upsert_image_products(ids, embeddings, documents, metadatas)
    logger.info(
        "ChromaDB upsert 완료. 컬렉션 '%s' 총 %d건",
        chroma_service.IMAGE_COLLECTION_NAME, total,
    )


if __name__ == "__main__":
    asyncio.run(main())
