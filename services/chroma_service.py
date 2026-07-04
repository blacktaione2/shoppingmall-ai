"""
services/chroma_service.py
ChromaDB(HttpClient, Client-Server 모드) 접근 전담 모듈.

- 기동: ChromaDB 를 별도 프로세스로 띄우고(서버) HttpClient 로 접속한다.
  배포 환경에서는 `chroma run` CLI 를 systemd 서비스로 관리한다(호스트 8001).
  (Docker 미사용 — 배포 인스턴스의 가상화 제약으로 컨테이너 대신 CLI+systemd 채택)
- 거리 공간: cosine
  (text-embedding-3-small 벡터는 L2 정규화되어 cosine·dot·L2 랭킹이 동일하지만,
   cosine 거리는 0~2 로 해석이 직관적이라 명시적으로 cosine 사용)
- chromadb 클라이언트 메서드는 전부 동기(블로킹) → asyncio.to_thread 로 래핑해
  FastAPI 이벤트 루프를 막지 않는다. (FAQ 핸들러의 DB 비블로킹 패턴과 동일)

[멀티모달 — Dual Indexing]
- 기존 'products' 텍스트 컬렉션(1536차원, text-embedding-3-small)에 더해
  'products_image' 이미지 컬렉션(512차원, OpenCLIP)을 함께 관리한다.
- 두 컬렉션은 '차원이 다르므로' 반드시 분리되어야 한다(섞으면 dimension mismatch).
- 싱글톤은 컬렉션 1개(_collection) → 이름별 캐시(_collections dict)로 확장하되,
  기존 함수 시그니처/기본 동작(products)은 그대로 유지한다(하위호환).
"""
import os
import asyncio

import chromadb
from dotenv import load_dotenv

# 이 모듈이 단독으로 가장 먼저 import 될 경우(예: 인덱싱/점검 스크립트)
# .env 가 아직 os.environ 에 로드되지 않아 CHROMA_HOST/CHROMA_PORT 가 .env 설정을 못 읽고
# 디폴트값("localhost"/"8001")에 우연히 의존하게 되는 문제를 방지한다.
# gpt_service.py / oracle_db.py 와 동일하게 모듈 자체에서 호출 (idempotent).
load_dotenv()

COLLECTION_NAME = "products"     # 상품 텍스트 임베딩 컬렉션명(3~63자, 소문자 시작/끝)
# 이미지(CLIP) 임베딩 컬렉션명. .env 로 교체 가능.
IMAGE_COLLECTION_NAME = os.getenv("CLIP_IMAGE_COLLECTION", "products_image")
DISTANCE_SPACE = "cosine"        # HNSW 거리 공간

_client = None                   # chromadb.HttpClient 싱글톤
# 컬렉션 핸들을 '이름별'로 캐시(기존 단일 _collection 대체).
#   · _collections["products"]       → 텍스트 컬렉션
#   · _collections["products_image"] → 이미지 컬렉션
_collections: dict = {}


def _get_client():
    """CHROMA_HOST / CHROMA_PORT 로 HttpClient 를 1회만 생성해 재사용."""
    global _client
    if _client is None:
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8001"))
        _client = chromadb.HttpClient(host=host, port=port)
    return _client


def _get_collection(name: str = COLLECTION_NAME):
    """지정한 컬렉션을 get_or_create(cosine)로 확보해 이름별로 캐시·재사용.

    name 기본값은 'products'(텍스트) → 기존 호출부는 인자 없이 그대로 동작한다.
    'products_image' 를 넘기면 이미지 컬렉션을 반환한다.
    """
    col = _collections.get(name)
    if col is None:
        client = _get_client()
        col = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": DISTANCE_SPACE},
        )
        _collections[name] = col
    return col


# ---------------- 동기 내부 구현 (to_thread 로 감싸 호출) ----------------

def _upsert_sync(ids, embeddings, documents, metadatas, collection_name=COLLECTION_NAME):
    col = _get_collection(collection_name)
    # upsert: 동일 id 재실행 시 중복 add 오류 없이 갱신(idempotent)
    col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return col.count()


def _query_sync(query_embedding, n_results, where, collection_name=COLLECTION_NAME):
    col = _get_collection(collection_name)
    return col.query(
        query_embeddings=[query_embedding],            # 단일 쿼리 → 1개짜리 배치
        n_results=n_results,
        where=where,                                   # None 이면 메타데이터 필터 없음
        include=["documents", "metadatas", "distances"],
    )


def _count_sync(collection_name=COLLECTION_NAME):
    return _get_collection(collection_name).count()


def _delete_sync(ids, collection_name=COLLECTION_NAME):
    """주어진 id 리스트를 컬렉션에서 삭제. 반환: 삭제 후 총 건수.
    존재하지 않는 id 를 delete 해도 chromadb 는 예외 없이 무시한다(idempotent)."""
    col = _get_collection(collection_name)
    col.delete(ids=ids)
    return col.count()


def _heartbeat_sync():
    return _get_client().heartbeat()


def _get_embeddings_by_ids_sync(ids, collection_name=COLLECTION_NAME) -> dict:
    """주어진 id 들의 임베딩을 조회해 {id: list[float]} 로 반환.

    ChromaDB get(include=["embeddings"]) 은 임베딩을 numpy 배열로 돌려줄 수 있어
    순수 Python 연산(가중합산) 전에 list 로 변환한다.
    존재하지 않는 id 는 결과에서 자연히 빠진다(예외 아님).
    """
    col = _get_collection(collection_name)
    res = col.get(ids=ids, include=["embeddings"])
    out: dict = {}
    res_ids = res.get("ids") or []
    res_embs = res.get("embeddings")
    if res_embs is None:
        return out
    for i in range(len(res_ids)):
        emb = res_embs[i]
        if emb is None:
            continue
        # numpy 배열/그 외 시퀀스 → list[float] 통일
        out[res_ids[i]] = emb.tolist() if hasattr(emb, "tolist") else list(emb)
    return out


def _to_hits(res) -> list[dict]:
    """chromadb query 결과(2중 리스트)를 hits dict 리스트로 변환(공통 헬퍼)."""
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    hits = []
    for i in range(len(ids)):
        hits.append({
            "id": ids[i],
            "document": docs[i],
            "metadata": metas[i],
            "distance": dists[i],
        })
    return hits


# ---------------- 외부 공개 async API (텍스트 컬렉션, 기존 유지) ----------------

async def upsert_products(ids, embeddings, documents, metadatas) -> int:
    """상품 텍스트 임베딩 upsert(idempotent). 반환: upsert 후 컬렉션 총 건수."""
    return await asyncio.to_thread(_upsert_sync, ids, embeddings, documents, metadatas)


async def search_similar(query_embedding, n_results: int = 4, where: dict | None = None) -> list[dict]:
    """
    쿼리 임베딩으로 유사 상품 top-k 검색(텍스트 컬렉션 'products').
      where: 선택적 메타데이터 필터(dict). SEMANTIC 핸들러는 None(순수 벡터검색),
             추후 STRUCTURED 핸들러가 category/price 필터로 재사용 가능.
    반환: [{id, document, metadata, distance}, ...]  (distance 오름차순 = 가까운 순)
    """
    res = await asyncio.to_thread(_query_sync, query_embedding, n_results, where)
    return _to_hits(res)


async def count() -> int:
    """텍스트 컬렉션에 저장된 임베딩 총 개수."""
    return await asyncio.to_thread(_count_sync)


async def delete_product(product_id) -> int:
    """상품 1건을 ChromaDB 'products' 컬렉션에서 삭제. 반환: 삭제 후 총 건수.

    관리자가 Spring Boot 에서 상품을 삭제하면 FastApiSyncService 가
    DELETE /admin/products/{id} 를 호출하고, 그 핸들러(routers/admin.py)가 이 함수를 쓴다.
    id 는 인덱싱 시점과 동일하게 문자열(str(product_id))로 맞춘다(index_products.py 규칙).
    """
    return await asyncio.to_thread(_delete_sync, [str(product_id)])


# ---------------- 외부 공개 async API (이미지 컬렉션, 신규) ----------------

async def upsert_image_products(ids, embeddings, documents, metadatas) -> int:
    """상품 '이미지(CLIP)' 임베딩을 products_image 컬렉션에 upsert. 반환: 총 건수.

    documents/metadatas 는 텍스트 컬렉션과 '동일 규칙'으로 채운다(상품명/설명 등).
    이렇게 해야 이미지 검색 결과도 재랭킹(document 기반)·RAG 컨텍스트에 그대로 쓰인다.
    임베딩 벡터만 CLIP(512차원)이라는 점이 텍스트 컬렉션과의 유일한 차이다.
    """
    return await asyncio.to_thread(
        _upsert_sync, ids, embeddings, documents, metadatas, IMAGE_COLLECTION_NAME
    )


async def search_similar_image(query_embedding, n_results: int = 4,
                               where: dict | None = None) -> list[dict]:
    """CLIP 쿼리 임베딩으로 products_image 컬렉션에서 top-k 검색.

    반환 형식은 search_similar 와 동일([{id, document, metadata, distance}, ...]).
    호출 측(rag_pipeline)이 텍스트 결과와 product_id 기준으로 병합·중복제거한다.
    """
    res = await asyncio.to_thread(
        _query_sync, query_embedding, n_results, where, IMAGE_COLLECTION_NAME
    )
    return _to_hits(res)


async def count_image() -> int:
    """이미지 컬렉션(products_image)에 저장된 임베딩 총 개수."""
    return await asyncio.to_thread(_count_sync, IMAGE_COLLECTION_NAME)


async def delete_image_product(product_id) -> int:
    """상품 1건을 products_image 컬렉션에서 삭제. 반환: 삭제 후 총 건수.

    상품 삭제 동기화 시 텍스트(delete_product)와 함께 호출해 양쪽 컬렉션 일관성을 맞춘다.
    이미지 컬렉션에 해당 id 가 없어도(이미지 미등록 상품) idempotent 하게 무시된다.
    """
    return await asyncio.to_thread(_delete_sync, [str(product_id)], IMAGE_COLLECTION_NAME)


async def heartbeat() -> int:
    """ChromaDB 서버 연결 확인(나노초 타임스탬프 반환). 연결 실패 시 예외 발생."""
    return await asyncio.to_thread(_heartbeat_sync)


async def get_embeddings_by_ids(ids: list[str]) -> dict:
    """상품 id 리스트의 임베딩을 텍스트 컬렉션에서 조회.

    반환: {id(str): list[float]}  (존재하지 않는 id 는 제외)
    개인화 취향 벡터(구매 이력 임베딩 가중합산) 계산에 사용된다.
    """
    if not ids:
        return {}
    return await asyncio.to_thread(_get_embeddings_by_ids_sync, ids)
