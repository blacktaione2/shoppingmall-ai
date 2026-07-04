"""
services/embed_service.py
OpenAI 임베딩 생성 전담 모듈.

- 모델: text-embedding-3-small (1536차원, 확정)
- 단건(get_embedding) / 배치(get_embeddings) 모두 지원
- FastAPI async 환경에 맞춰 AsyncOpenAI 네이티브 비동기 사용
  (네트워크 I/O 라 to_thread 보다 AsyncOpenAI 가 효율적)
"""
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

# 이 모듈이 단독으로 가장 먼저 import 될 경우(예: 인덱싱/점검 스크립트)
# .env 가 아직 os.environ 에 로드되지 않은 상태일 수 있음.
# gpt_service.py / oracle_db.py 와 동일하게 모듈 자체에서 load_dotenv() 를 호출해
# "다른 모듈이 먼저 import 돼서 우연히 .env 가 로드됨" 같은 암묵적 의존성을 제거한다.
# (idempotent: 여러 번 호출돼도 안전)
load_dotenv()

EMBED_MODEL = "text-embedding-3-small"   # 확정 임베딩 모델
EMBED_DIM = 1536                          # 컬렉션 차원(문서/검증용)

_async_client: AsyncOpenAI | None = None  # 모듈 싱글톤(지연 초기화)


def _openai_timeout() -> float | None:
    """LLM_TIMEOUT(초)을 읽어 OpenAI SDK timeout 으로 사용(0 이하/오류면 None=무제한).

    gpt_service 와 동일 규칙. 순환 import 를 피하려 모듈별로 동일 헬퍼를 둔다.
    """
    raw = os.getenv("LLM_TIMEOUT", "30")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 30.0
    return val if val > 0 else None


def _get_client() -> AsyncOpenAI:
    """OPENAI_API_KEY 로 AsyncOpenAI 를 1회만 생성해 재사용한다."""
    global _async_client
    if _async_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다 (.env 확인).")
        # HTTP 요청 타임아웃(.env LLM_TIMEOUT, 기본 30초). 임베딩 호출 무한 대기 방지.
        _async_client = AsyncOpenAI(api_key=api_key, timeout=_openai_timeout())
    return _async_client


async def get_embedding(text: str) -> list[float]:
    """단일 문자열 → 1536차원 임베딩 벡터."""
    if text is None or not str(text).strip():
        raise ValueError("임베딩 대상 텍스트가 비어 있습니다.")
    client = _get_client()
    resp = await client.embeddings.create(model=EMBED_MODEL, input=str(text))
    return resp.data[0].embedding


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """여러 문자열 → 임베딩 벡터 리스트(배치 1회 호출, 입력 순서 보존)."""
    if not texts:
        return []
    # OpenAI 는 빈 문자열 input 에 400 을 반환하므로 공백 1칸으로 치환해 방어.
    cleaned = [t if (t and str(t).strip()) else " " for t in texts]
    client = _get_client()
    resp = await client.embeddings.create(model=EMBED_MODEL, input=cleaned)
    # 응답은 input 순서대로 오지만, index 기준 재정렬로 순서 보장을 한 번 더 확실히 한다.
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in ordered]
