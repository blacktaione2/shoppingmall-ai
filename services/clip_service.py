"""
services/clip_service.py  [멀티모달]
OpenCLIP 임베딩 전담 모듈 (이미지/텍스트 → 동일 임베딩 공간).

[핵심 설계 — 메모리 보호]
- 이 인스턴스(1GB RAM + swap)에서는 CLIP 모델을 서빙에 상주시키면 swap 스래싱으로
  응답이 무너진다. 그래서 다음 2가지를 분리한다.
    · 서빙(실시간 검색): CLIP_SERVING_ENABLED=false 이면 모델을 '아예 로드하지 않는다'.
                         torch/open_clip 을 import 조차 하지 않아 메모리/의존성 0.
    · 인덱싱(오프라인 스크립트): index_products_image.py 가 force=True 로 호출 →
                                플래그와 무관하게 모델을 강제 로드해 임베딩을 만든다.
                                스크립트 종료와 함께 프로세스가 죽으므로 메모리도 회수된다.

[전환 시나리오]
- 나중에 메모리 여유 있는 서버로 옮기면 .env 의 CLIP_SERVING_ENABLED 만 true 로 바꾸고
  FastAPI 를 재시작하면 끝. 코드 변경/재인덱싱 불필요(products_image 컬렉션은 그대로).

[임베딩 규칙]
- create_model_and_transforms 로 모델/전처리, get_tokenizer 로 토크나이저 확보.
- encode_image/encode_text 결과는 L2 정규화한다(코사인 유사도 기준 정렬을 위해).
  ChromaDB products_image 컬렉션도 cosine space 로 만든다(chroma_service).
- 모든 함수는 '동기'다. FastAPI 경로에서 호출할 땐 asyncio.to_thread 로 감쌀 것.

[모델 선택]
- 기본 ViT-B-32/openai (512차원). 한국어 텍스트 쿼리 정확도가 아쉬우면
  .env 의 CLIP_MODEL_NAME/CLIP_PRETRAINED 를 멀티링구얼 모델로 바꿔 재인덱싱하면 된다.
"""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── 환경 설정 ────────────────────────────────────────────────────────────
# 서빙 시 CLIP 검색 활성화 여부(기본 off → 모델 비로드, 텍스트 검색만 동작)
CLIP_SERVING_ENABLED = os.getenv("CLIP_SERVING_ENABLED", "false").lower() == "true"
# open_clip 모델/사전학습 가중치 이름(교체 가능)
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "openai")

# ── 모듈 싱글톤(지연 초기화) ─────────────────────────────────────────────
_model = None        # open_clip 모델
_preprocess = None   # 이미지 전처리 transform
_tokenizer = None    # 텍스트 토크나이저
_device = None       # "cuda" | "cpu"
_torch = None        # torch 모듈 핸들(no_grad 등에서 재사용)


def is_serving_enabled() -> bool:
    """서빙 경로에서 CLIP 이미지 검색을 사용할지 여부(.env 플래그)."""
    return CLIP_SERVING_ENABLED


def _ensure_loaded(force: bool = False) -> None:
    """CLIP 모델을 1회만 로드한다(지연 초기화).

    Args:
        force: True 면 CLIP_SERVING_ENABLED 와 무관하게 강제 로드(인덱싱 스크립트용).
               False(기본)면 서빙 플래그가 꺼져 있을 때 '로드하지 않고 그냥 반환'한다.
               → 이 경우 _model 은 None 으로 남고, encode_* 호출 시 RuntimeError.

    [중요] torch/open_clip 은 이 함수 안에서만 import 한다(lazy import).
           플래그가 꺼진 서빙 프로세스는 이 무거운 패키지를 메모리에 올리지 않는다.
    """
    global _model, _preprocess, _tokenizer, _device, _torch

    if _model is not None:
        return
    if not force and not CLIP_SERVING_ENABLED:
        # 서빙 비활성 + 강제 아님 → 로드 생략(메모리 0)
        return

    # ── 무거운 의존성은 여기서만 import ──
    import torch              # noqa: WPS433 (의도적 함수 내 import)
    import open_clip          # noqa: WPS433

    _torch = torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED,
    )
    model.eval()
    model.to(_device)

    _model = model
    _preprocess = preprocess
    _tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    logger.info(
        "CLIP 모델 로드 완료: %s/%s (device=%s, force=%s)",
        CLIP_MODEL_NAME, CLIP_PRETRAINED, _device, force,
    )


def _l2_normalize(feat):
    """torch 텐서 feature 를 L2 정규화(코사인 정렬용)."""
    return feat / feat.norm(dim=-1, keepdim=True)


def encode_image(pil_image, force: bool = False) -> list[float]:
    """PIL.Image → CLIP 이미지 임베딩(L2 정규화된 list[float]).

    Args:
        pil_image: PIL.Image 객체(호출 측에서 다운로드/디코딩 완료한 이미지).
        force: 인덱싱 스크립트에서 True(플래그 무시 강제 로드).
    Raises:
        RuntimeError: 서빙 플래그가 꺼져 있고 force=False 라 모델이 없을 때.
    """
    _ensure_loaded(force=force)
    if _model is None:
        raise RuntimeError(
            "CLIP 모델이 로드되지 않았습니다. "
            "서빙 검색은 CLIP_SERVING_ENABLED=true, 인덱싱은 force=True 가 필요합니다."
        )
    with _torch.no_grad():
        tensor = _preprocess(pil_image).unsqueeze(0).to(_device)
        feat = _model.encode_image(tensor)
        feat = _l2_normalize(feat)
    return feat[0].cpu().tolist()


def encode_text(text: str, force: bool = False) -> list[float]:
    """검색 질의 텍스트 → CLIP 텍스트 임베딩(L2 정규화된 list[float]).

    이미지 임베딩과 '동일 공간'이라 텍스트→이미지 교차 검색이 가능하다.
    Args:
        text: 검색 질의(자연어).
        force: 인덱싱/오프라인 테스트에서 True.
    Raises:
        RuntimeError: 모델 미로드 시.
    """
    if text is None or not str(text).strip():
        raise ValueError("CLIP 임베딩 대상 텍스트가 비어 있습니다.")
    _ensure_loaded(force=force)
    if _model is None:
        raise RuntimeError(
            "CLIP 모델이 로드되지 않았습니다. "
            "서빙 검색은 CLIP_SERVING_ENABLED=true, 인덱싱은 force=True 가 필요합니다."
        )
    with _torch.no_grad():
        tokens = _tokenizer([str(text)]).to(_device)
        feat = _model.encode_text(tokens)
        feat = _l2_normalize(feat)
    return feat[0].cpu().tolist()
