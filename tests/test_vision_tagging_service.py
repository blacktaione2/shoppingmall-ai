"""
tests/test_vision_tagging_service.py
services/vision_tagging_service.py 단위 테스트 (오프라인, 실제 OpenAI 호출 없음).

[전략]
- 프로젝트 관례에 따라 pytest-asyncio 없이 asyncio.run() 으로 실행.
- `_get_client()` 를 monkeypatch 해서 실제 네트워크 호출 없이 고정된
  응답/예외를 주입한다(tests/test_voice_endpoints.py 와 동일한 monkeypatch 전략).

[검증 시나리오]
1. 정상 응답 → "색상 소재 태그1/태그2 설명" 형식 문자열로 합쳐서 반환
2. API 예외 발생 → None 반환(호출 측이 스킵하고 계속 진행할 수 있도록)
3. 요청 payload에 detail:"low" 가 포함되는지(TPM 레이트리밋 대응)
4. 전송용 data URL 이미지가 512px 이하로 축소되는지, 원본 이미지 객체는
   mutate 되지 않는지(CLIP 임베딩에 재사용되는 동일 객체이므로)
"""
import asyncio
import base64
import io
import json

from PIL import Image

from services import vision_tagging_service


def _make_image(size=(1024, 768)) -> Image.Image:
    return Image.new("RGB", size, color=(200, 50, 50))


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """호출 인자를 기록하고 고정된 응답(또는 예외)을 돌려주는 가짜 completions."""

    def __init__(self, payload: dict | None = None, exc: Exception | None = None):
        self._payload = payload
        self._exc = exc
        self.last_kwargs: dict | None = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._exc is not None:
            raise self._exc
        return _FakeResponse(json.dumps(self._payload))


class _FakeChat:
    def __init__(self, completions: _FakeCompletions):
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions):
        self.chat = _FakeChat(completions)


# ────────────────────────────────────────────────────────────────────────
# 1) 정상 응답 → 문자열 조합
# ────────────────────────────────────────────────────────────────────────
def test_generate_image_caption_success(monkeypatch):
    payload = {
        "color": "화이트",
        "material": "코튼",
        "style_tags": ["캐주얼", "오버핏", "여름용"],
        "description": "시원한 여름용 반팔 티셔츠",
    }
    fake_completions = _FakeCompletions(payload=payload)
    monkeypatch.setattr(
        vision_tagging_service, "_get_client",
        lambda: _FakeClient(fake_completions), raising=True,
    )

    result = asyncio.run(vision_tagging_service.generate_image_caption(_make_image()))

    assert result == "화이트 코튼 캐주얼/오버핏/여름용 시원한 여름용 반팔 티셔츠"


# ────────────────────────────────────────────────────────────────────────
# 2) API 예외 → None (배치 스킵 대상)
# ────────────────────────────────────────────────────────────────────────
def test_generate_image_caption_api_error_returns_none(monkeypatch):
    fake_completions = _FakeCompletions(exc=RuntimeError("429 rate limit"))
    monkeypatch.setattr(
        vision_tagging_service, "_get_client",
        lambda: _FakeClient(fake_completions), raising=True,
    )

    result = asyncio.run(vision_tagging_service.generate_image_caption(_make_image()))

    assert result is None


# ────────────────────────────────────────────────────────────────────────
# 3) TPM 레이트리밋 대응: detail:"low" 가 요청에 포함되는지
# ────────────────────────────────────────────────────────────────────────
def test_generate_image_caption_uses_low_detail(monkeypatch):
    payload = {"color": "블랙", "material": "니트", "style_tags": ["캐주얼"], "description": "니트"}
    fake_completions = _FakeCompletions(payload=payload)
    monkeypatch.setattr(
        vision_tagging_service, "_get_client",
        lambda: _FakeClient(fake_completions), raising=True,
    )

    asyncio.run(vision_tagging_service.generate_image_caption(_make_image()))

    image_part = fake_completions.last_kwargs["messages"][0]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["detail"] == "low"


# ────────────────────────────────────────────────────────────────────────
# 4) 이미지 축소(512px) + 원본 mutate 방지
# ────────────────────────────────────────────────────────────────────────
def test_image_to_data_url_resizes_and_does_not_mutate_original():
    original = _make_image(size=(1024, 768))

    data_url = vision_tagging_service._image_to_data_url(original)

    # 원본 객체는 축소되지 않아야 함(CLIP 임베딩에 동일 객체가 재사용되므로)
    assert original.size == (1024, 768)

    assert data_url.startswith("data:image/jpeg;base64,")
    b64_payload = data_url.split(",", 1)[1]
    decoded = base64.b64decode(b64_payload)
    resized = Image.open(io.BytesIO(decoded))
    assert max(resized.size) <= 512


def test_image_to_data_url_small_image_unchanged_ratio():
    # 이미 512 이하인 이미지는 thumbnail 이 사실상 그대로 둠(비율도 유지)
    original = _make_image(size=(300, 200))

    data_url = vision_tagging_service._image_to_data_url(original)
    decoded = base64.b64decode(data_url.split(",", 1)[1])
    resized = Image.open(io.BytesIO(decoded))

    assert resized.size == (300, 200)