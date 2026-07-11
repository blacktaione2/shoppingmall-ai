"""
services/vision_tagging_service.py
상품 이미지를 Vision LLM(gpt-4o-mini)으로 분석해 색상/소재/스타일 태그 + 짧은 설명을
생성한다. Structured Outputs(JSON Schema)로 형식을 강제해, 결과를 그대로 Oracle
PRODUCT.IMAGE_CAPTION 에 적재할 수 있게 한다.

[호출 시점] 실시간 API가 아니라 scripts/index_products_image.py 의 배치 인덱싱
루프 안에서, 이미지 1건당 1회만 호출된다(상품 등록/재색인 시점 — 실시간 챗봇
응답 경로와 무관해 레이턴시/비용 부담이 없다).

[비용 통제] 이미 IMAGE_CAPTION 이 있는 상품은 호출 측(index_products_image.py)에서
건너뛴다 — 재실행마다 매번 다시 태깅하지 않도록.
"""
import base64
import io
import logging
import os

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_VISION_MODEL = "gpt-4o-mini"

_TAG_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "product_image_tags",
        "schema": {
            "type": "object",
            "properties": {
                "color": {"type": "string", "description": "주된 색상(한국어)"},
                "material": {"type": "string", "description": "소재 느낌(한국어, 예: 린넨/니트/가죽 등)"},
                "style_tags": {
                    "type": "array", "items": {"type": "string"},
                    "description": "스타일 키워드 3~5개(한국어, 예: 캐주얼/오버핏/여름용)",
                },
                "description": {"type": "string", "description": "이미지에 대한 한 문장 설명(한국어)"},
            },
            "required": ["color", "material", "style_tags", "description"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

_async_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        _async_client = AsyncOpenAI(api_key=api_key)
    return _async_client


def _image_to_data_url(pil_image) -> str:
    """PIL.Image → base64 data URL. 원격/로컬 URL 접근성과 무관하게 항상 동작하도록,
    이미 로드된 PIL 이미지를 그대로 재사용한다(CLIP 임베딩에 쓰는 것과 동일 객체).
    """
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


async def generate_image_caption(pil_image) -> str | None:
    """이미지 1건을 태깅해 "색상/소재/스타일/설명"을 합친 한 줄 문자열로 반환한다.

    실패(API 오류 등) 시 None 반환 — 호출 측이 해당 상품만 스킵하고 계속 진행한다
    (전체 배치가 이 실패로 멈추지 않도록).
    """
    try:
        client = _get_client()
        resp = await client.chat.completions.create(
            model=_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "이 상품 이미지를 분석해 색상/소재/스타일/설명을 알려줘."},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(pil_image)}},
                ],
            }],
            response_format=_TAG_SCHEMA,
        )
        import json
        data = json.loads(resp.choices[0].message.content)
        tags = "/".join(data.get("style_tags") or [])
        return f"{data.get('color', '')} {data.get('material', '')} {tags} {data.get('description', '')}".strip()
    except Exception:
        logger.exception("이미지 Vision 태깅 실패(스킵)")
        return None
