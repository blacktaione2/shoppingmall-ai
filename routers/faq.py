"""
FAQ 라우터: GET /chat/faq
- FAQ 테이블 목록을 JSON 배열로 반환
- ?category=배송 형태의 선택적 카테고리 필터 지원
"""
from typing import Optional

from fastapi import APIRouter, Query

from schemas.chat_schema import FaqItem
from database.oracle_db import fetch_faq_list

router = APIRouter(prefix="/chat", tags=["faq"])


@router.get("/faq", response_model=list[FaqItem])
def get_faq_list(
    category: Optional[str] = Query(default=None, description="FAQ 카테고리 필터 (선택)"),
) -> list[dict]:
    return fetch_faq_list(category=category)
