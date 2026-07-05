"""
schemas/product_query_schema.py
semantic_node 에서 "대화 맥락상 특정 상품 하나의 가격/재고를 묻는 질문인지"를
LLM 구조화 출력으로 판단하기 위한 스키마.

[배경] 기존엔 키워드/토큰 텍스트 매칭(_find_referenced_product 등)으로 이걸
판단했는데, 카탈로그에 같은 단어("오버핏" 등)를 공유하는 상품이 여럿이거나
히스토리가 비어있는 경우 등 계속 새로운 엣지케이스로 깨졌다. LLM이 카탈로그
전체를 보고 직접 판단하게 바꿔 이 부류의 문제를 구조적으로 없앤다.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ProductAttributeQuery(BaseModel):
    is_asking_about_specific_product: bool = Field(
        ...,
        description="대화 맥락상 특정 상품 하나의 가격 또는 재고를 묻는 질문인지 여부",
    )
    product_name: Optional[str] = Field(
        None,
        description="묻고 있는 상품의 정확한 이름. 반드시 제공된 상품 목록의 이름과 "
                    "정확히 일치해야 하며, 확신이 없으면 null.",
    )
    attribute: Optional[Literal["price", "stock"]] = Field(
        None,
        description="묻고 있는 속성. 가격이면 'price', 재고면 'stock'. 둘 다 아니면 null.",
    )
