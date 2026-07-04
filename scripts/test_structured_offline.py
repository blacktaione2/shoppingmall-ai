"""
scripts/test_structured_offline.py
STRUCTURED 핸들러 오프라인 테스트 (실제 Oracle DB 불필요)

검증 전략 (2단계)
  1) build_structured_query() : 순수 함수이므로 mock 없이 직접 호출해 (sql, binds) 검증
  2) search_products_structured() / handle_structured() :
     get_connection 을 MockConnection 으로 패치 → 실제 SQL 실행 경로(LOB/숫자 변환, 포맷)를
     그대로 통과시키며 가짜 row 로 결과 검증
"""
import os
import sys
import asyncio
from enum import Enum

# 프로젝트 루트를 path 에 추가 (scripts/ 하위에서 실행 대비)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database.oracle_db as db
import pipeline.structured_handler as sh
from schemas.intent_schema import IntentResult, IntentType, Entities


# ─────────────────────────────────────────────────────────────────────────────
# 테스트용 SortType (schemas.intent_schema.SortType 정의와 동일하게 재현) — Enum 경로 검증용
# ─────────────────────────────────────────────────────────────────────────────
class SortType(str, Enum):
    PRICE_ASC = "PRICE_ASC"
    PRICE_DESC = "PRICE_DESC"
    LATEST = "LATEST"
    DEFAULT = "DEFAULT"


# ─────────────────────────────────────────────────────────────────────────────
# Mock 인프라: 실제 DB 없이 SQL/binds 캡처 + 가짜 row 반환
# ─────────────────────────────────────────────────────────────────────────────
class _FakeLOB:
    """CLOB 흉내: .read() 가능 객체 → _read_lob 분기 검증용"""
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text


class MockCursor:
    def __init__(self, rows, description):
        self._rows = rows
        self.description = description
        self.executed_sql = None
        self.executed_binds = None

    def execute(self, sql, binds=None):
        self.executed_sql = sql
        self.executed_binds = binds or {}

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class MockConnection:
    """get_connection() 대체. 마지막 실행 cursor 를 클래스 변수로 보관해 검증 가능"""
    last_cursor = None

    def __init__(self, rows, description):
        self._rows = rows
        self._description = description

    def cursor(self):
        cur = MockCursor(self._rows, self._description)
        MockConnection.last_cursor = cur
        return cur

    def commit(self):  # 혹시 모를 호출 대비 (STRUCTURED 는 SELECT 라 미사용)
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


# PRODUCT SELECT 컬럼 순서 (search_products_structured 의 SELECT 와 일치해야 함)
_DESCRIPTION = [
    ("PRODUCT_ID",), ("PRODUCT_NAME",), ("CATEGORY",),
    ("PRICE",), ("DESCRIPTION",), ("STOCK",),
]


def _patch_connection(rows):
    """db.get_connection 을 가짜 rows 반환 버전으로 교체

    ※ search_products_structured 는 자기 모듈의 전역 get_connection 을 호출하므로
       db.get_connection 만 바꿔치기하면 패치가 정확히 적용된다.
    """
    def _factory(*args, **kwargs):
        return MockConnection(rows, _DESCRIPTION)
    db.get_connection = _factory


# ─────────────────────────────────────────────────────────────────────────────
# 1) 순수 빌더 build_structured_query() 검증 (mock 불필요)
# ─────────────────────────────────────────────────────────────────────────────
def test_builder_category_only():
    sql, binds = db.build_structured_query(category="운동화")
    assert "WHERE CATEGORY = :category" in sql
    assert binds["category"] == "운동화"
    assert binds["limit_rows"] == 5
    assert "FETCH FIRST :limit_rows ROWS ONLY" in sql
    print("✅ test_builder_category_only")


def test_builder_price_max_only():
    sql, binds = db.build_structured_query(price_max=50000)
    assert "PRICE <= :price_max" in sql
    assert binds["price_max"] == 50000
    assert "WHERE" in sql
    print("✅ test_builder_price_max_only")


def test_builder_price_zero_min():
    # 경계값: price_min=0 도 유효 조건이어야 함 (truthy 검사였다면 누락됨)
    sql, binds = db.build_structured_query(price_min=0)
    assert "PRICE >= :price_min" in sql
    assert binds["price_min"] == 0
    print("✅ test_builder_price_zero_min (0원 경계값 포함)")


def test_builder_keywords():
    sql, binds = db.build_structured_query(keywords=["나이키", "운동화"])
    assert "PRODUCT_NAME LIKE :kw0" in sql
    assert "PRODUCT_NAME LIKE :kw1" in sql
    assert " OR " in sql
    assert binds["kw0"] == "%나이키%"
    assert binds["kw1"] == "%운동화%"
    print("✅ test_builder_keywords")


def test_builder_combined_and():
    sql, binds = db.build_structured_query(
        category="신발", price_max=80000, sort_by="PRICE_ASC"
    )
    assert "CATEGORY = :category" in sql
    assert "PRICE <= :price_max" in sql
    assert " AND " in sql                      # 조건 AND 결합
    assert "ORDER BY PRICE ASC" in sql
    print("✅ test_builder_combined_and")


def test_builder_no_entities():
    sql, binds = db.build_structured_query()
    assert "WHERE" not in sql                  # 조건 없으면 WHERE 절 자체가 없어야 함
    assert "ORDER BY PRODUCT_ID ASC" in sql    # 기본 정렬
    assert binds == {"limit_rows": 5}          # limit 만 존재
    print("✅ test_builder_no_entities")


def test_builder_keyword_limit():
    # 키워드 6개 → 5개로 절단
    sql, binds = db.build_structured_query(keywords=["a", "b", "c", "d", "e", "f"])
    assert "kw4" in binds
    assert "kw5" not in binds
    assert binds["kw4"] == "%e%"
    print("✅ test_builder_keyword_limit (6→5 절단)")


def test_builder_keyword_str_promotion():
    # 단일 문자열이 들어와도 문자 단위로 쪼개지지 않아야 함
    sql, binds = db.build_structured_query(keywords="운동화")
    assert binds.get("kw0") == "%운동화%"
    assert "kw1" not in binds                   # '운','동','화' 로 쪼개지면 안 됨
    print("✅ test_builder_keyword_str_promotion (문자열 방어)")


def test_builder_keyword_blank_filtered():
    # 공백/빈 문자열은 제거
    sql, binds = db.build_structured_query(keywords=["  ", "", "나이키"])
    assert binds.get("kw0") == "%나이키%"
    assert "kw1" not in binds
    print("✅ test_builder_keyword_blank_filtered")


def test_builder_sort_mapping():
    # 모든 sort_by 값 → ORDER BY 매핑 검증
    cases = [
        ("PRICE_ASC", "ORDER BY PRICE ASC"),
        ("PRICE_DESC", "ORDER BY PRICE DESC"),
        ("LATEST", "ORDER BY PRODUCT_ID DESC"),
        ("DEFAULT", "ORDER BY PRODUCT_ID ASC"),
        (None, "ORDER BY PRODUCT_ID ASC"),
    ]
    for sort_val, expected in cases:
        sql, _ = db.build_structured_query(sort_by=sort_val)
        assert expected in sql, f"sort_by={sort_val} 매핑 실패: {sql}"
    print("✅ test_builder_sort_mapping (5종)")


def test_builder_sort_enum():
    # SortType Enum 멤버로 들어와도 .value 로 정상 매핑
    sql, _ = db.build_structured_query(sort_by=SortType.PRICE_DESC)
    assert "ORDER BY PRICE DESC" in sql
    print("✅ test_builder_sort_enum (Enum 경로)")


def test_builder_sort_injection_defense():
    # 화이트리스트 밖 값 → 기본 정렬로 무력화 (SQL Injection 방어)
    sql, _ = db.build_structured_query(sort_by="PRICE; DROP TABLE PRODUCT--")
    assert "ORDER BY PRODUCT_ID ASC" in sql
    assert "DROP" not in sql
    print("✅ test_builder_sort_injection_defense")


def test_builder_full_combo():
    # 전 항목 조합 + 키워드 OR 그룹 괄호 확인
    sql, binds = db.build_structured_query(
        category="운동화", price_min=10000, price_max=100000,
        keywords=["나이키", "에어"], sort_by="PRICE_ASC", limit=3,
    )
    assert "CATEGORY = :category" in sql
    assert "PRICE >= :price_min" in sql
    assert "PRICE <= :price_max" in sql
    assert "(PRODUCT_NAME LIKE :kw0 ESCAPE '\\' OR PRODUCT_NAME LIKE :kw1 ESCAPE '\\')" in sql
    assert "ORDER BY PRICE ASC" in sql
    assert binds["limit_rows"] == 3
    print("✅ test_builder_full_combo (전 조건 + OR 괄호)")


# ─────────────────────────────────────────────────────────────────────────────
# 2) search_products_structured() 실행 + 변환 검증 (connection mock)
# ─────────────────────────────────────────────────────────────────────────────
def test_search_number_conversion():
    from decimal import Decimal
    rows = [
        (Decimal("1"), "나이키 운동화", "운동화", Decimal("89000"),
         _FakeLOB("가벼운 러닝화"), Decimal("15")),
    ]
    _patch_connection(rows)
    result = db.search_products_structured(category="운동화")
    p = result[0]
    assert p["product_id"] == 1 and isinstance(p["product_id"], int)
    assert p["price"] == 89000 and isinstance(p["price"], int)   # Decimal→int
    assert p["stock"] == 15 and isinstance(p["stock"], int)
    assert p["description"] == "가벼운 러닝화"                     # LOB.read() 수행됨
    print("✅ test_search_number_conversion (Decimal→int + LOB read)")


def test_search_empty():
    _patch_connection([])
    result = db.search_products_structured(category="없는카테고리")
    assert result == []
    print("✅ test_search_empty")


# ─────────────────────────────────────────────────────────────────────────────
# 3) handle_structured() 응답 포맷 검증 (async + connection mock)
#    [수정] handle_structured(question, intent_result) 는 IntentResult 만
#    받는다(intent_result.entities 가 항상 진짜 Entities 인스턴스). 과거엔 dict/object/None을
#    직접 entities 자리에 넣어도 받아주는 호환 헬퍼가 있었으나 의도적으로 제거되었고,
#    그 호환성만 검증하던 테스트(dict/object/None) 3개는 더 이상 존재하지 않는 동작을 검증하던
#    것이라 함께 삭제했다. 아래 2개는 IntentResult 로 감싸 동일한 포맷/엣지 검증을 유지한다.
# ─────────────────────────────────────────────────────────────────────────────
def test_handler_format_and_soldout():
    rows = [
        (1, "나이키 에어맥스", "운동화", 89000, _FakeLOB("d1"), 15),
        (2, "아디다스 슬리퍼", "슬리퍼", 35000, _FakeLOB("d2"), 0),   # 품절
    ]
    _patch_connection(rows)
    intent_result = IntentResult(intent=IntentType.STRUCTURED_QUERY, entities=Entities(category="운동화"))
    out = asyncio.run(sh.handle_structured("운동화 추천", intent_result))
    assert out.startswith("[검색결과]")
    assert "나이키 에어맥스" in out
    assert "89,000원" in out                       # 천단위 콤마
    assert "아디다스 슬리퍼 [품절]" in out          # stock=0 → 품절 태그
    assert "2개를 찾았어요" in out                  # 헤더 건수
    print("✅ test_handler_format_and_soldout")


def test_handler_no_result():
    _patch_connection([])
    intent_result = IntentResult(intent=IntentType.STRUCTURED_QUERY, entities=Entities())
    out = asyncio.run(sh.handle_structured("없는상품", intent_result))
    assert out.startswith("[검색결과]")
    assert "찾지 못했" in out
    print("✅ test_handler_no_result")


# ─────────────────────────────────────────────────────────────────────────────
# 러너
# ─────────────────────────────────────────────────────────────────────────────
def main():
    tests = [
        # 1) 순수 빌더
        test_builder_category_only,
        test_builder_price_max_only,
        test_builder_price_zero_min,
        test_builder_keywords,
        test_builder_combined_and,
        test_builder_no_entities,
        test_builder_keyword_limit,
        test_builder_keyword_str_promotion,
        test_builder_keyword_blank_filtered,
        test_builder_sort_mapping,
        test_builder_sort_enum,
        test_builder_sort_injection_defense,
        test_builder_full_combo,
        # 2) 실행 + 변환
        test_search_number_conversion,
        test_search_empty,
        # 3) 핸들러 포맷
        test_handler_format_and_soldout,
        test_handler_no_result,
    ]
    print("=" * 56)
    print(" STRUCTURED 핸들러 오프라인 테스트")
    print("=" * 56)
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"❌ {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"💥 {t.__name__}: {type(e).__name__}: {e}")
    print("=" * 56)
    if failed == 0:
        print(f"🎉 전체 {len(tests)}개 테스트 통과!")
    else:
        print(f"⚠️  {failed}/{len(tests)}개 실패")
    print("=" * 56)
    return failed


if __name__ == "__main__":
    sys.exit(main())