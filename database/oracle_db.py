"""
database/oracle_db.py
Oracle Autonomous DB 연결 모듈 (Wallet 기반 TLS)
"""
import os
import oracledb
from dotenv import load_dotenv

load_dotenv()

import logging

logger = logging.getLogger(__name__)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DSN = os.getenv("DB_DSN")
WALLET_DIR = os.getenv("WALLET_DIR", "")
WALLET_PASSWORD = os.getenv("WALLET_PASSWORD", "") or None


def _get_int(name: str, default: int) -> int:
    """.env 정수 설정 헬퍼(파싱 실패 시 기본값)."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ── [개선] Oracle Connection Pool ────────────────────────────────────────
# 기존엔 모든 함수가 get_connection() 으로 '매 호출마다 새 커넥션'을 만들었다.
# FastAPI 동시 요청 시 연결 생성 오버헤드 + Autonomous DB 세션 수 압박이 생긴다.
# create_pool 로 커넥션을 재사용해 이를 줄인다.
#
# [안전장치 — 회귀 0 보장]
#   · 풀 생성은 init_pool() 에서 시도하되, 실패하면 _pool 을 None 으로 두고
#     get_connection() 이 '기존 방식(직접 connect)'으로 자동 폴백한다.
#     → Wallet/네트워크 등으로 풀 생성이 안 되는 환경에서도 기존과 동일하게 동작.
#   · get_connection() 의 반환은 풀/직접연결 모두 'with 문으로 닫는' 객체라
#     기존 모든 호출부(`with get_connection() as conn:`)가 그대로 호환된다.
#     (pool.acquire() 로 얻은 커넥션은 with 블록 종료 시 풀에 '반납'된다)
_pool: oracledb.ConnectionPool | None = None

# 풀 크기(.env 로 조정 가능). 단일 인스턴스/ARM 2 OCPU 기준 보수적 기본값.
_POOL_MIN = _get_int("ORACLE_POOL_MIN", 2)
_POOL_MAX = _get_int("ORACLE_POOL_MAX", 10)
_POOL_INCREMENT = _get_int("ORACLE_POOL_INCREMENT", 1)


def init_pool() -> None:
    """[개선] 앱 기동(main.py lifespan) 시 1회 호출해 커넥션 풀을 생성한다.

    실패해도 예외를 올리지 않고 _pool=None 으로 남겨, get_connection() 이
    기존 직접연결 방식으로 폴백하도록 한다(조용한 회귀 방지 + 기동 보호).
    이미 풀이 있으면(중복 호출) no-op.
    """
    global _pool
    if _pool is not None:
        return
    if not all([DB_USER, DB_PASSWORD, DB_DSN]):
        logger.warning("DB 설정 누락 → 풀 생성 건너뜀(직접연결 폴백)")
        return
    try:
        if WALLET_DIR:
            _pool = oracledb.create_pool(
                user=DB_USER,
                password=DB_PASSWORD,
                dsn=DB_DSN,
                config_dir=WALLET_DIR,
                wallet_location=WALLET_DIR,
                wallet_password=WALLET_PASSWORD,
                min=_POOL_MIN,
                max=_POOL_MAX,
                increment=_POOL_INCREMENT,
            )
        else:
            _pool = oracledb.create_pool(
                user=DB_USER,
                password=DB_PASSWORD,
                dsn=DB_DSN,
                min=_POOL_MIN,
                max=_POOL_MAX,
                increment=_POOL_INCREMENT,
            )
        logger.info(
            "Oracle 커넥션 풀 생성 완료 (min=%d, max=%d)", _POOL_MIN, _POOL_MAX
        )
    except Exception:
        _pool = None
        logger.exception("Oracle 풀 생성 실패 → 직접연결 방식으로 폴백")


def close_pool() -> None:
    """[개선] 앱 종료 시 호출해 풀을 정리한다(없으면 no-op)."""
    global _pool
    if _pool is not None:
        try:
            _pool.close()
            logger.info("Oracle 커넥션 풀 정리 완료")
        except Exception:
            logger.exception("Oracle 풀 정리 중 예외(무시)")
        finally:
            _pool = None


def _connect_direct() -> oracledb.Connection:
    """풀 미사용(폴백) 시 기존 방식의 직접 커넥션 생성."""
    if WALLET_DIR:
        return oracledb.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            dsn=DB_DSN,
            config_dir=WALLET_DIR,
            wallet_location=WALLET_DIR,
            wallet_password=WALLET_PASSWORD,
        )
    return oracledb.connect(user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN)


def get_connection() -> oracledb.Connection:
    """
    Oracle DB 커넥션 획득 (Thin 모드, Wallet 지원).
    - 풀이 준비돼 있으면 풀에서 acquire()(재사용) → with 종료 시 풀에 반납.
    - 풀이 없으면(미초기화/생성실패) 기존 방식대로 직접 connect.
    두 경로 모두 `with get_connection() as conn:` 패턴과 호환된다.
    """
    if not all([DB_USER, DB_PASSWORD, DB_DSN]):
        raise RuntimeError(
            ".env 의 DB_USER / DB_PASSWORD / DB_DSN 설정을 확인하세요."
        )
    if _pool is not None:
        return _pool.acquire()
    return _connect_direct()


def fetch_all_products() -> list[dict]:
    """
    PRODUCT 테이블 전체 조회 > dict 리스트 반환
    품절 상품(STOCK=0)도 포함하여 인덱싱 (검색 시 metadata로 필터링)
    """
    sql = """
        SELECT PRODUCT_ID,
               PRODUCT_NAME,
               CATEGORY,
               PRICE,
               DESCRIPTION,
               STOCK,
               IMAGE_URL,
               STATUS
          FROM PRODUCT
         ORDER BY PRODUCT_ID
    """
    products = []
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            columns = [col[0].lower() for col in cursor.description]
            rows = cursor.fetchall()

            for row in rows:
                record = dict(zip(columns, row))
                record["description"] = _read_lob(record.get("description"))
                products.append(record)
    return products


def fetch_product_by_id(product_id) -> dict | None:
    sql = """
        SELECT PRODUCT_ID,
               PRODUCT_NAME,
               CATEGORY,
               PRICE,
               DESCRIPTION,
               STOCK,
               IMAGE_URL,
               STATUS
          FROM PRODUCT
         WHERE PRODUCT_ID = :product_id
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, {"product_id": product_id})
            columns = [col[0].lower() for col in cursor.description]
            row = cursor.fetchone()
            if row is None:
                return None
            record = dict(zip(columns, row))
            record["description"] = _read_lob(record.get("description"))
            return record


def _read_lob(value):
    if value is not None and hasattr(value, "read"):
        return value.read()
    return value


def _escape_like(value: str) -> str:
    """LIKE 패턴 메타문자(\\, %, _)를 이스케이프한다(ESCAPE '\\' 절과 함께 사용).

    사용자 키워드에 % 나 _ 가 들어오면 의도치 않은 전체/한글자 와일드카드 매칭이
    되므로 리터럴로 취급한다. SQL 자체는 바인드 변수라 인젝션과는 무관하며,
    이는 '검색 정확성' 방어다.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_STRUCTURED_KEYWORD_MAX = 5

_STRUCTURED_SORT_MAP = {
    "PRICE_ASC":  "PRICE ASC",
    "PRICE_DESC": "PRICE DESC",
    "LATEST":     "PRODUCT_ID DESC",
    "DEFAULT":    "PRODUCT_ID ASC",
}
_DEFAULT_ORDER_BY = "PRODUCT_ID ASC"

# LLM 이 생성할 수 있는 흔한 동의어 별칭. 화이트리스트(_STRUCTURED_SORT_MAP)
# 밖의 임의 문자열은 여전히 매칭되지 않고 기본값으로 폴백한다(주입 방어 유지).
_SORT_KEY_ALIASES = {
    "NEWEST": "LATEST",
}


def _to_number(value):
    if value is None:
        return None
    f = float(value)
    return int(f) if f.is_integer() else f


def _resolve_order_by(sort_by) -> str:
    """sort_by → 실제 ORDER BY 절 화이트리스트 매핑.

    Agent 경로의 sort_by 는 LLM 이 채우는 plain str 이라(SortType enum 강제 불가)
    대소문자 정규화 + 동의어 별칭으로 매칭 실패를 줄인다. 화이트리스트 밖의 값은
    기본 정렬로 폴백해 ORDER BY 주입을 방어한다.
    """
    if sort_by is None:
        return _DEFAULT_ORDER_BY
    key = sort_by.value if hasattr(sort_by, "value") else str(sort_by)
    key = key.strip().upper()
    key = _SORT_KEY_ALIASES.get(key, key)
    return _STRUCTURED_SORT_MAP.get(key, _DEFAULT_ORDER_BY)


def build_structured_query(
    category: str | None = None,
    price_min=None,
    price_max=None,
    keywords: list[str] | None = None,
    sort_by=None,
    limit: int = 5,
) -> tuple[str, dict]:
    # [판매중단 제외] STATUS='ACTIVE'가 아닌 상품(DISCONTINUED 등)은 챗봇 조건검색
    # 결과에서 항상 제외한다 — 관리자가 "판매중단" 처리한 상품이 계속 추천되는 것 방지.
    where_clauses: list[str] = ["STATUS = 'ACTIVE'"]
    binds: dict = {}

    if category:
        where_clauses.append("CATEGORY = :category")
        binds["category"] = category

    if price_min is not None:
        where_clauses.append("PRICE >= :price_min")
        binds["price_min"] = price_min
    if price_max is not None:
        where_clauses.append("PRICE <= :price_max")
        binds["price_max"] = price_max

    if isinstance(keywords, str):
        keywords = [keywords]
    cleaned = [kw.strip() for kw in keywords if kw and kw.strip()] if keywords else []
    cleaned = cleaned[:_STRUCTURED_KEYWORD_MAX]
    # [버그 수정] LLM이 category("상의")를 keywords에도 중복 추출하는 경우,
    # PRODUCT_NAME에는 카테고리명 자체가 등장하지 않아 AND 결합 시 결과가
    # 0건으로 사라진다("5만원 이하 상의" → 카테고리+가격은 맞는데 0건).
    # category와 정확히 같은 키워드만 제거(부분/포함 매칭은 다른 키워드를
    # 잘못 지울 위험이 있어 넣지 않음).
    if category:
        cleaned = [kw for kw in cleaned if kw != category]
    if cleaned:
        kw_conditions = []
        for i, kw in enumerate(cleaned):
            bind_name = f"kw{i}"
            binds[bind_name] = f"%{_escape_like(kw)}%"
            kw_conditions.append(f"PRODUCT_NAME LIKE :{bind_name} ESCAPE '\\'")
        where_clauses.append("(" + " OR ".join(kw_conditions) + ")")

    order_by = _resolve_order_by(sort_by)

    sql = (
        "SELECT PRODUCT_ID, PRODUCT_NAME, CATEGORY, PRICE, DESCRIPTION, STOCK, IMAGE_URL "
        "FROM PRODUCT"
    )
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    sql += f" ORDER BY {order_by}"
    sql += " FETCH FIRST :limit_rows ROWS ONLY"
    binds["limit_rows"] = limit

    return sql, binds


def search_products_structured(
    category: str | None = None,
    price_min=None,
    price_max=None,
    keywords: list[str] | None = None,
    sort_by=None,
    limit: int = 5,
) -> list[dict]:
    sql, binds = build_structured_query(
        category=category,
        price_min=price_min,
        price_max=price_max,
        keywords=keywords,
        sort_by=sort_by,
        limit=limit,
    )

    products: list[dict] = []
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, binds)
            columns = [col[0].lower() for col in cursor.description]
            rows = cursor.fetchall()
            for row in rows:
                record = dict(zip(columns, row))
                record["description"] = _read_lob(record.get("description"))
                record["product_id"] = _to_number(record.get("product_id"))
                record["price"] = _to_number(record.get("price"))
                record["stock"] = _to_number(record.get("stock"))
                products.append(record)
    return products


def fetch_faq_list(category: str | None = None) -> list[dict]:
    sql = """
        SELECT FAQ_ID, QUESTION, ANSWER, CATEGORY
          FROM FAQ
    """
    binds: dict = {}
    if category:
        sql += " WHERE CATEGORY = :category"
        binds["category"] = category
    sql += " ORDER BY FAQ_ID"

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, binds)
            rows = cursor.fetchall()
            return [
                {
                    "faq_id": int(row[0]),
                    "question": row[1],
                    "answer": _read_lob(row[2]),
                    "category": row[3],
                }
                for row in rows
            ]


_FAQ_KEYWORD_MAX = 10


def search_faq_by_keywords(keywords: list[str], limit: int = 3) -> list[dict]:
    cleaned = [kw.strip() for kw in keywords if kw and kw.strip()]
    cleaned = cleaned[:_FAQ_KEYWORD_MAX]
    if not cleaned:
        return []

    like_conditions = []
    score_terms = []
    binds: dict = {}
    for i, kw in enumerate(cleaned):
        bind_name = f"kw{i}"
        binds[bind_name] = f"%{_escape_like(kw)}%"
        like_conditions.append(
            f"(QUESTION LIKE :{bind_name} ESCAPE '\\' "
            f"OR ANSWER LIKE :{bind_name} ESCAPE '\\')"
        )
        score_terms.append(
            f"(CASE WHEN QUESTION LIKE :{bind_name} ESCAPE '\\' THEN 2 "
            f"WHEN ANSWER LIKE :{bind_name} ESCAPE '\\' THEN 1 ELSE 0 END)"
        )

    binds["limit_rows"] = limit
    sql = f"""
        SELECT FAQ_ID, QUESTION, ANSWER, CATEGORY
          FROM FAQ
         WHERE {" OR ".join(like_conditions)}
         ORDER BY {" + ".join(score_terms)} DESC, FAQ_ID
         FETCH FIRST :limit_rows ROWS ONLY
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, binds)
            rows = cursor.fetchall()
            return [
                {
                    "faq_id": int(row[0]),
                    "question": row[1],
                    "answer": _read_lob(row[2]),
                    "category": row[3],
                }
                for row in rows
            ]


def fetch_purchase_history(member_id: int, limit: int = 50) -> list[dict]:
    # [수정] 취소/환불된 주문은 "실제 구매 취향"이 아니므로 개인화 벡터 계산에서 제외.
    # 취향과 달라 반품한 상품이 오히려 추천 가중치를 높이는 왜곡을 방지한다.
    # (ORDER_STATUS 는 영문 enum CHECK 제약 — _ORDER_STATUS_KO 매핑과 동일 체계)
    sql = """
        SELECT oi.PRODUCT_ID,
               oi.QUANTITY,
               o.ORDER_DATE
          FROM ORDER_ITEM oi
          JOIN ORDERS o ON oi.ORDER_ID = o.ORDER_ID
         WHERE o.MEMBER_ID = :member_id
           AND o.ORDER_STATUS NOT IN ('CANCELLED', 'REFUNDED')
         ORDER BY o.ORDER_DATE DESC
         FETCH FIRST :limit_rows ROWS ONLY
    """
    binds = {"member_id": member_id, "limit_rows": limit}
    history: list[dict] = []
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, binds)
            rows = cursor.fetchall()
            for row in rows:
                pid = _to_number(row[0])
                qty = _to_number(row[1])
                history.append({
                    "product_id": int(pid) if pid is not None else None,
                    "quantity": int(qty) if qty is not None else 1,
                    "order_date": row[2],
                })
    return history


_ORDERS_BASE_SQL = """
    SELECT o.ORDER_ID,
           o.ORDER_DATE,
           o.ORDER_STATUS,
           o.TOTAL_PRICE,
           p.PRODUCT_NAME,
           oi.QUANTITY,
           oi.PRICE
      FROM ORDERS o
      LEFT JOIN ORDER_ITEM oi ON o.ORDER_ID = oi.ORDER_ID
      LEFT JOIN PRODUCT p     ON oi.PRODUCT_ID = p.PRODUCT_ID
     WHERE o.MEMBER_ID = :member_id
"""


# ORDERS.ORDER_STATUS 는 영문 enum(CHECK 제약: PENDING/PAID/…/REFUNDED)으로 저장된다.
# 화면 표시는 한글이라 어댑터 단계에서 한글 라벨로 변환한다(Mock/레거시 order_handler 와
# 동일하게 '한글 status' dict 를 생산 → _format_order_* 와 이모지 맵이 그대로 매칭됨).
_ORDER_STATUS_KO = {
    "PENDING":   "결제대기",
    "PAID":      "결제완료",
    "PREPARING": "배송준비중",
    "SHIPPED":   "배송중",
    "DELIVERED": "배송완료",
    "CANCELLED": "주문취소",
    "REFUNDED":  "환불완료",
}


def _to_status_label(raw) -> str:
    """ORDER_STATUS 원본값 → 한글 라벨.

    영문 enum(SHIPPED 등)은 한글로 변환한다. 매핑에 없는 값(레거시 데이터·한글 등)은
    원본을 그대로 유지해 기존 동작/테스트와 호환시키고, None·빈값만 '상태 미확인' 폴백.
    """
    if raw is None:
        return "상태 미확인"
    key = str(raw).strip()
    if not key:
        return "상태 미확인"
    return _ORDER_STATUS_KO.get(key.upper(), key)


def _build_orders_from_rows(rows) -> list[dict]:
    orders_map: dict = {}
    order_seq: list[dict] = []
    for r in rows:
        oid = r[0]
        if oid not in orders_map:
            order_date = r[1]
            o = {
                "order_id": str(int(oid)) if oid is not None else "",
                "order_date": order_date.strftime("%Y-%m-%d") if order_date else "",
                "status": _to_status_label(r[2]),
                "items": [],
                "total_price": int(r[3]) if r[3] is not None else None,
            }
            orders_map[oid] = o
            order_seq.append(o)
        product_name = r[4]
        if product_name is not None:
            qty = _to_number(r[5])
            price = _to_number(r[6])
            orders_map[oid]["items"].append({
                "product_name": product_name,
                "quantity": int(qty) if qty is not None else 1,
                "price": int(price) if price is not None else 0,
            })
    for o in order_seq:
        if o["total_price"] is None:
            o["total_price"] = sum(i["price"] * i["quantity"] for i in o["items"])
    return order_seq


def fetch_orders(member_id: int) -> list[dict]:
    sql = _ORDERS_BASE_SQL + " ORDER BY o.ORDER_DATE DESC, o.ORDER_ID DESC, oi.ID ASC"
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, {"member_id": member_id})
            rows = cursor.fetchall()
    return _build_orders_from_rows(rows)


def fetch_order_by_id(member_id: int, order_id) -> dict | None:
    try:
        oid = int(str(order_id).strip())
    except (ValueError, TypeError, AttributeError):
        return None
    sql = _ORDERS_BASE_SQL + " AND o.ORDER_ID = :order_id ORDER BY oi.ID ASC"
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, {"member_id": member_id, "order_id": oid})
            rows = cursor.fetchall()
    orders = _build_orders_from_rows(rows)
    return orders[0] if orders else None


def resolve_chat_token(chat_token: str) -> int | None:
    sql = """
        SELECT MEMBER_ID
          FROM CHAT_TOKEN
         WHERE CHAT_TOKEN = :chat_token
           AND EXPIRE_DATE > SYSDATE
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, {"chat_token": chat_token})
            row = cursor.fetchone()
    return int(row[0]) if row is not None else None


_QUESTION_MAX_BYTES = 4000


def save_chat_history(member_id: int, question: str, answer: str, chat_type: str) -> None:
    encoded = question.encode("utf-8")
    if len(encoded) > _QUESTION_MAX_BYTES:
        question = encoded[:_QUESTION_MAX_BYTES].decode("utf-8", errors="ignore")

    sql = """
        INSERT INTO CHAT_HISTORY
            (CHAT_ID, MEMBER_ID, QUESTION, ANSWER, CHAT_TYPE, CHAT_DATE)
        VALUES
            (CHAT_HISTORY_SEQ.NEXTVAL, :member_id, :question, :answer, :chat_type, SYSTIMESTAMP)
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                sql,
                {
                    "member_id": member_id,
                    "question": question,
                    "answer": answer,
                    "chat_type": chat_type,
                },
            )
        conn.commit()
