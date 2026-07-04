--------------------------------------------------------------------------------
-- migration_phase2.sql
-- PHASE 2 (멀티모달 + 개인화) 대비 스키마 보강 — 기존 운영 DB 적용용
--------------------------------------------------------------------------------
-- 대상: 이미 schema.sql 로 구축되어 데이터가 들어있는 운영 DB
--      (신규 환경은 갱신된 schema.sql 한 방이면 되므로 이 파일 불필요)
--
-- 안전성: 모든 변경을 PL/SQL EXCEPTION 으로 감싸 '이미 적용된 경우 조용히 통과'
--        하도록 멱등(idempotent) 처리. 여러 번 돌려도 안전하다.
--
-- 변경 요약
--   1) ORDER_ITEM."COUNT" → QUANTITY 리네임  (예약어 충돌 제거 / 개인화 집계 안전)
--   2) PRODUCT.IMAGE_URL  VARCHAR2(500) → VARCHAR2(1000)  (긴 이미지 URL 대비)
--   3) 인덱스 보강
--        · CHAT_TOKEN(CHAT_TOKEN) UNIQUE — 매 챗 요청 토큰 조회 풀스캔 제거
--        · ORDERS(MEMBER_ID)             — 개인화 구매이력 조회
--        · ORDER_ITEM(ORDER_ID)          — 개인화 조인
--        · ORDER_ITEM(PRODUCT_ID)        — 개인화 조인/집계
--        · CART(MEMBER_ID), CART(PRODUCT_ID)        — FK 인덱스(부모 DML 락 방지/조회)
--        · CHAT_HISTORY(MEMBER_ID, CHAT_DATE)       — 향후 이력 조회/분석
--
-- ⚠️ Spring Boot 영향(1) 리네임 후 OrderItem 엔티티를 함께 고쳐야 한다.
--    @Column(name = "COUNT")  →  @Column(name = "QUANTITY")
--    (또는 필드명을 quantity 로 두고 매핑) → 빌드 후 재배포.
--------------------------------------------------------------------------------


-- 1) ORDER_ITEM 수량 컬럼 리네임: "COUNT" → QUANTITY ---------------------------
--    이미 QUANTITY 면 ORA-00957(중복 컬럼) 또는 ORA-00904 → 무시.
DECLARE
    e_already EXCEPTION;
    PRAGMA EXCEPTION_INIT(e_already, -957);   -- duplicate column name
    e_no_col  EXCEPTION;
    PRAGMA EXCEPTION_INIT(e_no_col, -904);    -- invalid identifier(이미 리네임됨)
BEGIN
    EXECUTE IMMEDIATE 'ALTER TABLE ORDER_ITEM RENAME COLUMN "COUNT" TO QUANTITY';
    DBMS_OUTPUT.PUT_LINE('ORDER_ITEM.COUNT -> QUANTITY 리네임 완료');
EXCEPTION
    WHEN e_already OR e_no_col THEN
        DBMS_OUTPUT.PUT_LINE('ORDER_ITEM.QUANTITY 이미 적용됨 — 스킵');
    WHEN OTHERS THEN
        -- 그 외 오류는 드러내되 스크립트는 계속 진행
        DBMS_OUTPUT.PUT_LINE('ORDER_ITEM 리네임 경고: ' || SQLERRM);
END;
/


-- 2) PRODUCT.IMAGE_URL 길이 확장: 500 → 1000 ----------------------------------
--    동일 길이로 재실행해도 무해. 축소가 아니므로 데이터 손실 없음.
DECLARE
BEGIN
    EXECUTE IMMEDIATE 'ALTER TABLE PRODUCT MODIFY (IMAGE_URL VARCHAR2(1000))';
    DBMS_OUTPUT.PUT_LINE('PRODUCT.IMAGE_URL VARCHAR2(1000) 적용 완료');
EXCEPTION
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('IMAGE_URL MODIFY 경고: ' || SQLERRM);
END;
/


-- 3) 인덱스 보강 ---------------------------------------------------------------
--    이미 존재(ORA-00955)면 스킵. 한 블록에서 7개를 개별 EXCEPTION 처리.
DECLARE
    e_exists EXCEPTION;
    PRAGMA EXCEPTION_INIT(e_exists, -955);    -- name is already used by an existing object

    PROCEDURE mk(p_sql VARCHAR2, p_name VARCHAR2) IS
    BEGIN
        EXECUTE IMMEDIATE p_sql;
        DBMS_OUTPUT.PUT_LINE('인덱스 생성: ' || p_name);
    EXCEPTION
        WHEN e_exists THEN
            DBMS_OUTPUT.PUT_LINE('인덱스 이미 존재 — 스킵: ' || p_name);
        WHEN OTHERS THEN
            -- 중복 컬럼 인덱스(ORA-01408) 등도 조용히 통과
            DBMS_OUTPUT.PUT_LINE('인덱스 경고(' || p_name || '): ' || SQLERRM);
    END;
BEGIN
    -- 필수
    mk('CREATE UNIQUE INDEX IX_CHAT_TOKEN_TOKEN ON CHAT_TOKEN (CHAT_TOKEN)', 'IX_CHAT_TOKEN_TOKEN');
    mk('CREATE INDEX IX_ORDERS_MEMBER       ON ORDERS (MEMBER_ID)',          'IX_ORDERS_MEMBER');
    mk('CREATE INDEX IX_ORDER_ITEM_ORDER    ON ORDER_ITEM (ORDER_ID)',       'IX_ORDER_ITEM_ORDER');
    mk('CREATE INDEX IX_ORDER_ITEM_PRODUCT  ON ORDER_ITEM (PRODUCT_ID)',     'IX_ORDER_ITEM_PRODUCT');
    -- 권장(FK 인덱스/조회 최적화)
    mk('CREATE INDEX IX_CART_MEMBER         ON CART (MEMBER_ID)',            'IX_CART_MEMBER');
    mk('CREATE INDEX IX_CART_PRODUCT        ON CART (PRODUCT_ID)',           'IX_CART_PRODUCT');
    mk('CREATE INDEX IX_CHAT_HISTORY_MEMBER ON CHAT_HISTORY (MEMBER_ID, CHAT_DATE)', 'IX_CHAT_HISTORY_MEMBER');
END;
/

COMMIT;

--------------------------------------------------------------------------------
-- 적용 후 확인(선택)
--   SELECT COLUMN_NAME FROM USER_TAB_COLUMNS WHERE TABLE_NAME='ORDER_ITEM' ORDER BY COLUMN_ID;
--   SELECT INDEX_NAME, TABLE_NAME, UNIQUENESS FROM USER_INDEXES
--    WHERE INDEX_NAME LIKE 'IX_%' ORDER BY TABLE_NAME, INDEX_NAME;
--------------------------------------------------------------------------------
