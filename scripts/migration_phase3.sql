--==============================================================================
-- migration_phase3.sql — 상품 물리 삭제 지원
--
-- [배경]
-- 관리자가 상품을 삭제할 때 CART/ORDER_ITEM 에 그 상품을 참조하는 행이 있으면
-- FK 제약(ORA-02292) 위반으로 500 에러가 났다(AdminProductController.delete()에
-- try-catch가 없어 그대로 노출). 한 번은 "판매중단(소프트 삭제, STATUS 전환)"으로
-- 우회했었으나, 실제 삭제 기능을 요구해 이번에 정식으로 지원한다.
--
-- [설계]
-- - ORDER_ITEM 은 이미 PRICE(주문 시점 단가)를 스냅샷으로 갖고 있었다. 같은 원리로
--   PRODUCT_NAME 도 스냅샷으로 추가한다 — 상품이 삭제돼도 과거 주문 내역의
--   상품명이 사라지지 않는다.
-- - CART 는 이력 보존이 필요 없으므로 상품 삭제 시 CASCADE(같이 삭제).
-- - ORDER_ITEM 은 주문 이력 자체는 보존해야 하므로 SET NULL(PRODUCT_ID 만 비워짐,
--   PRODUCT_NAME/PRICE 스냅샷은 그대로 남음).
--
-- 실행 순서 중요 — 반드시 위에서부터 순서대로 실행할 것.
--==============================================================================

-- 1) 상품명 스냅샷 컬럼 추가
ALTER TABLE ORDER_ITEM ADD PRODUCT_NAME VARCHAR2(200);

-- 2) 과거 주문 데이터 소급 채우기 (1회성 — 이미 있는 주문 건들의 상품명을 지금
--    시점의 PRODUCT.PRODUCT_NAME 으로 채운다. 이후 신규 주문은 OrdersService 가
--    주문 생성 시점에 직접 저장한다.)
UPDATE ORDER_ITEM oi
   SET PRODUCT_NAME = (
       SELECT p.PRODUCT_NAME FROM PRODUCT p WHERE p.PRODUCT_ID = oi.PRODUCT_ID
   )
 WHERE PRODUCT_NAME IS NULL;
COMMIT;

-- 3) PRODUCT_ID NULL 허용 (삭제 시 SET NULL 을 받으려면 NOT NULL 이면 안 됨)
ALTER TABLE ORDER_ITEM MODIFY PRODUCT_ID NULL;

-- 4) ORDER_ITEM 쪽 FK 재정의 — 삭제 시 SET NULL(주문 항목 자체·스냅샷은 보존)
ALTER TABLE ORDER_ITEM DROP CONSTRAINT FK_ORDER_ITEM_PRODUCT;
ALTER TABLE ORDER_ITEM ADD CONSTRAINT FK_ORDER_ITEM_PRODUCT
    FOREIGN KEY (PRODUCT_ID) REFERENCES PRODUCT (PRODUCT_ID) ON DELETE SET NULL;

-- 5) CART 쪽 FK 재정의 — 삭제 시 CASCADE(장바구니는 이력 보존 불필요)
ALTER TABLE CART DROP CONSTRAINT FK_CART_PRODUCT;
ALTER TABLE CART ADD CONSTRAINT FK_CART_PRODUCT
    FOREIGN KEY (PRODUCT_ID) REFERENCES PRODUCT (PRODUCT_ID) ON DELETE CASCADE;
