--------------------------------------------------------------------------------
-- schema.sql  ·  쇼핑몰 AI 챗봇 프로젝트 — 신규 환경 전체 DDL
-- Oracle 23ai (Autonomous Database 호환)
--
-- ┌──────────────────────────────────────────────────────────────────────────┐
-- │  테이블 구성 & AI 파이프라인 연계 개요                                   │
-- │                                                                          │
-- │  [회원/인증]                                                             │
-- │    MEMBER ──────── 쇼핑몰 회원 (Spring Security 인증 주체)              │
-- │    EMAIL_VERIFY ── 이메일 인증코드 (회원가입 본인확인)                   │
-- │    CHAT_TOKEN ──── AI 챗봇 세션 토큰 (Spring ↔ FastAPI 식별 브릿지)    │
-- │                    · Spring 로그인 시 UUID 발급 → FastAPI 가 역조회     │
-- │                    · resolve_chat_token() 가 CHAT_TOKEN → MEMBER_ID 변환│
-- │                    · IX_CHAT_TOKEN_TOKEN(UNIQUE) 로 매 요청 풀스캔 제거 │
-- │                                                                          │
-- │  [상품]                                                                  │
-- │    PRODUCT ─────── 상품 카탈로그 (Oracle DB + ChromaDB 이중 관리)       │
-- │                    · DESCRIPTION 이 ChromaDB 텍스트 임베딩 소스         │
-- │                    · IMAGE_URL 이 CLIP 이미지 임베딩 소스 (PHASE 2)     │
-- │                    · STRUCTURED_QUERY 인텐트: CATEGORY/PRICE/STOCK 컬럼 │
-- │                    · SEMANTIC_SEARCH 인텐트: ChromaDB 벡터 → 여기서 JOIN│
-- │                    · 개인화(PHASE 2): ORDER_ITEM 집계 → 취향벡터 생성  │
-- │                                                                          │
-- │  [쇼핑]                                                                  │
-- │    CART ────────── 장바구니                                              │
-- │    ORDERS ──────── 주문 헤더                                             │
-- │    ORDER_ITEM ──── 주문 상세 (ORDER_INQUIRY 인텐트 + 개인화 집계 대상) │
-- │                    · fetch_orders() / fetch_order_by_id(): 실DB 조회    │
-- │                    · 개인화: QUANTITY × 최신성감쇠 가중합 → 취향벡터    │
-- │                                                                          │
-- │  [AI 챗봇]                                                               │
-- │    FAQ ─────────── FAQ DB (FAQ 인텐트 핸들러 검색 대상)                 │
-- │    CHAT_HISTORY ── 대화 이력 (로그인 회원 전용 영속 저장)               │
-- │                    · CHAT_TYPE: 9종 인텐트/경로 식별자 (CHECK 제약)     │
-- │                      ┌ 라우터 그래프 : STRUCTURED_QUERY, SEMANTIC_SEARCH│
-- │                      │                FAQ, ORDER_INQUIRY                 │
-- │                      │                COMPLAINT, SMALL_TALK             │
-- │                      └ Agent 경로   : AGENT, MULTI_AGENT, MCP_AGENT    │
-- │                    · PHASE 3: 경로별 비교 분석 기반 데이터              │
-- │                                                                          │
-- │  [인덱스 설계 원칙]                                                      │
-- │    Oracle 은 FK 에 인덱스를 자동 생성하지 않는다. 부모 테이블 DML 시   │
-- │    자식 테이블 풀스캔 락을 방지하기 위해 FK 컬럼을 명시적으로 인덱싱.  │
-- │    AI 파이프라인 고빈도 쿼리(토큰 역조회, 개인화 조인, 카테고리 필터)  │
-- │    도 함께 커버한다.                                                     │
-- └──────────────────────────────────────────────────────────────────────────┘
--
-- 실행 순서
--   1) 본 스크립트 전체 실행 (테이블 + 인덱스 + 시드 데이터 한 번에)
--   2) Spring Boot 기동 → AdminAccountInitializer 가 admin/admin1234 자동 INSERT
--      (관리자 비밀번호는 BCrypt 해시 → 평문 DDL 에 노출하지 않음)
--   3) FastAPI: python -m scripts.index_products → ChromaDB 22개 전체 색인
--      (CLIP 이미지 검색 활성 시: python -m scripts.index_products_image 추가 실행)
--
-- 주의
--   · 기존 운영 DB 에는 이 파일 대신 scripts/migration_phase2.sql 로 증분 적용.
--   · '&' 포함 SQL 실행 전 반드시: SET DEFINE OFF
--   · Spring Boot 엔티티 @Column 매핑 주의 (주석 참고):
--       Product.name          → PRODUCT_NAME
--       Product.stockQuantity → STOCK
--       OrderItem.quantity    → QUANTITY  (PHASE 2: COUNT 예약어 → 리네임)
--------------------------------------------------------------------------------


--==============================================================================
-- 0. 기존 객체 정리 — FK 역순 DROP (재실행 안전, 없으면 조용히 통과)
--==============================================================================
BEGIN EXECUTE IMMEDIATE 'DROP TABLE ORDER_ITEM   CASCADE CONSTRAINTS'; EXCEPTION WHEN OTHERS THEN NULL; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP TABLE CHAT_HISTORY CASCADE CONSTRAINTS'; EXCEPTION WHEN OTHERS THEN NULL; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP TABLE ORDERS       CASCADE CONSTRAINTS'; EXCEPTION WHEN OTHERS THEN NULL; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP TABLE CART         CASCADE CONSTRAINTS'; EXCEPTION WHEN OTHERS THEN NULL; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP TABLE CHAT_TOKEN   CASCADE CONSTRAINTS'; EXCEPTION WHEN OTHERS THEN NULL; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP TABLE EMAIL_VERIFY CASCADE CONSTRAINTS'; EXCEPTION WHEN OTHERS THEN NULL; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP TABLE FAQ          CASCADE CONSTRAINTS'; EXCEPTION WHEN OTHERS THEN NULL; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP TABLE PRODUCT      CASCADE CONSTRAINTS'; EXCEPTION WHEN OTHERS THEN NULL; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP TABLE MEMBER       CASCADE CONSTRAINTS'; EXCEPTION WHEN OTHERS THEN NULL; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP SEQUENCE MEMBER_SEQ';       EXCEPTION WHEN OTHERS THEN NULL; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP SEQUENCE PRODUCT_SEQ';      EXCEPTION WHEN OTHERS THEN NULL; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP SEQUENCE ORDER_SEQ';        EXCEPTION WHEN OTHERS THEN NULL; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP SEQUENCE CHAT_HISTORY_SEQ'; EXCEPTION WHEN OTHERS THEN NULL; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP SEQUENCE FAQ_SEQ';          EXCEPTION WHEN OTHERS THEN NULL; END;
/


--==============================================================================
-- 1. 시퀀스 (Spring Boot allocationSize=1 과 일치하도록 INCREMENT BY 1)
--==============================================================================
CREATE SEQUENCE MEMBER_SEQ       START WITH 1 INCREMENT BY 1 NOCACHE NOORDER;
CREATE SEQUENCE PRODUCT_SEQ      START WITH 1 INCREMENT BY 1 NOCACHE NOORDER;
CREATE SEQUENCE ORDER_SEQ        START WITH 1 INCREMENT BY 1 NOCACHE NOORDER;
CREATE SEQUENCE CHAT_HISTORY_SEQ START WITH 1 INCREMENT BY 1 NOCACHE NOORDER;
CREATE SEQUENCE FAQ_SEQ          START WITH 1 INCREMENT BY 1 NOCACHE NOORDER;


--==============================================================================
-- 2. MEMBER — 쇼핑몰 회원 (Spring Security 인증 주체)
--
-- [AI 연계]
--   · 로그인 시 CHAT_TOKEN 이 발급돼 FastAPI 와 회원 식별 브릿지 역할을 한다.
--   · CHAT_HISTORY / ORDERS 의 MEMBER_ID FK 기준 → 개인화·대화이력 조회.
--
-- [컬럼 설계 의도]
--   · ROLE   : Spring Security GrantedAuthority. 'USER'/'ADMIN' 두 값만 허용.
--   · STATUS : 계정 상태. INACTIVE(자발 비활성) / SUSPENDED(관리자 제재) 구분.
--   · VERIFIED: CHAR(1) — 'Y'/'N' 이진 플래그, 이메일 인증 완료 여부.
--   · CI_VALUE: 본인인증 연계 식별자 예약 컬럼 (현재 미사용, 향후 CI 연동용).
--   · CREATED_AT/UPDATED_AT: Spring @CreationTimestamp/@UpdateTimestamp 가 관리.
--==============================================================================
CREATE TABLE MEMBER (
    MEMBER_ID   NUMBER         NOT NULL,
    LOGIN_ID    VARCHAR2(50)   NOT NULL,
    PASSWORD    VARCHAR2(255)  NOT NULL,  -- BCrypt 해시 (평문 저장 금지)
    NAME        VARCHAR2(50)   NOT NULL,
    GENDER      VARCHAR2(10),             -- 'M' / 'F' / 'OTHER' (Spring 에서 검증)
    BIRTH_DATE  DATE,
    EMAIL       VARCHAR2(100)  NOT NULL,
    PHONE       VARCHAR2(20),
    ADDRESS     VARCHAR2(255),
    ROLE        VARCHAR2(20)   DEFAULT 'USER'   NOT NULL,
    STATUS      VARCHAR2(10)   DEFAULT 'ACTIVE'   NOT NULL,
    CI_VALUE    VARCHAR2(100),
    VERIFIED    CHAR(1)        DEFAULT 'N'   NOT NULL,
    CREATED_AT  TIMESTAMP,
    UPDATED_AT  TIMESTAMP,
    --
    CONSTRAINT PK_MEMBER         PRIMARY KEY (MEMBER_ID),
    CONSTRAINT UK_MEMBER_LOGIN   UNIQUE      (LOGIN_ID),
    CONSTRAINT UK_MEMBER_EMAIL   UNIQUE      (EMAIL),
    CONSTRAINT CK_MEMBER_ROLE    CHECK       (ROLE    IN ('USER', 'ADMIN')),
    CONSTRAINT CK_MEMBER_STATUS  CHECK       (STATUS  IN ('ACTIVE', 'INACTIVE', 'SUSPENDED')),
    CONSTRAINT CK_MEMBER_VERIFIED CHECK      (VERIFIED IN ('Y', 'N'))
);


--==============================================================================
-- 3. PRODUCT — 상품 카탈로그
--
-- [AI 연계 — 이 프로젝트에서 가장 중심이 되는 테이블]
--   · STRUCTURED_QUERY 인텐트: CATEGORY/PRICE/STOCK/PRODUCT_NAME 을 동적 SQL 로 검색.
--     (build_structured_query: 바인드변수 + ORDER BY 화이트리스트로 SQL Injection 차단)
--   · SEMANTIC_SEARCH 인텐트: DESCRIPTION 을 text-embedding-3-small 로 임베딩 →
--     ChromaDB(products 컬렉션, 1536차원)에 저장. 사용자 질문 벡터와 코사인 유사도.
--   · CLIP 이미지 검색(PHASE 2): IMAGE_URL 을 OpenCLIP(ViT-B-32, 512차원)으로 임베딩 →
--     ChromaDB(products_image 컬렉션). CLIP_SERVING_ENABLED=true 시 활성.
--   · 개인화 추천(PHASE 2): ORDER_ITEM 집계로 구매 상품 임베딩 가중합산 →
--     취향벡터(personalization_service). PERSONALIZATION_ENABLED=true 시 활성.
--
-- [컬럼 설계 의도]
--   · PRODUCT_NAME: Spring @Column(name="PRODUCT_NAME"), Java 필드명 name 과 다름.
--   · STOCK       : Spring @Column(name="STOCK"), Java 필드명 stockQuantity 와 다름.
--                   STOCK=0 + STATUS='SOLD_OUT' 이 품절 상태의 표준 조합.
--   · DESCRIPTION : ChromaDB 벡터 임베딩 소스. 풍부한 자연어 설명이 RAG 품질에 직결.
--   · IMAGE_URL   : VARCHAR2(1000) — 긴 CDN URL 대비(PHASE 2 확장, 기존 500→1000).
--   · STATUS      : 'ACTIVE'=판매중 / 'SOLD_OUT'=품절(재고=0) / 'DISCONTINUED'=판매종료.
--                   AI 파이프라인은 STATUS 무관 전체를 검색해 상태를 답변에 포함한다.
--==============================================================================
CREATE TABLE PRODUCT (
    PRODUCT_ID    NUMBER         NOT NULL,
    PRODUCT_NAME  VARCHAR2(255)  NOT NULL,
    PRICE         NUMBER         NOT NULL,
    STOCK         NUMBER         DEFAULT 0   NOT NULL,
    DESCRIPTION   VARCHAR2(4000),
    CATEGORY      VARCHAR2(100),            -- '상의'·'하의'·'신발'·'아우터'·'전자기기'·'뷰티'
    CREATED_AT    TIMESTAMP               DEFAULT SYSTIMESTAMP,
    IMAGE_URL     VARCHAR2(1000),           -- PHASE 2: CLIP 임베딩 + 검색결과 카드 썸네일
    MANUFACTURER  VARCHAR2(255),
    STATUS        VARCHAR2(20)   DEFAULT 'ACTIVE'   NOT NULL,
    --
    CONSTRAINT PK_PRODUCT        PRIMARY KEY (PRODUCT_ID),
    CONSTRAINT CK_PRODUCT_PRICE  CHECK       (PRICE  > 0),
    CONSTRAINT CK_PRODUCT_STOCK  CHECK       (STOCK >= 0),
    CONSTRAINT CK_PRODUCT_STATUS CHECK       (STATUS IN ('ACTIVE', 'SOLD_OUT', 'DISCONTINUED'))
);


--==============================================================================
-- 4. FAQ — 자주 묻는 질문
--
-- [AI 연계]
--   · FAQ 인텐트: fetch_faq_list(category) 로 전체 또는 카테고리별 조회 →
--     질문 임베딩 유사도로 가장 관련 있는 FAQ 를 반환.
--   · 운영자가 Oracle SQL 또는 Spring Admin 화면에서 직접 관리.
--==============================================================================
CREATE TABLE FAQ (
    FAQ_ID    NUMBER          NOT NULL,
    QUESTION  VARCHAR2(1000)  NOT NULL,
    ANSWER    VARCHAR2(2000)  NOT NULL,
    CATEGORY  VARCHAR2(100)   NOT NULL,     -- '배송'·'교환'·'환불'·'회원'·'주문' 등
    --
    CONSTRAINT PK_FAQ PRIMARY KEY (FAQ_ID)
);


--==============================================================================
-- 5. EMAIL_VERIFY — 이메일 인증코드
--
-- [설계 의도]
--   · PK 가 EMAIL 이라 회원당 1개만 존재(재요청 시 upsert).
--   · LOGTIME 으로 만료 여부 판단 (Spring 에서 3분 유효 처리).
--==============================================================================
CREATE TABLE EMAIL_VERIFY (
    EMAIL     VARCHAR2(100)  NOT NULL,
    AUTHCODE  VARCHAR2(10)   NOT NULL,
    LOGTIME   DATE,
    --
    CONSTRAINT PK_EMAIL_VERIFY PRIMARY KEY (EMAIL)
);


--==============================================================================
-- 6. CHAT_TOKEN — AI 챗봇 세션 토큰 (Spring ↔ FastAPI 회원 식별 브릿지)
--
-- [AI 연계 — FastAPI 의 모든 인증 경로가 이 테이블을 거친다]
--   · Spring 로그인 시 UUID(v4) 발급 → 이 테이블에 upsert.
--   · FastAPI resolve_chat_token(token): CHAT_TOKEN 값으로 MEMBER_ID 역조회.
--     → IX_CHAT_TOKEN_TOKEN(UNIQUE) 가 매 챗 요청의 풀스캔을 방지한다.
--   · 조회된 MEMBER_ID 가 LangGraph State.member_id 로 주입 →
--     주문조회(ORDER_INQUIRY), 개인화(SEMANTIC_SEARCH), 이력 저장 등에 사용.
--   · EXPIRE_DATE 로 세션 만료 제어 (Spring 에서 갱신).
--
-- [설계 의도]
--   · PK = MEMBER_ID (회원당 토큰 1개 강제, 멀티 탭은 같은 토큰 공유).
--   · CHAT_TOKEN 컬럼에 별도 UNIQUE INDEX → 역조회 성능 + 유일성 이중 보장.
--==============================================================================
CREATE TABLE CHAT_TOKEN (
    MEMBER_ID    NUMBER        NOT NULL,
    CHAT_TOKEN   VARCHAR2(36)  NOT NULL,  -- UUID v4 (하이픈 포함 36자)
    EXPIRE_DATE  TIMESTAMP     NOT NULL,
    --
    CONSTRAINT PK_CHAT_TOKEN         PRIMARY KEY (MEMBER_ID),
    CONSTRAINT UK_CHAT_TOKEN_TOKEN   UNIQUE      (CHAT_TOKEN),
    CONSTRAINT FK_CHAT_TOKEN_MEMBER  FOREIGN KEY (MEMBER_ID) REFERENCES MEMBER (MEMBER_ID)
);
-- ↑ UNIQUE 제약이 내부적으로 인덱스를 생성하므로 별도 IX_CHAT_TOKEN_TOKEN 불필요.
--   UK_CHAT_TOKEN_TOKEN 이 인덱스 역할을 겸한다.


--==============================================================================
-- 7. CART — 장바구니
--==============================================================================
CREATE TABLE CART (
    ID          NUMBER GENERATED BY DEFAULT AS IDENTITY,
    MEMBER_ID   NUMBER   NOT NULL,
    PRODUCT_ID  NUMBER   NOT NULL,
    QUANTITY    NUMBER   DEFAULT 1   NOT NULL,
    CREATED_AT  TIMESTAMP          DEFAULT SYSTIMESTAMP,
    --
    CONSTRAINT PK_CART         PRIMARY KEY (ID),
    -- [동시성] 같은 회원·같은 상품은 1행 보장 — CartService.addCart 의 확인→병합이
    -- 동시 요청에서 중복 행을 만들면 이후 단건 조회(findByMemberIdAndProductId)가
    -- NonUniqueResultException 으로 영구 고장나는 것을 DB 레벨에서 차단한다.
    CONSTRAINT UK_CART_MEMBER_PRODUCT UNIQUE (MEMBER_ID, PRODUCT_ID),
    CONSTRAINT CK_CART_QTY     CHECK       (QUANTITY > 0),
    CONSTRAINT FK_CART_MEMBER  FOREIGN KEY (MEMBER_ID)  REFERENCES MEMBER  (MEMBER_ID),
    CONSTRAINT FK_CART_PRODUCT FOREIGN KEY (PRODUCT_ID) REFERENCES PRODUCT (PRODUCT_ID)
);


--==============================================================================
-- 8. ORDERS — 주문 헤더
--
-- [AI 연계]
--   · ORDER_INQUIRY 인텐트: fetch_orders(member_id) 가 ORDERS ⋈ ORDER_ITEM ⋈ PRODUCT
--     단일 JOIN 으로 조회 (N+1 쿼리 방지). IX_ORDERS_MEMBER 가 풀스캔 차단.
--   · 개인화(PHASE 2): 구매 상품 임베딩 가중합산에 ORDER_DATE(최신성 감쇠) 사용.
--     weight = quantity × 0.5^(elapsed_days / halflife)
--
-- [컬럼 설계 의도]
--   · ORDER_STATUS: 주문 라이프사이클 7단계. AI 챗봇이 상태 문자열을 그대로 답변에 포함.
--   · TOTAL_PRICE : 주문 시점 금액 스냅샷 (이후 가격 변동과 무관하게 보존).
--==============================================================================
CREATE TABLE ORDERS (
    ORDER_ID      NUMBER        NOT NULL,
    MEMBER_ID     NUMBER        NOT NULL,
    TOTAL_PRICE   NUMBER        NOT NULL,
    ORDER_STATUS  VARCHAR2(20)  DEFAULT 'PENDING'   NOT NULL,
    ORDER_DATE    TIMESTAMP               DEFAULT SYSTIMESTAMP,
    --
    CONSTRAINT PK_ORDERS           PRIMARY KEY (ORDER_ID),
    CONSTRAINT CK_ORDERS_PRICE     CHECK       (TOTAL_PRICE >= 0),
    CONSTRAINT CK_ORDERS_STATUS    CHECK       (ORDER_STATUS IN (
                                       'PENDING',     -- 결제 대기
                                       'PAID',        -- 결제 완료
                                       'PREPARING',   -- 상품 준비중
                                       'SHIPPED',     -- 배송중
                                       'DELIVERED',   -- 배송 완료
                                       'CANCELLED',   -- 취소
                                       'REFUNDED'     -- 환불 완료
                                   )),
    CONSTRAINT FK_ORDERS_MEMBER    FOREIGN KEY (MEMBER_ID) REFERENCES MEMBER (MEMBER_ID)
);


--==============================================================================
-- 9. ORDER_ITEM — 주문 상세
--
-- [AI 연계]
--   · ORDER_INQUIRY: ORDERS ⋈ ORDER_ITEM ⋈ PRODUCT 조인의 핵심 교차 테이블.
--   · 개인화(PHASE 2): QUANTITY 와 주문일(ORDERS.ORDER_DATE)로 취향벡터 가중합산.
--
-- [컬럼 설계 의도]
--   · QUANTITY: [PHASE 2] 예약어 COUNT → QUANTITY 리네임. Spring @Column(name="QUANTITY").
--   · PRICE   : 주문 시점 상품 단가 스냅샷 (이후 상품 가격 변동과 분리).
--==============================================================================
CREATE TABLE ORDER_ITEM (
    ID          NUMBER GENERATED BY DEFAULT AS IDENTITY,
    ORDER_ID    NUMBER   NOT NULL,
    PRODUCT_ID  NUMBER   NOT NULL,
    QUANTITY    NUMBER   DEFAULT 1   NOT NULL,
    PRICE       NUMBER   NOT NULL,
    --
    CONSTRAINT PK_ORDER_ITEM         PRIMARY KEY (ID),
    CONSTRAINT CK_ORDER_ITEM_QTY     CHECK       (QUANTITY > 0),
    CONSTRAINT CK_ORDER_ITEM_PRICE   CHECK       (PRICE   >= 0),
    CONSTRAINT FK_ORDER_ITEM_ORDER   FOREIGN KEY (ORDER_ID)   REFERENCES ORDERS  (ORDER_ID),
    CONSTRAINT FK_ORDER_ITEM_PRODUCT FOREIGN KEY (PRODUCT_ID) REFERENCES PRODUCT (PRODUCT_ID)
);


--==============================================================================
-- 10. CHAT_HISTORY — AI 챗봇 대화 이력 (로그인 회원 전용)
--
-- [이 프로젝트의 AI 파이프라인 전체가 이 테이블 한 곳에 기록된다]
--
-- [AI 연계]
--   · 로그인 회원의 모든 챗봇 대화를 MEMBER_ID + CHAT_DATE 로 적재.
--   · CHAT_TYPE: 어느 AI 경로가 처리했는지를 나타내는 9종 식별자 (CHECK 제약).
--
--     ┌─────────────────────┬────────────────────────────────────────────────┐
--     │ CHAT_TYPE 값        │ 의미                                           │
--     ├─────────────────────┼────────────────────────────────────────────────┤
--     │ STRUCTURED_QUERY    │ 동적 SQL → Oracle DB 상품 검색 (GPT 미사용)   │
--     │ SEMANTIC_SEARCH     │ 벡터 유사도 → ChromaDB + GPT RAG 응답         │
--     │ FAQ                 │ FAQ DB 검색 응답 (GPT 미사용)                  │
--     │ ORDER_INQUIRY       │ 실DB 주문 조회 (GPT 미사용)                    │
--     │ COMPLAINT           │ 감정 공감 응답 (GPT 사용)                      │
--     │ SMALL_TALK          │ 일반 대화 (GPT 사용)                           │
--     ├─────────────────────┼────────────────────────────────────────────────┤
--     │ AGENT               │ 단일 ReAct Agent (/chat/agent)                 │
--     │ MULTI_AGENT         │ 멀티 Agent Supervisor (/chat/multi-agent)      │
--     │ MCP_AGENT           │ MCP 외부 서비스 연동 Agent (/chat/mcp-agent)   │
--     └─────────────────────┴────────────────────────────────────────────────┘
--
--   · PHASE 3 비교 분석: CHAT_TYPE 로 GROUP BY 하면 경로별 사용 빈도,
--     MEMBER_ID 별 선호 경로 등을 SQL 수준에서 바로 분석할 수 있다.
--
-- [컬럼 설계 의도]
--   · ANSWER: CLOB — GPT 응답, RAG 응답 등 길이 제한 없이 수용.
--   · QUESTION: VARCHAR2(4000) — 사용자 입력은 API 에서 max_length=1000 제한.
--==============================================================================
CREATE TABLE CHAT_HISTORY (
    CHAT_ID    NUMBER          NOT NULL,
    MEMBER_ID  NUMBER          NOT NULL,
    QUESTION   VARCHAR2(4000)  NOT NULL,
    ANSWER     CLOB            NOT NULL,
    CHAT_TYPE  VARCHAR2(20)    NOT NULL,
    CHAT_DATE  TIMESTAMP                 DEFAULT SYSTIMESTAMP,
    --
    CONSTRAINT PK_CHAT_HISTORY        PRIMARY KEY (CHAT_ID),
    CONSTRAINT CK_CHAT_HISTORY_TYPE   CHECK       (CHAT_TYPE IN (
                                          -- ── 라우터 그래프 (6종 인텐트) ──
                                          'STRUCTURED_QUERY',
                                          'SEMANTIC_SEARCH',
                                          'FAQ',
                                          'ORDER_INQUIRY',
                                          'COMPLAINT',
                                          'SMALL_TALK',
                                          -- ── Agent 경로 (3종) ─────────
                                          'AGENT',
                                          'MULTI_AGENT',
                                          'MCP_AGENT'
                                      )),
    CONSTRAINT FK_CHAT_HISTORY_MEMBER FOREIGN KEY (MEMBER_ID) REFERENCES MEMBER (MEMBER_ID)
);


--==============================================================================
-- 11. 인덱스
--
-- [설계 원칙]
--   A) Oracle 은 FK 컬럼에 인덱스를 자동 생성하지 않는다.
--      → 부모 테이블 DML 시 자식 테이블 풀스캔 락(lock) 방지를 위해 명시 생성.
--   B) AI 파이프라인 고빈도 쿼리 패턴을 분석해 필요한 인덱스를 추가한다.
--   C) UK_CHAT_TOKEN_TOKEN 은 UNIQUE 제약이 인덱스를 겸하므로 별도 생성 불필요.
--==============================================================================

-- ── [A] FK 인덱스 (부모 DML 락 방지 + 조인 성능) ──────────────────────────

-- CART
CREATE INDEX IX_CART_MEMBER  ON CART (MEMBER_ID);   -- 회원 장바구니 조회
CREATE INDEX IX_CART_PRODUCT ON CART (PRODUCT_ID);  -- 상품 삭제 시 자식 락 방지

-- ORDERS
CREATE INDEX IX_ORDERS_MEMBER ON ORDERS (MEMBER_ID);
-- · ORDER_INQUIRY 인텐트: fetch_orders(member_id) 의 첫 번째 필터
-- · 개인화(PHASE 2): 회원 구매 이력 전체 조회 기점

-- ORDER_ITEM
CREATE INDEX IX_ORDER_ITEM_ORDER   ON ORDER_ITEM (ORDER_ID);
-- · ORDERS ⋈ ORDER_ITEM JOIN 시 드리븐 측 인덱스
CREATE INDEX IX_ORDER_ITEM_PRODUCT ON ORDER_ITEM (PRODUCT_ID);
-- · 개인화: 구매한 PRODUCT_ID → ChromaDB 임베딩 조회 전 집계
-- · 상품 삭제 시 자식 락 방지

-- CHAT_HISTORY
CREATE INDEX IX_CHAT_HISTORY_MEMBER ON CHAT_HISTORY (MEMBER_ID, CHAT_DATE DESC);
-- · 회원별 최신 대화 이력 조회 (복합 인덱스 → 정렬 생략)

-- ── [B] AI 파이프라인 쿼리 최적화 인덱스 ────────────────────────────────────

-- PRODUCT — STRUCTURED_QUERY 인텐트 핵심 필터
CREATE INDEX IX_PRODUCT_CATEGORY       ON PRODUCT (CATEGORY);
-- · build_structured_query: "WHERE CATEGORY = :category"
--   카테고리 필터 단독 사용 시 범위 스캔

CREATE INDEX IX_PRODUCT_CATEGORY_PRICE ON PRODUCT (CATEGORY, PRICE);
-- · 카테고리 + 가격 범위 복합 필터가 가장 빈번한 패턴
--   "WHERE CATEGORY = :cat AND PRICE BETWEEN :min AND :max"
--   선두 컬럼(CATEGORY) 단독 조건도 커버 → IX_PRODUCT_CATEGORY 대체 가능

CREATE INDEX IX_PRODUCT_STATUS         ON PRODUCT (STATUS);
-- · Admin 화면 상태별 상품 조회 / 향후 STATUS 필터 추가 대비

-- FAQ — FAQ 인텐트 핸들러
CREATE INDEX IX_FAQ_CATEGORY ON FAQ (CATEGORY);
-- · fetch_faq_list(category): "WHERE CATEGORY = :category"

-- CHAT_HISTORY — PHASE 3 경로별 비교 분석
CREATE INDEX IX_CHAT_HISTORY_TYPE ON CHAT_HISTORY (CHAT_TYPE, CHAT_DATE DESC);
-- · "GROUP BY CHAT_TYPE" 집계, 경로별 최신 이력 조회
-- · PHASE 3 비교 분석 쿼리: 라우터 vs Agent 사용 빈도, 시계열 추이


--==============================================================================
-- 12. Oracle COMMENT ON — SQL Developer / DBeaver 등 툴에서 컬럼 설명 표시
--==============================================================================

COMMENT ON TABLE  MEMBER       IS '쇼핑몰 회원. Spring Security 인증 주체. CHAT_TOKEN 통해 AI 챗봇과 연결.';
COMMENT ON TABLE  PRODUCT      IS '상품 카탈로그. DESCRIPTION → ChromaDB 임베딩(SEMANTIC), IMAGE_URL → CLIP 임베딩(PHASE2).';
COMMENT ON TABLE  FAQ          IS 'FAQ. AI 파이프라인 FAQ 인텐트 핸들러 검색 대상.';
COMMENT ON TABLE  EMAIL_VERIFY IS '이메일 인증코드. PK=EMAIL(회원당 1개), LOGTIME 기준 3분 만료.';
COMMENT ON TABLE  CHAT_TOKEN   IS 'AI 챗봇 세션 토큰. Spring 로그인 시 UUID 발급 → FastAPI resolve_chat_token() 역조회.';
COMMENT ON TABLE  CART         IS '장바구니.';
COMMENT ON TABLE  ORDERS       IS '주문 헤더. AI ORDER_INQUIRY 인텐트 + 개인화 추천(PHASE2) 집계 대상.';
COMMENT ON TABLE  ORDER_ITEM   IS '주문 상세. QUANTITY=[PHASE2 예약어 리네임]. 개인화 취향벡터 가중합산 소스.';
COMMENT ON TABLE  CHAT_HISTORY IS 'AI 챗봇 대화 이력. CHAT_TYPE 9종으로 경로별 분류. PHASE3 비교 분석 기반.';

COMMENT ON COLUMN MEMBER.ROLE      IS 'Spring Security 권한. USER(일반)/ADMIN(관리자).';
COMMENT ON COLUMN MEMBER.STATUS    IS 'ACTIVE(정상)/INACTIVE(자발비활성)/SUSPENDED(관리자제재).';
COMMENT ON COLUMN MEMBER.VERIFIED  IS '이메일 인증 완료 여부. Y/N.';
COMMENT ON COLUMN MEMBER.PASSWORD  IS 'BCrypt 해시. 평문 저장 금지.';

COMMENT ON COLUMN PRODUCT.PRODUCT_NAME IS 'Spring @Column(name="PRODUCT_NAME"). Java 필드명 name 과 다름.';
COMMENT ON COLUMN PRODUCT.STOCK        IS 'Spring @Column(name="STOCK"). Java 필드명 stockQuantity 와 다름. SOLD_OUT 시 0.';
COMMENT ON COLUMN PRODUCT.DESCRIPTION  IS 'ChromaDB text-embedding-3-small 임베딩 소스. 풍부한 설명이 RAG 품질에 직결.';
COMMENT ON COLUMN PRODUCT.IMAGE_URL    IS 'PHASE2: OpenCLIP(ViT-B-32) 이미지 임베딩 소스. 검색결과 카드 썸네일. VARCHAR2(1000) 확장(PHASE2).';
COMMENT ON COLUMN PRODUCT.STATUS       IS 'ACTIVE(판매중)/SOLD_OUT(품절,STOCK=0)/DISCONTINUED(판매종료). AI는 전체 검색 후 상태를 답변에 포함.';

COMMENT ON COLUMN CHAT_TOKEN.CHAT_TOKEN  IS 'UUID v4. FastAPI resolve_chat_token() 역조회 키. UK_CHAT_TOKEN_TOKEN 으로 유일성 보장.';
COMMENT ON COLUMN CHAT_TOKEN.EXPIRE_DATE IS '세션 만료일시. Spring 이 로그인마다 갱신.';

COMMENT ON COLUMN ORDERS.ORDER_STATUS IS 'PENDING→PAID→PREPARING→SHIPPED→DELIVERED / CANCELLED / REFUNDED 라이프사이클.';
COMMENT ON COLUMN ORDERS.TOTAL_PRICE  IS '주문 시점 금액 스냅샷. 이후 상품 가격 변동과 무관하게 보존.';

COMMENT ON COLUMN ORDER_ITEM.QUANTITY IS 'PHASE2: COUNT(예약어) → QUANTITY 리네임. Spring @Column(name="QUANTITY").';
COMMENT ON COLUMN ORDER_ITEM.PRICE    IS '주문 시점 단가 스냅샷.';

COMMENT ON COLUMN CHAT_HISTORY.CHAT_TYPE IS '9종: STRUCTURED_QUERY/SEMANTIC_SEARCH/FAQ/ORDER_INQUIRY/COMPLAINT/SMALL_TALK(라우터그래프) + AGENT/MULTI_AGENT/MCP_AGENT(Agent경로).';
COMMENT ON COLUMN CHAT_HISTORY.ANSWER    IS 'CLOB. GPT/RAG 응답 길이 제한 없이 수용.';


--==============================================================================
-- 13. 상품 시드 데이터 — 의류15 + 전자기기4 + 뷰티3 = 22개
--     FastAPI 기동 후 python -m scripts.index_products 로 ChromaDB 재색인 필수
--==============================================================================
INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '화이트 베이직 크루넥 티셔츠', 35000, 45, '사계절 활용 가능한 흰색 크루넥 티셔츠입니다. 부드러운 100% 코튼 소재로 피부 자극이 적고 흡습성이 뛰어납니다. 슬림핏 실루엣으로 와이드 데님 팬츠나 슬랙스와 코디하면 깔끔한 데일리 룩이 완성됩니다. 봄, 여름에는 단독으로, 가을·겨울에는 양털 후리스나 트렌치 코트 안에 이너로 착용하기 좋습니다. 소풍, 캠퍼스, 카페 등 가벼운 외출 상황에 두루 어울립니다.', '상의', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '스트라이프 오버핏 셔츠', 55000, 0, '네이비&화이트 스트라이프 패턴의 오버핏 셔츠입니다. 고밀도 면혼방 소재로 구김이 적고 통기성이 좋습니다. 루즈한 실루엣 덕분에 미니 플리츠 스커트나 와이드 데님 팬츠와 함께 입으면 트렌디한 캐주얼 룩이 연출됩니다. 봄·여름 데이트룩, 브런치 카페 방문 등 세미캐주얼 상황에 잘 어울립니다. 현재 재고가 모두 소진된 품절 상품으로, 재입고 알림을 신청하시면 입고 즉시 안내드립니다.', '상의', SYSTIMESTAMP, NULL, NULL, 'SOLD_OUT');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '슬림핏 터틀넥 니트', 68000, 30, '부드러운 울혼방 소재의 슬림핏 터틀넥 니트입니다. 목을 따뜻하게 감싸주는 높은 넥라인이 특징으로 찬바람을 효과적으로 차단합니다. 테이퍼드 슬랙스나 와이드 데님 팬츠와 매칭하면 세련된 겨울 오피스룩이 완성됩니다. 아이보리, 차콜, 네이비 3가지 컬러로 구성되어 아우터 레이어링에 모두 잘 어울립니다. 가을·겨울 실내 근무, 미팅, 데이트 등 다양한 자리에 적합합니다.', '상의', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '린넨 반팔 셔츠', 48000, 60, '천연 린넨 소재로 제작된 루즈핏 반팔 셔츠입니다. 린넨 특유의 빳빳한 질감과 통기성 덕분에 땀이 많은 여름철에도 시원하게 착용할 수 있습니다. 코튼 조거 팬츠나 미니 플리츠 스커트와 함께 입으면 가볍고 산뜻한 여름 캐주얼 룩이 완성됩니다. 베이지, 올리브, 스카이블루 색상으로 구성되어 여름 해변, 여행, 주말 나들이 등에 두루 활용할 수 있습니다.', '상의', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '와이드 데님 팬츠', 72000, 38, '편안한 핏감의 와이드 데님 팬츠입니다. 스트레치 혼방 데님 소재로 움직임이 자유롭습니다. 화이트 크루넥 티셔츠나 스트라이프 오버핏 셔츠와 매칭하면 세련된 캐주얼 코디가 완성됩니다. 경량 패딩 점퍼나 트렌치 코트와 함께 레이어링하면 계절이 바뀌는 환절기에도 활용도가 높습니다. 데이트, 쇼핑, 캠퍼스 등 다양한 일상 상황에 잘 어울리며 인디고·연청·블랙 워싱 컬러로 구성되어 있습니다.', '하의', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '슬랙스 테이퍼드 팬츠', 65000, 0, '발목으로 갈수록 좁아지는 테이퍼드 실루엣의 슬랙스입니다. 폴리·레이온 혼방 소재로 드레이프 핏이 자연스럽고 구김이 잘 생기지 않습니다. 터틀넥 니트나 린넨 반팔 셔츠 위에 걸쳐 오피스 캐주얼 룩으로 완성하거나, 클래식 캔버스 스니커즈와 코디해 편안한 세미포멀 스타일링이 가능합니다. 출근, 비즈니스 미팅, 세미나 등 격식 있는 자리에 적합합니다. 현재 재고가 소진된 품절 상품으로, 재입고 시 알림 신청을 받고 있습니다.', '하의', SYSTIMESTAMP, NULL, NULL, 'SOLD_OUT');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '코튼 조거 팬츠', 45000, 55, '부드러운 코튼 혼방 소재의 조거 팬츠입니다. 허리 밴딩과 발목 밴딩 구조로 착용감이 편안하며 보온성도 적당합니다. 오버핏 티셔츠나 양털 후리스와 매칭하면 편안한 홈웨어 겸 외출복으로 활용 가능합니다. 슬립온 로퍼나 캔버스 스니커즈와 함께 신으면 캐주얼한 데일리 스타일링이 완성됩니다. 운동, 산책, 집 근처 편의점 등 가벼운 일상 외출에 최적화되어 있으며 그레이·블랙·카키 3색으로 구성됩니다.', '하의', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '미니 플리츠 스커트', 42000, 20, '잔잔한 플리츠가 살아있는 미니 기장 스커트입니다. 폴리 소재로 가볍고 움직임에 따라 자연스럽게 펼쳐집니다. 스트라이프 오버핏 셔츠나 슬림핏 터틀넥 니트와 코디하면 여성스러운 캐주얼 룩이 완성됩니다. 첼시 앵클 부츠와 함께 신으면 가을·겨울 트렌디한 스타일링이 가능하고, 캔버스 스니커즈와 매칭하면 봄·여름 발랄한 룩이 연출됩니다. 쇼핑, 데이트, 파티 등 다양한 자리에 잘 어울리며 블랙·베이지·체크 패턴으로 구성됩니다.', '하의', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '방수 트래킹 스니커즈', 89000, 25, '방수 코팅 원단과 미끄럼 방지 러버 아웃솔을 적용한 트래킹 스니커즈입니다. 비오는 날에도 발이 젖지 않아 우천 시 외출에 최적화된 신발입니다. 두꺼운 쿠션 미들솔이 장시간 보행 시에도 발바닥 피로를 줄여줍니다. 와이드 데님 팬츠나 코튼 조거 팬츠와 매칭하면 활동적인 아웃도어 캐주얼 스타일이 완성됩니다. 등산, 캠핑, 비 오는 날 출근, 장거리 여행 등 야외 활동 상황에 강력히 추천드립니다.', '신발', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '클래식 캔버스 스니커즈', 55000, 70, '군더더기 없는 클래식한 디자인의 캔버스 스니커즈입니다. 면 캔버스 갑피와 천연 고무 아웃솔로 가볍고 통기성이 좋습니다. 어떤 하의와도 잘 어울리는 화이트 베이스 컬러로, 데님 팬츠, 슬랙스, 플리츠 스커트, 조거 팬츠 등 다양한 코디가 가능합니다. 봄·여름 데일리 착화에 최적화되어 있으며 캠퍼스, 카페, 쇼핑몰 등 도심 캐주얼 상황에 가장 많이 활용됩니다. 화이트, 블랙, 레드 컬러로 구성됩니다.', '신발', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '첼시 앵클 부츠', 120000, 15, '사이드 고어 밴드가 있는 클래식 첼시 스타일의 앵클 부츠입니다. 비건 레더 소재로 광택감이 있어 고급스러운 느낌을 줍니다. 5cm 블록힐로 키가 커 보이는 효과와 함께 장시간 착용해도 안정감 있는 착화감을 제공합니다. 슬랙스 테이퍼드 팬츠나 미니 플리츠 스커트와 코디하면 세련된 가을·겨울 포멀 캐주얼 룩이 완성됩니다. 출근, 데이트, 파티 등 격식과 트렌드를 동시에 챙겨야 하는 자리에 적합하며 블랙·다크브라운 컬러로 구성됩니다.', '신발', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '슬립온 로퍼', 78000, 40, '끈 없이 편하게 신고 벗을 수 있는 슬립온 로퍼입니다. 페니 로퍼 스타일에 쿠셔닝 인솔을 추가해 하루 종일 신어도 발이 편안합니다. 부드러운 PU 소재 갑피로 내구성이 좋고 관리가 쉽습니다. 슬랙스나 와이드 데님 팬츠와 매칭하면 클래식한 아이비룩이 연출되며, 코튼 조거 팬츠와 함께하면 편안한 캐주얼 룩도 가능합니다. 오피스, 학교, 카페, 갤러리 등 세미캐주얼 환경에 두루 어울리며 블랙·카멜·버건디 컬러로 구성됩니다.', '신발', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '오버핏 양털 후리스', 98000, 22, '두툼한 양털 플리스 소재의 오버핏 집업 후리스입니다. 부드럽고 가벼운 소재로 보온성이 뛰어나며 정전기가 잘 생기지 않는 항정전기 가공 처리가 되어 있습니다. 화이트 크루넥 티셔츠나 터틀넥 니트 위에 레이어링하면 두껍지 않으면서도 따뜻한 겨울 레이어드 룩이 완성됩니다. 와이드 데님 팬츠나 코튼 조거 팬츠와 함께 입으면 편안한 아웃도어 캐주얼 스타일링이 가능합니다. 캠핑, 등산 베이스캠프, 일상 외출 등 다양한 야외 활동에 적합하며 아이보리·그레이·블랙 컬러로 구성됩니다.', '아우터', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '싱글 트렌치 코트', 148000, 12, '클래식한 실루엣의 미디 기장 싱글 트렌치 코트입니다. 고밀도 면혼방 원단으로 봄비나 가을비 정도는 어느 정도 막아주는 발수 코팅 처리가 되어 있습니다. 허리 벨트로 라인을 조절할 수 있어 다양한 체형에 잘 맞으며, 단정한 A라인 실루엣을 연출합니다. 슬림핏 터틀넥 니트나 스트라이프 셔츠 위에 걸치면 세련된 시티룩이 완성되고, 첼시 앵클 부츠와 코디하면 완성도 높은 포멀 캐주얼 스타일이 만들어집니다. 출근, 비즈니스 미팅, 고급 레스토랑 방문 등 격식 있는 자리에 가장 적합하며 베이지·카멜·블랙 컬러로 구성됩니다.', '아우터', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '경량 패딩 점퍼', 135000, 8, '압축 보관이 가능한 초경량 패딩 점퍼입니다. 90% 구스다운 충전재로 가볍지만 보온성이 매우 뛰어나며 겨울 방한에 최적화되어 있습니다. 리플 나일론 소재의 겉감은 생활 방수 처리가 되어 있어 갑작스러운 소나기에도 빠르게 물을 튕겨냅니다. 와이드 데님 팬츠나 슬랙스 위에 걸치면 활동적이면서도 따뜻한 겨울 데일리 룩이 완성됩니다. 방수 트래킹 스니커즈와 함께 코디하면 야외 활동에 최적화된 기능성 스타일링이 가능합니다. 스키장, 해외여행, 혹한기 출근 등 강추위 환경에서도 충분한 보온을 보장하며 블랙·네이비·올리브 컬러로 구성됩니다.', '아우터', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '무선 블루투스 이어폰', 89000, 50, '블루투스 5.3을 지원하는 커널형 무선 이어폰입니다. 액티브 노이즈 캔슬링 기능으로 지하철, 카페 등 시끄러운 환경에서도 음악에 몰입할 수 있습니다. 한 번 충전으로 최대 8시간, 충전 케이스 포함 최대 32시간 재생이 가능해 하루 종일 사용하기에 충분합니다. IPX4 생활 방수를 지원해 운동 중 땀이나 가벼운 빗방울에도 안심하고 사용할 수 있습니다. 출퇴근, 운동, 재택근무 화상회의 등 다양한 일상 상황에 두루 활용되며 화이트·블랙 두 가지 컬러로 구성됩니다.', '전자기기', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '경량 노트북', 990000, 18, '무게 1kg 미만의 초경량 14인치 노트북입니다. 최신 저전력 프로세서와 16GB 메모리를 탑재해 문서 작업, 웹 서핑, 영상 시청을 부드럽게 처리합니다. 한 번 충전으로 최대 18시간 사용 가능한 대용량 배터리로 외부 작업 시 충전기 없이도 하루를 버틸 수 있습니다. 풀 알루미늄 메탈 바디로 견고하면서도 가방에 쏙 들어가는 슬림한 두께를 자랑합니다. 대학생 과제, 카페 원격근무, 출장 등 휴대성이 중요한 상황에 최적화되어 있습니다.', '전자기기', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '스마트워치', 259000, 33, '심박수, 수면, 스트레스 지수까지 측정하는 헬스케어 스마트워치입니다. 1.4인치 AMOLED 디스플레이로 야외 햇빛 아래에서도 화면이 또렷하게 보입니다. 100가지 이상의 운동 모드를 지원해 러닝, 수영, 헬스 등 다양한 활동을 기록할 수 있으며 5ATM 방수로 수영 중에도 착용 가능합니다. 한 번 충전으로 최대 14일 사용할 수 있어 잦은 충전의 번거로움이 없습니다. 운동 관리, 건강 모니터링, 스마트폰 알림 확인 등 일상 전반에서 활용도가 높습니다.', '전자기기', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '미니 블루투스 스피커', 49000, 0, '손바닥 크기의 휴대용 블루투스 스피커입니다. 작은 크기에도 360도 사운드를 구현해 풍부하고 균형 잡힌 음질을 들려줍니다. IPX7 완전 방수 등급으로 욕실, 수영장, 해변 등 물기가 많은 환경에서도 안심하고 사용할 수 있습니다. 한 번 충전으로 최대 12시간 연속 재생이 가능하며 카라비너 스트랩이 있어 가방이나 텐트에 간편하게 걸 수 있습니다. 캠핑, 피크닉, 샤워 시간 등 다양한 야외·실내 상황에 잘 어울립니다. 현재 재고가 모두 소진된 품절 상품으로, 재입고 알림을 신청하시면 입고 즉시 안내드립니다.', '전자기기', SYSTIMESTAMP, NULL, NULL, 'SOLD_OUT');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '수분 크림', 28000, 80, '건조한 피부에 깊은 수분을 채워주는 고보습 수분 크림입니다. 히알루론산과 세라마이드 성분이 피부 장벽을 강화하고 수분 증발을 막아 촉촉함을 오래 유지시켜 줍니다. 끈적임 없이 산뜻하게 흡수되는 젤크림 제형으로 모든 피부 타입에 부담 없이 사용할 수 있습니다. 무향·무색소·약산성 처방으로 민감성 피부도 안심하고 사용 가능합니다. 아침 기초 마무리, 저녁 나이트 크림, 환절기 집중 보습 등 사계절 데일리 보습 루틴에 적합합니다.', '뷰티', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, 'SPF50 선크림', 19000, 65, 'SPF50+ PA++++ 최고 등급의 자외선 차단 기능을 갖춘 데일리 선크림입니다. 백탁 현상 없이 자연스럽게 발리는 가벼운 제형으로 메이크업 베이스로도 활용 가능합니다. 끈적임이 적어 더운 여름철에도 산뜻하게 사용할 수 있으며 땀과 물에 강한 워터프루프 처방으로 야외 활동 시에도 지속력이 우수합니다. 출근길, 등산, 해변, 운동 등 자외선 노출이 많은 모든 상황에서 피부를 보호합니다. 매일 아침 기초 케어 마지막 단계에 사용하기를 권장합니다.', '뷰티', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');

INSERT INTO PRODUCT (PRODUCT_ID, PRODUCT_NAME, PRICE, STOCK, DESCRIPTION, CATEGORY, CREATED_AT, IMAGE_URL, MANUFACTURER, STATUS)
VALUES (PRODUCT_SEQ.NEXTVAL, '비타민C 세럼', 36000, 42, '순수 비타민C 유도체를 고농도로 담은 브라이트닝 세럼입니다. 칙칙한 피부톤을 환하게 가꾸고 멜라닌 생성을 억제해 잡티와 색소 침착 개선에 도움을 줍니다. 가볍고 빠르게 흡수되는 워터리 제형으로 끈적임 없이 사용할 수 있으며 항산화 성분이 외부 자극으로부터 피부를 보호합니다. 토너 다음 단계에 2~3방울 발라주면 이후 크림의 흡수를 도와줍니다. 칙칙해진 피부톤 관리, 환절기 피부 컨디션 케어 등 집중 미백 루틴에 적합합니다. 개봉 후에는 냉장 보관을 권장합니다.', '뷰티', SYSTIMESTAMP, NULL, NULL, 'ACTIVE');


--==============================================================================
-- 14. FAQ 시드 데이터 (8건)
--==============================================================================
INSERT INTO FAQ (FAQ_ID, QUESTION, ANSWER, CATEGORY)
VALUES (FAQ_SEQ.NEXTVAL, '배송은 보통 얼마나 걸리나요?', '결제 완료 후 영업일 기준 2~3일 이내에 상품을 받아보실 수 있습니다. 주문 폭주나 기상 악화 시 1~2일 정도 지연될 수 있으며, 도서산간 지역은 추가 배송일이 소요될 수 있습니다.', '배송');

INSERT INTO FAQ (FAQ_ID, QUESTION, ANSWER, CATEGORY)
VALUES (FAQ_SEQ.NEXTVAL, '배송비는 얼마인가요?', '5만 원 이상 구매 시 무료배송이며, 미만일 경우 3,000원의 배송비가 부과됩니다. 제주 및 도서산간 지역은 3,000원의 추가 배송비가 발생합니다.', '배송');

INSERT INTO FAQ (FAQ_ID, QUESTION, ANSWER, CATEGORY)
VALUES (FAQ_SEQ.NEXTVAL, '교환은 어떻게 신청하나요?', '상품 수령 후 7일 이내에 마이페이지 주문내역에서 교환 신청이 가능합니다. 단순 변심에 의한 교환은 왕복 배송비가 부과되며, 상품 불량의 경우 무료로 교환해 드립니다.', '교환');

INSERT INTO FAQ (FAQ_ID, QUESTION, ANSWER, CATEGORY)
VALUES (FAQ_SEQ.NEXTVAL, '환불은 언제 처리되나요?', '반품 상품이 물류센터에 도착해 검수가 완료되면 영업일 기준 3~5일 이내에 환불이 진행됩니다. 카드 결제의 경우 카드사 사정에 따라 취소 반영까지 추가 시일이 걸릴 수 있습니다.', '환불');

INSERT INTO FAQ (FAQ_ID, QUESTION, ANSWER, CATEGORY)
VALUES (FAQ_SEQ.NEXTVAL, '환불이 불가능한 경우도 있나요?', '착용 흔적이 있거나 택을 제거한 상품, 세탁한 상품, 화장품류의 개봉 상품은 환불이 제한됩니다. 단, 상품 하자가 있는 경우에는 위 기준과 관계없이 환불이 가능합니다.', '환불');

INSERT INTO FAQ (FAQ_ID, QUESTION, ANSWER, CATEGORY)
VALUES (FAQ_SEQ.NEXTVAL, '회원가입은 어떻게 하나요?', '상단 메뉴의 회원가입 버튼을 눌러 아이디, 비밀번호, 이메일 인증을 거치면 가입이 완료됩니다. 이메일 인증번호는 발송 후 3분간 유효합니다.', '회원');

INSERT INTO FAQ (FAQ_ID, QUESTION, ANSWER, CATEGORY)
VALUES (FAQ_SEQ.NEXTVAL, '비밀번호를 잊어버렸어요.', '로그인 화면의 비밀번호 찾기를 이용하시면 가입 시 등록한 이메일로 임시 비밀번호가 발송됩니다. 로그인 후 마이페이지에서 새 비밀번호로 변경해 주세요.', '회원');

INSERT INTO FAQ (FAQ_ID, QUESTION, ANSWER, CATEGORY)
VALUES (FAQ_SEQ.NEXTVAL, '주문 후 배송지를 변경할 수 있나요?', '상품이 배송 준비 단계로 넘어가기 전이라면 고객센터를 통해 배송지 변경이 가능합니다. 이미 발송된 경우에는 변경이 어려우니 주문 시 배송지를 정확히 입력해 주세요.', '주문');


--==============================================================================
-- 15. 커밋
--==============================================================================
COMMIT;


--------------------------------------------------------------------------------
-- 완료 후 확인 쿼리 (선택)
--
-- 테이블 목록:
--   SELECT TABLE_NAME FROM USER_TABLES ORDER BY TABLE_NAME;
--
-- 제약 목록 (CHECK 포함):
--   SELECT TABLE_NAME, CONSTRAINT_NAME, CONSTRAINT_TYPE, SEARCH_CONDITION
--     FROM USER_CONSTRAINTS WHERE CONSTRAINT_TYPE IN ('P','U','R','C')
--    ORDER BY TABLE_NAME, CONSTRAINT_TYPE;
--
-- 인덱스 목록:
--   SELECT INDEX_NAME, TABLE_NAME, UNIQUENESS, COLUMN_NAME
--     FROM USER_IND_COLUMNS JOIN USER_INDEXES USING (INDEX_NAME)
--    ORDER BY TABLE_NAME, INDEX_NAME;
--
-- 상품 시드 확인:
--   SELECT PRODUCT_ID, PRODUCT_NAME, CATEGORY, PRICE, STOCK, STATUS
--     FROM PRODUCT ORDER BY PRODUCT_ID;
--
-- CHAT_HISTORY 경로별 집계 (PHASE 3 비교 분석):
--   SELECT CHAT_TYPE, COUNT(*) AS CNT, ROUND(AVG(LENGTH(ANSWER)),0) AS AVG_ANS_LEN
--     FROM CHAT_HISTORY GROUP BY CHAT_TYPE ORDER BY CNT DESC;
--
-- 다음 단계:
--   1) Spring Boot 기동 → AdminAccountInitializer 가 admin/admin1234 자동 생성
--   2) FastAPI: python -m scripts.index_products  → ChromaDB 22개 재색인
--   3) (선택) FastAPI: python -m scripts.index_products_image  → CLIP 이미지 색인
--------------------------------------------------------------------------------
