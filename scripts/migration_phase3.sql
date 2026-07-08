--------------------------------------------------------------------------------
-- migration_phase3.sql
-- PHASE 3 (챗봇 다중 세션) 대비 스키마 보강 — 기존 운영 DB 적용용
--------------------------------------------------------------------------------
-- 대상: 이미 schema.sql 로 구축되어 데이터가 들어있는 운영 DB
--      (신규 환경은 갱신된 schema.sql 한 방이면 되므로 이 파일 불필요)
--
-- 안전성: 이미 적용된 경우(ORA-00955 등) 조용히 통과하도록 멱등 처리.
--        여러 번 돌려도 안전하다.
--
-- 변경 요약
--   1) CHAT_SESSION 테이블 신규 생성 — 회원당 여러 대화방(Thread) 지원.
--      LangGraph 의 thread_id 는 이제 CHAT_TOKEN 이 아니라 SESSION_ID 를 쓴다
--      (CHAT_TOKEN 은 계속 인증 전용, 회원당 1개 유지).
--   2) IX_CHAT_SESSION_MEMBER 인덱스 — 회원별 대화방 목록(최근 활동순) 조회.
--------------------------------------------------------------------------------

-- 1) CHAT_SESSION 테이블 생성 ---------------------------------------------------
--    이미 존재하면 ORA-00955(name is already used) → 무시.
DECLARE
    e_exists EXCEPTION;
    PRAGMA EXCEPTION_INIT(e_exists, -955);
BEGIN
    EXECUTE IMMEDIATE '
        CREATE TABLE CHAT_SESSION (
            SESSION_ID   VARCHAR2(36)   NOT NULL,
            MEMBER_ID    NUMBER         NOT NULL,
            TITLE        VARCHAR2(100)  DEFAULT ''새 대화'' NOT NULL,
            CREATED_AT   TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
            UPDATED_AT   TIMESTAMP      DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT PK_CHAT_SESSION        PRIMARY KEY (SESSION_ID),
            CONSTRAINT FK_CHAT_SESSION_MEMBER FOREIGN KEY (MEMBER_ID) REFERENCES MEMBER (MEMBER_ID)
        )';
    DBMS_OUTPUT.PUT_LINE('CHAT_SESSION 테이블 생성 완료');
EXCEPTION
    WHEN e_exists THEN
        DBMS_OUTPUT.PUT_LINE('CHAT_SESSION 이미 존재 — 스킵');
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('CHAT_SESSION 생성 경고: ' || SQLERRM);
END;
/


-- 2) 인덱스 생성 ----------------------------------------------------------------
DECLARE
    e_exists EXCEPTION;
    PRAGMA EXCEPTION_INIT(e_exists, -955);
BEGIN
    EXECUTE IMMEDIATE 'CREATE INDEX IX_CHAT_SESSION_MEMBER ON CHAT_SESSION (MEMBER_ID, UPDATED_AT DESC)';
    DBMS_OUTPUT.PUT_LINE('인덱스 생성: IX_CHAT_SESSION_MEMBER');
EXCEPTION
    WHEN e_exists THEN
        DBMS_OUTPUT.PUT_LINE('인덱스 이미 존재 — 스킵: IX_CHAT_SESSION_MEMBER');
    WHEN OTHERS THEN
        DBMS_OUTPUT.PUT_LINE('인덱스 경고(IX_CHAT_SESSION_MEMBER): ' || SQLERRM);
END;
/

COMMIT;

--------------------------------------------------------------------------------
-- 적용 후 확인(선택)
--   SELECT TABLE_NAME FROM USER_TABLES WHERE TABLE_NAME = 'CHAT_SESSION';
--   SELECT INDEX_NAME FROM USER_INDEXES WHERE TABLE_NAME = 'CHAT_SESSION';
--------------------------------------------------------------------------------
