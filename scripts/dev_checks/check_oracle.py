"""
scripts/dev_checks/check_oracle.py
Oracle 연결 점검용 개발 스크립트.

PRODUCT 테이블 건수를 출력해 Wallet 기반 DB 연결이 정상인지 빠르게 확인한다.
운영 코드가 아니라 배포/디버깅 시 수동 점검용이다.

실행(프로젝트 루트에서):
    python -m scripts.dev_checks.check_oracle
"""
from database.oracle_db import fetch_all_products

products = fetch_all_products()
print("PRODUCT 건수:", len(products))
