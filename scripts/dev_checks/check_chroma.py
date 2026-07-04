"""
scripts/dev_checks/check_chroma.py
ChromaDB 연결 점검용 개발 스크립트.

heartbeat 와 컬렉션 문서 수를 출력해 ChromaDB 서버 접속이 정상인지 확인한다.
운영 코드가 아니라 배포/디버깅 시 수동 점검용이다.

실행(프로젝트 루트에서):
    python -m scripts.dev_checks.check_chroma
"""
import asyncio

from services import chroma_service

print("heartbeat:", asyncio.run(chroma_service.heartbeat()))
print("count   :", asyncio.run(chroma_service.count()))
