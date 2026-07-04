# tests/legacy/conftest.py
# 이 폴더의 테스트는 일반 pytest 수집 대상에서 제외된다.
# 레거시 파이프라인(LangGraph 전환 이전) 동작을 보존용으로만 유지.
# 실행 필요 시: python tests/legacy/test_strip_prefix.py
collect_ignore_glob = ["*.py"]
