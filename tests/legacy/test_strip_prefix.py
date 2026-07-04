"""
strip_known_prefix() 단위 테스트 [LEGACY — pipeline/router.py 대상, graph/ 경로는 프리픽스 자체가 없음]

[실행 방법]
이 파일을 실제 FastAPI 프로젝트(shoppingmall_ai/) 의 tests/ 폴더에 두고,
프로젝트 루트에서 실행하세요:

    cd shoppingmall_ai
    python tests/test_strip_prefix.py

- pipeline/router.py 의 실제 strip_known_prefix 를 import 하여 검증합니다(복사본 아님).
- router.py 가 import 하는 실제 핸들러/스키마 모듈이 모두 존재해야 합니다(정상 프로젝트면 충족).
"""
import os
import sys

# tests/ 의 부모(= FastAPI 프로젝트 루트)를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.router import (
    strip_known_prefix,
    _KNOWN_PREFIXES,
)

# 환각 가드 폴백/안내 문구 형태 재현 (hallucination_guard import 회피용)
SEMANTIC_FALLBACK = "[추천] 죄송합니다, 정확한 상품 정보를 확인하기 어려워요. 다른 검색어로 다시 시도해 주세요."
GUIDANCE_SUFFIX = "\n\n📌 정확한 주문/배송 상태는 '주문조회' 메뉴에서 확인해 주세요."

CASES = [
    ("[검색결과] 나이키 운동화 89,000원 입니다.", "나이키 운동화 89,000원 입니다."),
    ("[추천] 겨울에는 롱패딩을 추천드려요.", "겨울에는 롱패딩을 추천드려요."),
    ("[FAQ] 배송은 평균 2~3일 소요됩니다.", "배송은 평균 2~3일 소요됩니다."),
    ("[주문조회] 주문번호 ORD-20250115-0001 은 배송중입니다.", "주문번호 ORD-20250115-0001 은 배송중입니다."),
    ("[상담] 불편을 드려 죄송합니다.", "불편을 드려 죄송합니다."),
    ("[잡담] 안녕하세요! 무엇을 도와드릴까요?", "안녕하세요! 무엇을 도와드릴까요?"),
    ("그냥 평범한 답변입니다.", "그냥 평범한 답변입니다."),
    ("[중요] GPT가 임의로 강조한 표현은 보존되어야 합니다.",
     "[중요] GPT가 임의로 강조한 표현은 보존되어야 합니다."),
    ("", ""),
    ("[FAQ]공백없이붙은본문", "공백없이붙은본문"),
    ("답변 중간에 [FAQ] 가 있으면 건드리지 않음", "답변 중간에 [FAQ] 가 있으면 건드리지 않음"),
    (SEMANTIC_FALLBACK, "죄송합니다, 정확한 상품 정보를 확인하기 어려워요. 다른 검색어로 다시 시도해 주세요."),
    ("[상담] 정말 죄송합니다." + GUIDANCE_SUFFIX, "정말 죄송합니다." + GUIDANCE_SUFFIX),
    ("[추천]    여러 공백 뒤 본문", "여러 공백 뒤 본문"),
    ("[FAQ] 끝에 공백 있음   ", "끝에 공백 있음   "),
]


def run():
    passed = failed = 0
    print(f"[정보] _KNOWN_PREFIXES = {_KNOWN_PREFIXES}\n")
    for i, (src, expected) in enumerate(CASES, 1):
        result = strip_known_prefix(src)
        ok = result == expected
        print(f"{'✅' if ok else '❌'} #{i:02d} in={src!r}")
        if ok:
            passed += 1
        else:
            print(f"        기대={expected!r}")
            print(f"        실제={result!r}")
            failed += 1
    print(f"\n=== 결과: {passed} PASSED / {failed} FAILED (총 {len(CASES)}) ===")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
