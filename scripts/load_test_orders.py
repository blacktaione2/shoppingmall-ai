"""
load_test_orders.py — 재고 비관적 락 부하테스트 (측정 항목 ⑦).

같은 상품에 동시 주문 N건을 쏘아 "성공 주문 수 == 초기 재고" 를 검증한다.
락이 없으면(수정 전) 초기 재고보다 많은 주문이 성공하는 초과판매가 재현되고,
락이 있으면(수정 후) 정확히 재고만큼만 성공해야 한다.

판정 기준 (Spring OrderController.directOrder 계약):
- 성공: 302 redirect Location 에 /orders/complete/ 포함
- 재고 부족 거절: redirect Location 에 /product/detail?id=...&msg=... 포함

사용 예 (관리자 화면에서 테스트 상품 재고를 10으로 맞춘 뒤):
    python scripts/load_test_orders.py --login-id user1 --password pw1 \
        --product-id 15 --stock 10 --concurrency 50

의존성: requests (pip install requests)
"""
import argparse
import concurrent.futures
import logging
import time

import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def login(base_url: str, login_id: str, password: str) -> requests.Session:
    s = requests.Session()
    res = s.post(base_url + "/login", data={"id": login_id, "pwd": password},
                 allow_redirects=False, timeout=10)
    # 성공 시 redirect:/main, 실패 시 200(loginForm 재렌더)
    if res.status_code != 302:
        raise SystemExit(f"로그인 실패 (HTTP {res.status_code}) — 계정/비밀번호 확인")
    logger.info("로그인 성공: %s", login_id)
    return s


def fire_order(session: requests.Session, base_url: str, product_id: int) -> tuple[str, float]:
    """주문 1건. ('success'|'rejected'|'error', 소요 ms) 반환."""
    started = time.perf_counter()
    try:
        res = session.post(base_url + "/order/direct",
                           data={"productId": product_id, "count": 1},
                           allow_redirects=False, timeout=30)
        elapsed = (time.perf_counter() - started) * 1000
        loc = res.headers.get("Location", "")
        if res.status_code == 302 and "/orders/complete/" in loc:
            return "success", elapsed
        if res.status_code == 302 and "/product/detail" in loc:
            return "rejected", elapsed          # 재고 부족 등 정상 거절
        return "error", elapsed                  # 예상 밖 응답 (로그인 만료 등)
    except requests.RequestException:
        return "error", (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="재고 락 동시 주문 부하테스트")
    parser.add_argument("--base-url", default="http://localhost:8080", help="Spring Boot 주소")
    parser.add_argument("--login-id", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--product-id", type=int, required=True, help="테스트 상품 PRODUCT_ID")
    parser.add_argument("--stock", type=int, required=True,
                        help="테스트 직전 설정한 초기 재고 (성공 수 판정 기준)")
    parser.add_argument("--concurrency", type=int, default=50, help="동시 주문 수")
    args = parser.parse_args()

    session = login(args.base_url, args.login_id, args.password)

    logger.info("동시 주문 %d건 발사 → PRODUCT_ID=%d (초기 재고 %d)",
                args.concurrency, args.product_id, args.stock)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(fire_order, session, args.base_url, args.product_id)
                   for _ in range(args.concurrency)]
        results = [f.result() for f in futures]
    wall_ms = (time.perf_counter() - started) * 1000

    statuses = [r[0] for r in results]
    latencies = sorted(r[1] for r in results)
    n_success = statuses.count("success")
    n_rejected = statuses.count("rejected")
    n_error = statuses.count("error")

    logger.info("-" * 50)
    logger.info("성공 %d / 거절 %d / 오류 %d  (전체 %.0fms)",
                n_success, n_rejected, n_error, wall_ms)
    logger.info("주문 레이턴시: p50 %.0fms / p95 %.0fms / max %.0fms  ← 락 대기 포함",
                latencies[len(latencies) // 2],
                latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
                latencies[-1])
    logger.info("-" * 50)
    if n_success == args.stock:
        logger.info("✅ 판정: 성공 주문 수(%d) == 초기 재고(%d) — 초과판매 0건, 락 정상 동작",
                    n_success, args.stock)
    elif n_success > args.stock:
        logger.info("❌ 판정: 초과판매 %d건 발생 (성공 %d > 재고 %d) — 락 미동작 의심",
                    n_success - args.stock, n_success, args.stock)
    else:
        logger.info("⚠️ 판정: 성공 %d < 재고 %d — 재고 초기값 또는 오류(%d건) 확인 필요",
                    n_success, args.stock, n_error)
    logger.info("사후 확인: 관리자 화면에서 해당 상품 재고 0 + SOLD_OUT 전환 여부,")
    logger.info("           내 주문 목록의 주문 건수 == %d 인지 대조", n_success)


if __name__ == "__main__":
    main()
