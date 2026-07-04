"""
tests/test_personalization.py  개인화 취향 벡터 서비스 단위 테스트.

검증 항목:
  1) _normalize: L2 정규화 / 0벡터 방어
  2) blend_vectors: 재정규화 + 가중 방향
  3) _recency_weight: 지수 감쇠(반감기=0.5), None/tz-aware 방어
  4) get_preference_vector: 가중합산(수량×최신성), 캐시, 폴백(이력없음/None/장애)

외부 의존(oracle/chroma)은 monkeypatch 로 대체해 네트워크 없이 검증한다.
"""
import asyncio
import math
from datetime import datetime, timedelta, timezone

import pytest

from services import personalization_service as ps
import database.oracle_db as odb
import services.chroma_service as cs


def _run(coro):
    return asyncio.run(coro)


# ── 1) _normalize ────────────────────────────────────────────────────────
def test_normalize_unit_length():
    v = ps._normalize([3.0, 4.0])
    assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-9
    assert v == [0.6, 0.8]


def test_normalize_zero_vector():
    assert ps._normalize([0.0, 0.0]) == [0.0, 0.0]


# ── 2) blend_vectors ─────────────────────────────────────────────────────
def test_blend_renormalized_and_weighted():
    q = ps._normalize([1.0, 0.0, 0.0])
    p = ps._normalize([0.0, 1.0, 0.0])
    b = ps.blend_vectors(q, p, alpha=0.7, beta=0.3)
    # 재정규화되어 단위 길이
    assert abs(math.sqrt(sum(x * x for x in b)) - 1.0) < 1e-9
    # alpha>beta 이므로 q 방향(0번 성분)이 더 큼
    assert b[0] > b[1]


# ── 3) _recency_weight ───────────────────────────────────────────────────
def test_recency_today_is_one():
    now = datetime.now()
    assert abs(ps._recency_weight(now, now.timestamp()) - 1.0) < 1e-6


def test_recency_halflife_is_half():
    now = datetime.now()
    old = now - timedelta(days=ps.HALFLIFE_DAYS)
    assert abs(ps._recency_weight(old, now.timestamp()) - 0.5) < 0.01


def test_recency_none_and_tzaware():
    now = datetime.now()
    assert ps._recency_weight(None, now.timestamp()) == 1.0
    aware = now.replace(tzinfo=timezone.utc)
    # tz-aware 여도 예외 없이 동작
    assert ps._recency_weight(aware, now.timestamp()) >= 0.0


# ── 4) get_preference_vector ─────────────────────────────────────────────
def test_preference_vector_weighted_sum(monkeypatch):
    now = datetime.now()

    def fake_history(member_id, limit=50):
        return [
            {"product_id": 1, "quantity": 2, "order_date": now},                       # w=2.0
            {"product_id": 2, "quantity": 1, "order_date": now - timedelta(days=180)}, # w=0.25
        ]

    async def fake_emb(ids):
        m = {"1": [1.0, 0.0, 0.0], "2": [0.0, 1.0, 0.0]}
        return {k: m[k] for k in ids if k in m}

    monkeypatch.setattr(odb, "fetch_purchase_history", fake_history)
    monkeypatch.setattr(cs, "get_embeddings_by_ids", fake_emb)
    ps.clear_cache()

    vec = _run(ps.get_preference_vector(7))
    exp = ps._normalize([2.0, 0.25, 0.0])  # 상품2: 1*0.5^(180/90)=0.25
    assert all(abs(vec[i] - exp[i]) < 1e-9 for i in range(3))


def test_preference_vector_cached(monkeypatch):
    now = datetime.now()
    calls = {"n": 0}

    def counting_history(member_id, limit=50):
        calls["n"] += 1
        return [{"product_id": 1, "quantity": 1, "order_date": now}]

    async def fake_emb(ids):
        return {"1": [1.0, 0.0, 0.0]}

    monkeypatch.setattr(odb, "fetch_purchase_history", counting_history)
    monkeypatch.setattr(cs, "get_embeddings_by_ids", fake_emb)
    ps.clear_cache()

    _run(ps.get_preference_vector(7))
    _run(ps.get_preference_vector(7))
    assert calls["n"] == 1  # 두 번째는 캐시 → DB 재조회 없음


def test_preference_vector_fallbacks(monkeypatch):
    # 이력 없음 → None
    monkeypatch.setattr(odb, "fetch_purchase_history", lambda m, limit=50: [])
    ps.clear_cache()
    assert _run(ps.get_preference_vector(8)) is None

    # member_id None → None
    assert _run(ps.get_preference_vector(None)) is None

    # DB 장애 → None (예외 미전파)
    def boom(m, limit=50):
        raise RuntimeError("db down")

    monkeypatch.setattr(odb, "fetch_purchase_history", boom)
    ps.clear_cache()
    assert _run(ps.get_preference_vector(9)) is None
