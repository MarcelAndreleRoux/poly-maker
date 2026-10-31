"""Competition-aware reward ranking: the official score formula on live books."""

from __future__ import annotations

import pytest

from polymaker.catalog.competition import assess, book_points, order_points, parse_book, qmin


def test_order_points_formula_and_gates():
    v = 0.045
    assert order_points(0.0, v, 100, 50) == pytest.approx(100.0)
    assert order_points(0.025, v, 100, 50) == pytest.approx(((0.045 - 0.025) / 0.045) ** 2 * 100)
    assert order_points(0.045, v, 100, 50) == 0.0  # at the band edge scores nothing
    assert order_points(0.01, v, 49, 50) == 0.0  # below the size floor
    assert order_points(0.01, 0.0, 100, 50) == 0.0


def test_qmin_two_sidedness_rule():
    assert qmin(100, 100, 0.5) == 100
    assert qmin(100, 10, 0.5) == pytest.approx(100 / 3)  # one-sided inside the band: a third
    assert qmin(100, 10, 0.95) == 10  # outside 0.10-0.90: plain min
    assert qmin(0, 300, 0.5) == 100


def test_assess_matches_hand_calculation(meta):
    # meta: band 3c, min 10 shares, pool 50/day, tick 0.01
    bids = [(0.49, 1000.0), (0.48, 500.0), (0.40, 9999.0)]  # 0.40 is outside the band
    asks = [(0.51, 1000.0), (0.52, 500.0)]
    snap = assess(meta, bids, asks, our_shares=100.0, our_dist_c=2.0)
    assert snap is not None
    assert snap.mid == pytest.approx(0.50)
    v = 0.03
    qb = order_points(0.01, v, 1000, 10) + order_points(0.02, v, 500, 10)
    qa = qb
    assert snap.comp_qmin == pytest.approx(round(qmin(qb, qa, 0.5), 1))
    assert snap.band_bid_shares == 1500  # the 0.40 level is excluded
    ours = order_points(0.02, v, 100, 10)
    assert snap.our_points == pytest.approx(round(ours, 2))
    assert snap.our_share == pytest.approx(ours / (ours + qb), abs=1e-4)
    assert snap.usd_per_day == pytest.approx(50 * ours / (ours + qb), abs=1e-3)
    assert snap.floor_ok == (snap.usd_per_day >= 1.0)
    assert snap.volume_24h == 0.0 and snap.band_turnover == 0.0  # fixture has no 24h volume


def test_assess_clamps_distance_to_the_touch(meta):
    # spread 6c: asking for 1c from mid is inside the spread -> placed at the touch (3c)
    bids = [(0.47, 100.0)]
    asks = [(0.53, 100.0)]
    snap = assess(meta, bids, asks, our_shares=100.0, our_dist_c=1.0)
    assert snap is not None
    assert snap.our_dist == pytest.approx(0.03)
    assert snap.our_points == 0.0  # 3c from mid == band edge for a 3c band -> unscored


def test_assess_rejects_empty_or_crossed_books(meta):
    assert assess(meta, [], [(0.5, 1)], 100, 2) is None
    assert assess(meta, [(0.5, 1)], [(0.5, 1)], 100, 2) is None


def test_book_points_and_parse_book():
    bids, asks = parse_book(
        {"bids": [{"price": "0.12", "size": "100"}], "asks": [{"price": "0.15", "size": "7"}]}
    )
    assert bids == [(0.12, 100.0)] and asks == [(0.15, 7.0)]
    assert book_points(bids, 0.135, 0.045, 50) == pytest.approx(order_points(0.015, 0.045, 100, 50))


def test_band_turnover_uses_24h_volume(meta):
    from dataclasses import replace

    m = replace(meta, volume_24hr=3000.0)
    snap = assess(m, [(0.49, 1000.0)], [(0.51, 1000.0)], 100.0, 2.0)
    assert snap is not None
    # 1000 bid shares at mid 0.50 + 1000 ask shares at 0.50 = 1000 USD resting -> eaten 3x a day
    assert snap.band_turnover == 3.0
