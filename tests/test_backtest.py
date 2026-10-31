"""Unit tests for the tape-replay backtester (synthetic tape, no network)."""

from __future__ import annotations

import dataclasses

from polymaker.backtest.data import Bar, MarketHistory, Tape, normalize_tape
from polymaker.backtest.sim import SimParams, run_sim
from polymaker.config import RiskConfig, StrategyProfile
from polymaker.domain import Side


def _hist(meta, bars, tape):  # type: ignore[no-untyped-def]
    return MarketHistory(meta=meta, tape=tape, bars=bars, start_ts=bars[0].ts, end_ts=bars[-1].ts)


def test_normalize_tape_maps_no_prints_to_yes_space(meta):
    rows = [
        {"asset": "yes-token", "side": "SELL", "price": "0.40", "size": "10", "timestamp": "100"},
        {"asset": "no-token", "side": "BUY", "price": "0.61", "size": "5", "timestamp": "50"},
        {"asset": "other", "side": "BUY", "price": "0.5", "size": "5", "timestamp": "60"},
    ]
    tape = normalize_tape(rows, meta)
    assert [t.ts for t in tape] == [50.0, 100.0]  # sorted, foreign asset dropped
    no_print = tape[0]
    assert no_print.aggressor is Side.SELL and abs(no_print.price - 0.39) < 1e-9
    assert no_print.raw_side is Side.BUY and no_print.raw_price == 0.61


def test_replay_fills_when_traded_through_and_scores_rewards(meta):
    """A flat 0.50 market: the quoter rests a bid below mid; a taker SELL print
    through that level must fill it, and resting in-band quotes must score."""
    profile = StrategyProfile(layers=1, base_size_usdc=20.0, q_max_usdc=200.0)
    meta = dataclasses.replace(meta, tick_size=0.001)  # fine tick: band edge rounding is not the subject
    bars = [Bar(60.0 * i, 0.50) for i in range(1, 40)]
    # one big sell through the whole bid stack 10 minutes in
    tape = [Tape(ts=600.5, aggressor=Side.SELL, price=0.40, size=500.0,
                 token_id="yes-token", raw_side=Side.SELL, raw_price=0.40)]
    r = run_sim(_hist(meta, bars, tape), profile, "p", RiskConfig(), SimParams(touch_depth=100.0), "t")
    assert r.fills, "traded-through bid must fill"
    f = r.fills[0]
    assert f.token_id == "yes-token" and f.side is Side.BUY and f.bought_yes
    assert f.price < 0.50  # filled at OUR resting price, below mid
    assert r.final_pos_yes > 0
    assert r.reward_minutes_scored > 0 and r.scored_fraction > 0.5
    assert r.placements >= 2  # both sides were quoted


def test_queue_model_drains_depth_before_filling(meta):
    """Prints AT our level must first consume the assumed queue ahead of us."""
    profile = StrategyProfile(layers=1, base_size_usdc=20.0, delta_min_ticks=1, min_edge_ticks=1)
    bars = [Bar(60.0 * i, 0.50) for i in range(1, 12)]
    # our YES bid sits at best bid (0.49 with a 2-tick spread); print at exactly that level
    small = Tape(300.5, Side.SELL, 0.49, 50.0, "yes-token", Side.SELL, 0.49)
    big = Tape(360.5, Side.SELL, 0.49, 500.0, "yes-token", Side.SELL, 0.49)
    r_small = run_sim(_hist(meta, bars, [small]), profile, "p", RiskConfig(),
                      SimParams(touch_depth=300.0), "t")
    r_big = run_sim(_hist(meta, bars, [big]), profile, "p", RiskConfig(),
                    SimParams(touch_depth=300.0), "t")
    assert not r_small.fills  # 50 shares < 300 ahead of us
    assert r_big.fills  # 500 shares drains the queue and reaches us


def test_engine_bug_toggle_changes_markout_bookkeeping(meta):
    """With engine_bugs the markout for a NO fill is recorded in token space and
    resolved against YES fair value, producing spurious toxicity; fixed does not."""
    profile = StrategyProfile(layers=1, base_size_usdc=20.0, q_max_usdc=500.0)
    bars = [Bar(60.0 * i, 0.20) for i in range(1, 30)]
    # taker BUY YES at 0.30 sweeps through our NO bid (= YES ask) -> we buy NO
    tape = [Tape(120.5, Side.BUY, 0.30, 400.0, "yes-token", Side.BUY, 0.30)]
    bug = run_sim(_hist(meta, bars, tape), profile, "p", RiskConfig(), SimParams(engine_bugs=True), "a")
    fix = run_sim(_hist(meta, bars, tape), profile, "p", RiskConfig(), SimParams(engine_bugs=False), "f")
    assert bug.fills and fix.fills
    assert bug.fills[0].token_id == "no-token"
    assert max(v for _, v in bug.tox_series) > 0.3
    assert max(v for _, v in fix.tox_series) < 0.05


def test_maker_fills_earn_rebates_on_fee_enabled_markets(meta):
    """Each maker fill is credited rebate_rate x the taker's fee (rate * p * (1-p) * size),
    and the credit flows into cash / total_pnl. Fee-free markets credit nothing."""
    profile = StrategyProfile(layers=1, base_size_usdc=20.0, q_max_usdc=200.0)
    bars = [Bar(60.0 * i, 0.50) for i in range(1, 40)]
    tape = [Tape(ts=600.5, aggressor=Side.SELL, price=0.40, size=500.0,
                 token_id="yes-token", raw_side=Side.SELL, raw_price=0.40)]
    params = SimParams(touch_depth=100.0)
    with_rebate = dataclasses.replace(meta, tick_size=0.001, rebate_rate=0.25)  # taker fee 1%
    r = run_sim(_hist(with_rebate, bars, tape), profile, "p", RiskConfig(), params, "t")
    assert r.fills and all(not f.taker for f in r.fills)
    for f in r.fills:
        assert abs(f.rebate_usd - 0.01 * 0.25 * f.price * (1.0 - f.price) * f.size) < 1e-9
    assert r.rebates_usd > 0 and abs(r.rebates_usd - sum(f.rebate_usd for f in r.fills)) < 1e-9
    no_rebate = dataclasses.replace(meta, tick_size=0.001, rebate_rate=0.0)
    r0 = run_sim(_hist(no_rebate, bars, tape), profile, "p", RiskConfig(), params, "t")
    assert r0.rebates_usd == 0.0
    assert abs((r.total_pnl - r0.total_pnl) - r.rebates_usd) < 1e-6


def test_bankroll_blocks_the_second_entry_it_cannot_fund(meta):
    """Two ~20 USD entries on a 25 USD bankroll: only the first is placed."""
    profile = StrategyProfile(layers=1, base_size_usdc=20.0, q_max_usdc=200.0)
    meta = dataclasses.replace(meta, tick_size=0.001)
    bars = [Bar(60.0 * i, 0.50) for i in range(1, 5)]
    free = run_sim(_hist(meta, bars, []), profile, "p", RiskConfig(), SimParams(), "t")
    tight = run_sim(_hist(meta, bars, []), profile, "p", RiskConfig(), SimParams(bankroll=25.0), "t")
    assert free.placements == 2 and tight.placements == 1
    enough = run_sim(_hist(meta, bars, []), profile, "p", RiskConfig(), SimParams(bankroll=45.0), "t")
    assert enough.placements == 2
