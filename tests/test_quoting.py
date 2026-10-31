"""Unit tests for pure quote construction — the strategy's decision core."""

from __future__ import annotations

import pytest

from polymaker.config import StrategyProfile
from polymaker.domain import Position, Regime, Side
from polymaker.strategy.quoting import (
    QuoteInputs,
    compute_fair_value,
    construct_quotes,
    round_to_tick,
)
from tests.conftest import view


def _inputs(meta, profile, **over):
    base = dict(
        meta=meta,
        regime=Regime.QUIET,
        fv=0.50,
        vol_short=0.0,
        toxicity=0.0,
        yes_view=view(0.49, 0.51),
        no_view=view(0.49, 0.51),
        pos_yes=Position("yes-token"),
        pos_no=Position("no-token"),
        profile=profile,
        now=1000.0,
    )
    base.update(over)
    return QuoteInputs(**base)


# ── round_to_tick ──────────────────────────────────────────────────────────


def test_round_to_tick_down_and_up():
    assert round_to_tick(0.5049, 0.01, 2, up=False) == 0.50
    assert round_to_tick(0.5051, 0.01, 2, up=True) == 0.51
    # clamps inside (0,1)
    assert round_to_tick(0.0, 0.01, 2, up=False) == 0.01
    assert round_to_tick(1.0, 0.01, 2, up=True) == 0.99


def test_compute_fair_value_flow_nudge():
    # positive flow nudges FV up, negative down, no flow = microprice
    assert compute_fair_value(0.50, 0.0, 0.01) == pytest.approx(0.50)
    assert compute_fair_value(0.50, 1.0, 0.01, weight=0.5) == pytest.approx(0.505)
    assert compute_fair_value(0.50, -1.0, 0.01, weight=0.5) == pytest.approx(0.495)


# ── two-sided quoting ────────────────────────────────────────────────────────


def test_quiet_market_quotes_both_sides_as_bids(meta, profile):
    tq = construct_quotes(_inputs(meta, profile))
    assert tq.regime == Regime.QUIET
    yes = [q for q in tq.quotes if q.token_id == "yes-token"]
    no = [q for q in tq.quotes if q.token_id == "no-token"]
    assert yes and no
    # both entry quotes are BUYs (USDC-collateralized two-sided quote)
    assert all(q.side == Side.BUY for q in yes)
    assert all(q.side == Side.BUY for q in no)


def test_pair_prices_sum_below_one(meta, profile):
    """BUY YES @ p and BUY NO @ q must satisfy p + q < 1 (merge edge)."""
    tq = construct_quotes(_inputs(meta, profile))
    top_yes = max(q.price for q in tq.quotes if q.token_id == "yes-token")
    top_no = max(q.price for q in tq.quotes if q.token_id == "no-token")
    assert top_yes + top_no < 1.0


def test_never_bids_through_fair_value(meta, profile):
    """No BUY should ever sit at or above FV - min_edge (YES) / (1-FV)-min_edge (NO)."""
    tq = construct_quotes(_inputs(meta, profile, fv=0.50))
    edge = profile.min_edge_ticks * meta.tick_size
    for q in tq.quotes:
        if q.side == Side.BUY and q.token_id == "yes-token":
            assert q.price <= 0.50 - edge + 1e-9
        if q.side == Side.BUY and q.token_id == "no-token":
            assert q.price <= 0.50 - edge + 1e-9  # NO fv is also 0.50 here


def test_layers_split_size(meta, profile):
    tq = construct_quotes(_inputs(meta, profile))
    yes = sorted((q for q in tq.quotes if q.token_id == "yes-token" and q.side == Side.BUY),
                 key=lambda q: -q.price)
    assert len(yes) == profile.layers
    # deeper layer is at a lower price
    assert yes[0].price > yes[1].price


# ── inventory skew ──────────────────────────────────────────────────────────


def test_long_yes_inventory_skews_quotes_down(meta, profile):
    """Holding YES should lower the YES bid and lower the implied YES ask (the
    leaned SELL YES exit, or 1 - NO bid) vs flat, so the market is invited to
    take the inventory off us."""
    flat = construct_quotes(_inputs(meta, profile, vol_short=0.02))
    longy = construct_quotes(
        _inputs(meta, profile, vol_short=0.02, pos_yes=Position("yes-token", 300, 0.5))
    )

    def top_bid(tq, tok):
        ps = [q.price for q in tq.quotes if q.token_id == tok and q.side == Side.BUY]
        return max(ps) if ps else None

    def implied_ask(tq):
        asks = [q.price for q in tq.quotes if q.token_id == "yes-token" and q.side == Side.SELL]
        asks += [1.0 - q.price for q in tq.quotes if q.token_id == "no-token" and q.side == Side.BUY]
        return min(asks)

    assert top_bid(longy, "yes-token") <= top_bid(flat, "yes-token")
    assert implied_ask(longy) <= implied_ask(flat)
    # the flatten order (SELL YES) is posted; the pair-adding BUY NO is shrunk by it
    assert any(q.token_id == "yes-token" and q.side == Side.SELL for q in longy.quotes)


def test_skew_floor_leans_on_quiet_tape(meta):
    """With sigma = 0 the old skew was exactly zero; the floor keeps a lean."""
    no_floor = StrategyProfile(skew_min_ticks=0.0)
    floor = StrategyProfile(skew_min_ticks=4.0)
    pos = Position("yes-token", 300, 0.5)  # u = 300*0.5/500 = 0.3 -> ~1.2 ticks of lean
    a = construct_quotes(_inputs(meta, no_floor, vol_short=0.0, pos_yes=pos))
    b = construct_quotes(_inputs(meta, floor, vol_short=0.0, pos_yes=pos))
    sell_a = min(q.price for q in a.quotes if q.side == Side.SELL)
    sell_b = min(q.price for q in b.quotes if q.side == Side.SELL)
    assert sell_b <= sell_a
    bid_a = max(q.price for q in a.quotes if q.token_id == "yes-token" and q.side == Side.BUY)
    bid_b = max(q.price for q in b.quotes if q.token_id == "yes-token" and q.side == Side.BUY)
    assert bid_b <= bid_a


def test_exit_leans_with_inventory(meta, profile):
    """The SELL exit is priced off the reservation price r, not fair value: a
    bigger bag -> a lower (more aggressive) exit."""
    small = construct_quotes(_inputs(meta, StrategyProfile(skew_min_ticks=6.0), vol_short=0.0,
                                     pos_yes=Position("yes-token", 50, 0.5)))
    big = construct_quotes(_inputs(meta, StrategyProfile(skew_min_ticks=6.0), vol_short=0.0,
                                   pos_yes=Position("yes-token", 450, 0.5)))
    ask_small = min(q.price for q in small.quotes if q.side == Side.SELL)
    ask_big = min(q.price for q in big.quotes if q.side == Side.SELL)
    assert ask_big < ask_small


def test_inventory_util_is_capital_at_risk():
    from polymaker.strategy.quoting import inventory_util

    # 1000 NO shares on a 0.19 market risk 810 USD, not 190
    assert inventory_util(-1000, 0.19, 500.0) == -1.0
    assert abs(inventory_util(1000, 0.19, 500.0) - 0.38) < 1e-9
    assert inventory_util(0, 0.5, 500.0) == 0.0


def test_reduce_only_emits_only_exits(meta, profile):
    tq = construct_quotes(
        _inputs(
            meta, profile, regime=Regime.REDUCE_ONLY,
            pos_yes=Position("yes-token", 100, 0.5),
        )
    )
    assert all(q.side == Side.SELL for q in tq.quotes)
    assert any(q.token_id == "yes-token" for q in tq.quotes)


def test_halted_pulls_all_quotes_event_keeps_exits(meta, profile):
    halted = construct_quotes(
        _inputs(meta, profile, regime=Regime.HALTED, pos_yes=Position("yes-token", 100, 0.5))
    )
    assert halted.is_empty
    event = construct_quotes(
        _inputs(meta, profile, regime=Regime.EVENT, pos_yes=Position("yes-token", 100, 0.5))
    )
    assert not [q for q in event.quotes if q.side == Side.BUY]  # entries pulled
    assert [q for q in event.quotes if q.side == Side.SELL]  # exit still working


# ── trade-out ────────────────────────────────────────────────────────────────


def _to_profile(**over):
    base = dict(trade_out_enabled=True, trade_out_drift_ticks=5.0, trade_out_max_slip_ticks=2,
                trade_out_depth_frac=0.5, exit_urgency_s=600.0)
    base.update(over)
    return StrategyProfile(**base)


def test_trade_out_requires_drift_and_a_forcing_condition(meta):
    p = _to_profile()
    under_water = Position("yes-token", 100, 0.60)  # fv 0.50 -> 10 ticks under
    # drift alone (10 >= 5 but < 2x) with no urgency, QUIET -> no trade-out
    tq = construct_quotes(_inputs(meta, p, pos_yes=under_water, yes_exit_urgency=0.0))
    assert not tq.trade_outs
    # urgent -> fires, bounded to half the best-bid size, worst = bid - 2 ticks
    tq = construct_quotes(_inputs(meta, p, pos_yes=under_water, yes_exit_urgency=1.0,
                                  yes_view=view(0.49, 0.51, bb_sz=60, ba_sz=60)))
    assert len(tq.trade_outs) == 1
    t = tq.trade_outs[0]
    assert t.token_id == "yes-token" and t.size == 30.0 and abs(t.worst_price - 0.47) < 1e-9
    # REDUCE_ONLY forces it too; HALTED never; cooldown gate blocks it
    assert construct_quotes(_inputs(meta, p, regime=Regime.REDUCE_ONLY, pos_yes=under_water)).trade_outs
    assert not construct_quotes(_inputs(meta, p, regime=Regime.HALTED, pos_yes=under_water)).trade_outs
    assert not construct_quotes(_inputs(meta, p, pos_yes=under_water, yes_exit_urgency=1.0,
                                        trade_out_allowed=False)).trade_outs


def test_trade_out_not_when_in_profit_or_disabled(meta):
    in_profit = Position("yes-token", 100, 0.40)  # fv 0.50 -> above cost
    tq = construct_quotes(_inputs(meta, _to_profile(), pos_yes=in_profit, yes_exit_urgency=1.0))
    assert not tq.trade_outs
    under_water = Position("yes-token", 100, 0.60)
    tq = construct_quotes(_inputs(meta, _to_profile(trade_out_enabled=False), pos_yes=under_water,
                                  yes_exit_urgency=1.0))
    assert not tq.trade_outs


def test_trade_out_fires_on_double_drift_without_urgency(meta):
    deep = Position("yes-token", 100, 0.62)  # 12 ticks under >= 2 x 5
    tq = construct_quotes(_inputs(meta, _to_profile(), pos_yes=deep, yes_exit_urgency=0.0))
    assert tq.trade_outs and tq.trade_outs[0].reason == "drift"


# ── exits ────────────────────────────────────────────────────────────────────


def test_exit_sell_priced_above_fv_when_not_urgent(meta, profile):
    tq = construct_quotes(
        _inputs(meta, profile, pos_yes=Position("yes-token", 100, 0.4), yes_exit_urgency=0.0)
    )
    sells = [q for q in tq.quotes if q.side == Side.SELL and q.token_id == "yes-token"]
    assert sells
    assert sells[0].price >= 0.50  # at/above FV, a passive maker exit


def test_exit_never_below_best_bid(meta, profile):
    tq = construct_quotes(
        _inputs(
            meta, profile,
            pos_yes=Position("yes-token", 100, 0.4),
            yes_view=view(0.49, 0.51),
            yes_exit_urgency=1.0,  # maximally urgent
        )
    )
    sells = [q for q in tq.quotes if q.side == Side.SELL and q.token_id == "yes-token"]
    assert sells
    assert sells[0].price >= 0.49  # still a maker order, never crosses down


def test_no_exit_when_position_is_dust(meta, profile):
    tq = construct_quotes(
        _inputs(meta, profile, pos_yes=Position("yes-token", 1.0, 0.4))  # below min_order_size
    )
    assert not [q for q in tq.quotes if q.side == Side.SELL]


# ── spread widening ──────────────────────────────────────────────────────────


def test_toxicity_widens_spread(meta, profile):
    """Higher toxicity should push the YES bid lower (wider spread)."""
    calm = construct_quotes(_inputs(meta, profile, regime=Regime.TRENDING, toxicity=0.0))
    toxic = construct_quotes(_inputs(meta, profile, regime=Regime.TRENDING, toxicity=0.02))

    def top_yes(tq):
        ps = [q.price for q in tq.quotes if q.token_id == "yes-token" and q.side == Side.BUY]
        return max(ps) if ps else None

    assert top_yes(toxic) < top_yes(calm)


def test_quiet_regime_clamps_spread_to_reward_band(meta, profile):
    """In QUIET, even with high vol the bid stays within the reward band of FV."""
    tq = construct_quotes(_inputs(meta, profile, regime=Regime.QUIET, vol_short=0.5))
    band = meta.rewards_max_spread / 100.0  # 0.03
    top_yes = max(q.price for q in tq.quotes if q.token_id == "yes-token" and q.side == Side.BUY)
    # bid should be within (band + a tick of rounding) of FV
    assert top_yes >= 0.50 - band - meta.tick_size


def test_base_size_shares_sizes_both_sides_in_shares(meta):
    """A share budget lands the same size on the cheap YES and the expensive NO
    (what the reward program counts); the USDC budget would not."""
    p = StrategyProfile(base_size_shares=50.0, layers=1, reward_size_mult=1.0, q_max_usdc=500.0)
    tq = construct_quotes(_inputs(meta, p, fv=0.20, yes_view=view(0.19, 0.21), no_view=view(0.79, 0.81)))
    buys = {q.token_id: q for q in tq.quotes if q.side is Side.BUY}
    assert buys["yes-token"].size == 50.0 and buys["no-token"].size == 50.0
    assert buys["yes-token"].price * 50 + buys["no-token"].price * 50 < 50.0  # < 50 USD resting
    usdc = StrategyProfile(base_size_usdc=8.0, layers=1, reward_size_mult=1.0, q_max_usdc=500.0)
    tq2 = construct_quotes(_inputs(meta, usdc, fv=0.20, yes_view=view(0.19, 0.21), no_view=view(0.79, 0.81)))
    sizes = {q.token_id: q.size for q in tq2.quotes if q.side is Side.BUY}
    assert sizes["no-token"] < sizes["yes-token"]  # 8 USD buys ~10 NO vs ~42 YES


def test_fit_entries_to_collateral_drops_only_unfundable_buys():
    from polymaker.domain import Quote
    from polymaker.strategy.quoting import fit_entries_to_collateral

    sell = Quote("yes-token", Side.SELL, 0.30, 100.0)   # exit: locks nothing
    buy_a = Quote("yes-token", Side.BUY, 0.20, 50.0)    # 10 USD
    buy_b = Quote("no-token", Side.BUY, 0.80, 50.0)     # 40 USD
    kept, dropped = fit_entries_to_collateral([sell, buy_a, buy_b], available=49.0)
    assert kept == [sell, buy_a] and dropped == [buy_b]
    kept, dropped = fit_entries_to_collateral([sell, buy_a, buy_b], available=50.0)
    assert kept == [sell, buy_a, buy_b] and dropped == []
    kept, dropped = fit_entries_to_collateral([buy_a, sell], available=0.0)
    assert kept == [sell] and dropped == [buy_a]
