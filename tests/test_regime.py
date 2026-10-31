"""Unit tests for the regime state machine."""

from __future__ import annotations

from polymaker.config import StrategyProfile
from polymaker.domain import Regime
from polymaker.strategy.regime import RegimeInputs, RegimeMachine


def _inp(**over):
    base = dict(
        now=1000.0,
        tick=0.01,
        fv=0.50,
        prev_fv=0.50,
        vol_ratio=1.0,
        flow_z=0.0,
        inventory_util=0.0,
        hours_to_end=1000.0,
    )
    base.update(over)
    return RegimeInputs(**base)


def test_default_is_quiet():
    p = StrategyProfile()
    assert RegimeMachine().decide(_inp(), p) == Regime.QUIET


def test_halts_take_priority():
    p = StrategyProfile()
    m = RegimeMachine()
    assert m.decide(_inp(risk_halt=True), p) == Regime.HALTED
    assert m.decide(_inp(ws_stale=True), p) == Regime.HALTED
    assert m.decide(_inp(market_resolved=True), p) == Regime.HALTED
    assert m.decide(_inp(hours_to_end=1.0), p) == Regime.HALTED  # inside halt window


def test_fv_jump_triggers_event_and_cooloff():
    p = StrategyProfile()  # event_jump_ticks=8, cooloff=60
    m = RegimeMachine()
    # jump of 0.10 = 10 ticks > 8 -> EVENT
    assert m.decide(_inp(fv=0.60, prev_fv=0.50), p) == Regime.EVENT
    # still in cooloff a moment later even without a jump
    assert m.decide(_inp(now=1030.0, fv=0.60, prev_fv=0.60), p) == Regime.EVENT
    # after cooloff expires, back to quiet
    assert m.decide(_inp(now=1100.0, fv=0.60, prev_fv=0.60), p) == Regime.QUIET


def test_sweep_flag_triggers_event():
    p = StrategyProfile()
    assert RegimeMachine().decide(_inp(sweep_flagged=True), p) == Regime.EVENT


def test_reduce_only_from_inventory_and_enddate():
    p = StrategyProfile()  # reduce_only_hours=24
    m = RegimeMachine()
    assert m.decide(_inp(inventory_util=1.0), p) == Regime.REDUCE_ONLY
    assert m.decide(_inp(risk_reduce_only=True), p) == Regime.REDUCE_ONLY
    assert m.decide(_inp(hours_to_end=12.0), p) == Regime.REDUCE_ONLY


def test_trending_from_flow_and_vol():
    p = StrategyProfile()  # trend_flow_z=1.5
    m = RegimeMachine()
    assert m.decide(_inp(flow_z=2.0), p) == Regime.TRENDING
    assert m.decide(_inp(vol_ratio=3.0), p) == Regime.TRENDING


def test_event_beats_reduce_only_and_trending():
    p = StrategyProfile()
    m = RegimeMachine()
    r = m.decide(_inp(sweep_flagged=True, inventory_util=1.0, flow_z=5.0), p)
    assert r == Regime.EVENT


# ── scheduled catalyst windows ──────────────────────────────────────────


def _win(start: float, end: float, flatten_hours=None, note="x"):
    from datetime import UTC, datetime

    from polymaker.config import CatalystWindow

    return CatalystWindow(
        start=datetime.fromtimestamp(start, tz=UTC),
        end=datetime.fromtimestamp(end, tz=UTC),
        note=note,
        flatten_hours=flatten_hours,
    )


def test_catalyst_window_halts_by_default():
    p = StrategyProfile()
    m = RegimeMachine()
    assert m.decide(_inp(catalyst_active=True), p) == Regime.HALTED
    # beats TRENDING / REDUCE_ONLY / sweep
    assert m.decide(_inp(catalyst_active=True, flow_z=9.0, sweep_flagged=True), p) == Regime.HALTED
    assert not m.in_cooloff  # a catalyst window never arms the EVENT cool-off


def test_catalyst_window_is_event_when_profile_keeps_exits():
    p = StrategyProfile(catalyst_keeps_exits=True)
    m = RegimeMachine()
    assert m.decide(_inp(catalyst_active=True), p) == Regime.EVENT
    assert not m.in_cooloff
    # window over -> straight back to normal, no lingering cool-off
    assert m.decide(_inp(), p) == Regime.QUIET


def test_catalyst_flatten_lead_in_is_reduce_only():
    p = StrategyProfile()
    m = RegimeMachine()
    assert m.decide(_inp(catalyst_flatten=True), p) == Regime.REDUCE_ONLY
    assert m.decide(_inp(catalyst_flatten=True, flow_z=9.0), p) == Regime.REDUCE_ONLY
    # a live sweep still wins over the lead-in
    assert m.decide(_inp(catalyst_flatten=True, sweep_flagged=True), p) == Regime.EVENT


def test_catalyst_state_timeline():
    from polymaker.strategy.regime import catalyst_state

    lead = 0.5  # profile default: 30 min
    # a lead-in that started before `now` (1000 - 1800 < 0) -> already flattening
    w = _win(1000.0, 2000.0, flatten_hours=None, note="debate")
    cs = catalyst_state([w], 0.0, lead)
    assert (cs.active, cs.flatten, cs.note) == (False, True, "debate")
    assert cs.next_change_s == 1000.0
    # well before: nothing, next change = lead-in start
    w2 = _win(10000.0, 11000.0, note="ruling")
    cs = catalyst_state([w2], 0.0, lead)
    assert (cs.active, cs.flatten) == (False, False)
    assert cs.next_change_s == 10000.0 - 1800.0
    # inside the lead-in
    cs = catalyst_state([w2], 9000.0, lead)
    assert (cs.active, cs.flatten, cs.note) == (False, True, "ruling")
    assert cs.next_change_s == 1000.0
    # inside the window
    cs = catalyst_state([w2], 10500.0, lead)
    assert (cs.active, cs.flatten, cs.note) == (True, False, "ruling")
    assert cs.next_change_s == 500.0
    # after: nothing ahead
    cs = catalyst_state([w2], 20000.0, lead)
    assert (cs.active, cs.flatten) == (False, False)
    assert cs.next_change_s == float("inf")


def test_catalyst_state_per_window_lead_overrides_profile_and_zero_disables():
    from polymaker.strategy.regime import catalyst_state

    w = _win(10000.0, 11000.0, flatten_hours=1.0)  # 1h lead regardless of profile
    assert catalyst_state([w], 10000.0 - 3000.0, 0.0).flatten is True
    assert catalyst_state([w], 10000.0 - 4000.0, 0.0).flatten is False
    w0 = _win(10000.0, 11000.0, flatten_hours=0.0)
    assert catalyst_state([w0], 9999.0, 5.0).flatten is False


def test_catalyst_state_overlapping_windows_active_wins():
    from polymaker.strategy.regime import catalyst_state

    a = _win(1000.0, 2000.0, note="a")
    b = _win(1500.0, 3000.0, note="b", flatten_hours=1.0)
    cs = catalyst_state([a, b], 1200.0, 0.0)  # inside a, in b's lead-in
    assert (cs.active, cs.flatten, cs.note) == (True, False, "a")
    assert cs.next_change_s == 300.0  # b starts at 1500 -> next boundary
    cs = catalyst_state([a, b], 2500.0, 0.0)  # a over, inside b
    assert (cs.active, cs.note) == (True, "b")
    assert cs.next_change_s == 500.0


def test_fair_value_outside_band_is_reduce_only():
    p = StrategyProfile()  # fv_band 0.10-0.90 by default
    m = RegimeMachine()
    assert m.decide(_inp(fv=0.05, prev_fv=0.05), p) == Regime.REDUCE_ONLY
    assert m.decide(_inp(fv=0.95, prev_fv=0.95), p) == Regime.REDUCE_ONLY
    assert m.decide(_inp(fv=0.10, prev_fv=0.10), p) == Regime.QUIET  # edges inclusive
    assert m.decide(_inp(fv=0.90, prev_fv=0.90), p) == Regime.QUIET
    off = StrategyProfile(fv_band_min=0.0, fv_band_max=1.0)
    assert m.decide(_inp(fv=0.05, prev_fv=0.05), off) == Regime.QUIET
