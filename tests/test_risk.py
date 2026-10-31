"""Tests for the RiskManager gates and circuit breakers."""

from __future__ import annotations

from polymaker.config import RiskConfig
from polymaker.domain import Fill, Side
from polymaker.risk.manager import KillFile, RiskManager
from polymaker.state.store import StateStore


def _rm(tmp_path, **over):
    cfg = RiskConfig(**{
        "max_total_exposure_usdc": 5000, "max_market_notional_usdc": 800,
        "max_event_group_loss_usdc": 1000, "daily_loss_kill_usdc": 250,
        **over,
    })
    store = StateStore(tmp_path / "s.db")
    return RiskManager(cfg, store), store


def test_daily_loss_kill_switch(tmp_path, meta):
    rm, store = _rm(tmp_path)
    # buy 1000 shares @ 0.50 -> -500 cash, +1000 inventory
    store.apply_fill(Fill(meta.yes.token_id, Side.BUY, 0.50, 1000, "t1"))
    rm.note_fill(Fill(meta.yes.token_id, Side.BUY, 0.50, 1000, "t1"))
    rm.update_mark(meta.yes.token_id, 0.50)
    rm.reset_day()
    assert rm.global_halt()[0] is False
    # fair value collapses to 0.20 -> unrealized loss 300 > 250 kill
    rm.update_mark(meta.yes.token_id, 0.20)
    halted, why = rm.global_halt()
    assert halted and "daily_loss" in why
    store.close()


def test_market_cap_triggers_reduce_only(tmp_path, meta):
    rm, store = _rm(tmp_path, max_market_notional_usdc=100)
    store.apply_fill(Fill(meta.yes.token_id, Side.BUY, 0.50, 300, "t1"))  # 150 notional > 100
    rm.update_mark(meta.yes.token_id, 0.50)
    rm.update_mark(meta.no.token_id, 0.50)
    d = rm.evaluate(meta, ws_stale=False, event_group_cost=0.0)
    assert d.reduce_only and d.reason == "market_cap"
    store.close()


def test_ws_stale_halts_market(tmp_path, meta):
    rm, store = _rm(tmp_path)
    d = rm.evaluate(meta, ws_stale=True, event_group_cost=0.0)
    assert d.halt and d.reason == "ws_stale"
    store.close()


def test_size_scale_tapers_near_cap(tmp_path, meta):
    rm, store = _rm(tmp_path, max_market_notional_usdc=100)
    # 85 notional -> 85% of cap -> should scale below 1.0 but not reduce-only
    store.apply_fill(Fill(meta.yes.token_id, Side.BUY, 0.50, 170, "t1"))  # 85 notional
    rm.update_mark(meta.yes.token_id, 0.50)
    rm.update_mark(meta.no.token_id, 0.50)
    d = rm.evaluate(meta, ws_stale=False, event_group_cost=0.0)
    assert not d.reduce_only
    assert 0.0 < d.size_scale < 1.0
    store.close()


def test_event_group_cap(tmp_path, meta):
    rm, store = _rm(tmp_path, max_event_group_loss_usdc=50)
    d = rm.evaluate(meta, ws_stale=False, event_group_cost=60.0)
    assert d.reduce_only and d.reason == "event_group_cap"
    store.close()


def test_error_rate_breaker(tmp_path, meta):
    rm, store = _rm(tmp_path, max_order_error_rate=0.25)
    for _ in range(15):
        rm.note_order_result(False)
    for _ in range(10):
        rm.note_order_result(True)  # 15/25 = 0.6 > 0.25
    assert rm.global_halt()[0] is True
    store.close()


def test_manual_kill(tmp_path, meta):
    rm, store = _rm(tmp_path)
    rm.kill()
    assert rm.global_halt() == (True, "manual_kill")
    store.close()


def test_total_loss_kill_switch(tmp_path, meta):
    rm, _ = _rm(tmp_path, total_loss_kill_usdc=40, start_equity_usdc=129)
    assert rm.check_total_loss(100.0, now=0.0) == ""
    assert rm.global_halt()[0] is False
    # 40 below the 129 start: needs 3 consecutive checks over >= 30 s
    assert rm.check_total_loss(89.0, now=100.0) == ""
    assert rm.check_total_loss(89.0, now=120.0) == ""
    assert rm.global_halt()[0] is False
    reason = rm.check_total_loss(88.0, now=140.0)
    assert reason.startswith("total_loss 41")
    assert rm.global_halt() == (True, reason)
    # sticky: a later recovery does not re-arm the engine
    assert rm.check_total_loss(200.0, now=200.0) == reason
    assert rm.global_halt()[0] is True


def test_total_loss_ignores_one_phantom_reading(tmp_path, meta):
    """2026-10-29 live: a disconnect read the balance as 0 for one poll."""
    rm, _ = _rm(tmp_path, total_loss_kill_usdc=40, start_equity_usdc=129)
    assert rm.check_total_loss(0.0, now=0.0) == ""      # strike 1
    assert rm.check_total_loss(129.0, now=20.0) == ""   # recovered -> strikes reset
    assert rm.check_total_loss(0.0, now=40.0) == ""     # strike 1 again
    assert rm.check_total_loss(0.0, now=45.0) == ""     # strike 2
    assert rm.check_total_loss(0.0, now=50.0) == ""     # strike 3 but only 10 s span
    assert rm.global_halt()[0] is False
    assert rm.check_total_loss(0.0, now=75.0).startswith("total_loss")  # 35 s span


def test_parse_collateral_raises_on_failed_read():
    import pytest

    from polymaker.execution.gateway import parse_collateral

    assert parse_collateral({"balance": "129040000"}) == 129.04
    assert parse_collateral({"balance": 61.58}) == 61.58
    for bad in ({}, None, {"error": "disconnected"}, {"balance": "n/a"}):
        with pytest.raises((RuntimeError, ValueError, TypeError)):
            parse_collateral(bad)


def test_total_loss_off_by_default(tmp_path, meta):
    rm, _ = _rm(tmp_path)
    assert rm.check_total_loss(-1000.0) == ""
    assert rm.global_halt()[0] is False


def test_kill_reason_propagates(tmp_path, meta):
    rm, _ = _rm(tmp_path)
    rm.kill("kill_file: total_loss 41")
    assert rm.global_halt() == (True, "kill_file: total_loss 41")


def test_kill_file_roundtrip(tmp_path):
    kf = KillFile(tmp_path / "deep" / "kill.json")
    assert not kf.exists() and kf.read() is None
    kf.write("total_loss 41 (equity 88 of 129)", 88.0)
    assert kf.exists()
    info = kf.read()
    assert info["reason"].startswith("total_loss 41") and info["equity"] == 88.0 and "when" in info
    kf.path.write_text("{not json")
    assert kf.read() == {"reason": "unreadable kill file"}
