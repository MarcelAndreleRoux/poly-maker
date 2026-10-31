"""Payout reconciliation: earnings rows vs wallet balance snapshots across 00:00 UTC."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from polymaker.rewards import day_earnings, format_reconcile, journal_fill_ts, reconcile

DAY = "2026-10-29"
BOUNDARY = datetime(2026, 10, 30, tzinfo=UTC).timestamp()


def _row(ts, day, settled, earned, bal):
    return {
        "ts": ts,
        "day": day,
        "settled": settled,
        "total": [{"earnings": earned}],
        "markets": [{"earnings": earned}],
        "balances": {"clob_collateral": bal, "pusd": bal, "usdce": 0.0, "usdc": 0.0},
    }


def test_day_earnings_prefers_total_then_markets():
    assert day_earnings({"total": [{"earnings": 0.16}], "markets": [{"earnings": 0.1}]}) == 0.16
    assert day_earnings(
        {"total": [], "markets": [{"earnings": 0.1}, {"earnings": 0.05}]}
    ) == pytest.approx(0.15)
    assert day_earnings({"total": [], "markets": []}) == 0.0
    assert day_earnings({"total_error": "x"}) is None


def test_credited_when_balance_grows_by_the_earnings():
    rows = [
        _row(BOUNDARY - 3600, DAY, False, 0.16, 129.045),
        _row(BOUNDARY + 3600, DAY, True, 0.16, 129.205),
    ]
    rec = reconcile(rows, now=BOUNDARY + 7200)[0]
    assert rec["status"] == "credited"
    assert "clob_collateral" in rec["credited_to"]
    assert rec["floor_suspect"] is False


def test_pending_inside_grace_then_not_credited_with_floor_flag():
    rows = [
        _row(BOUNDARY - 3600, DAY, False, 0.16, 129.045),
        _row(BOUNDARY + 3600, DAY, True, 0.16, 129.045),
    ]
    assert reconcile(rows, now=BOUNDARY + 7200)[0]["status"] == "pending"
    late = reconcile(rows, now=BOUNDARY + 30 * 3600)[0]
    assert late["status"] == "not_credited"
    assert late["floor_suspect"] is True  # 0.16 < 1 USD floor
    big = [
        _row(BOUNDARY - 3600, DAY, False, 2.5, 100.0),
        _row(BOUNDARY + 3600, DAY, True, 2.5, 100.0),
    ]
    assert reconcile(big, now=BOUNDARY + 30 * 3600)[0]["floor_suspect"] is False


def test_no_data_and_zero_earnings_days_skipped():
    rows = [_row(BOUNDARY + 3600, DAY, True, 0.16, 129.0)]  # nothing before the boundary
    rec = reconcile(rows, now=BOUNDARY + 7200)[0]
    assert rec["status"] == "no_data" and rec["missing"] == "before"
    assert reconcile([_row(BOUNDARY + 3600, DAY, True, 0.0, 129.0)]) == []


def test_fills_in_window_flagged_and_latest_row_per_day_wins():
    rows = [
        _row(BOUNDARY - 3600, DAY, False, 0.10, 129.0),
        _row(BOUNDARY + 1800, DAY, True, 0.12, 129.0),
        _row(BOUNDARY + 3600, DAY, True, 0.16, 110.0),
    ]  # a fill moved collateral
    rec = reconcile(rows, now=BOUNDARY + 7200, fill_ts=[BOUNDARY + 100, BOUNDARY - 90000])[0]
    assert rec["earned"] == 0.16
    assert rec["fills_in_window"] == 1
    assert "not a clean payout" in format_reconcile(rec)


def test_journal_fill_ts_reads_only_user_trades(tmp_path):
    (tmp_path / "live.jsonl").write_text(
        json.dumps({"ts": 1.0, "kind": "user_trade", "data": {}})
        + "\n"
        + json.dumps({"ts": 2.0, "kind": "pnl", "data": {}})
        + "\n"
        + "garbage\n"
    )
    assert journal_fill_ts(tmp_path) == [1.0]
    assert journal_fill_ts(tmp_path / "missing") == []


def test_extra_snaps_from_journal_fill_the_gap(tmp_path):
    from polymaker.rewards import journal_collateral_snaps

    lines = [
        {"ts": BOUNDARY - 7200, "kind": "pnl", "data": {"collateral": 129.045}},
        {"ts": BOUNDARY - 7100, "kind": "pnl", "data": {"collateral": 129.045}},  # thinned out
        {"ts": BOUNDARY - 3600, "kind": "collateral", "data": {"pusd": 129.045}},
        {"ts": BOUNDARY - 3500, "kind": "cycle", "data": {"collateral": 1.0}},  # wrong kind
    ]
    (tmp_path / "live.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    snaps = journal_collateral_snaps(tmp_path)
    assert [ts for ts, _ in snaps] == [BOUNDARY - 7200, BOUNDARY - 3600]
    rows = [_row(BOUNDARY + 3600, DAY, True, 0.16, 129.205)]  # tracker only exists after midnight
    rec = reconcile(rows, now=BOUNDARY + 7200, extra_snaps=snaps)[0]
    assert rec["status"] == "credited"
