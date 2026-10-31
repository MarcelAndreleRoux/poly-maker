"""poly_data orderFilled.csv -> taker tape: decode, filter, incremental cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from polymaker.backtest.data import normalize_tape
from polymaker.backtest.polydata import (
    COLUMNS,
    EXCHANGE_V2,
    NEG_RISK_EXCHANGE_V2,
    extract_taker_prints,
    taker_print,
)
from polymaker.domain import Side

EX = EXCHANGE_V2
YES, NO, OTHER = "yes-token", "no-token", "other-token"
U = 1_000_000


def _row(ts, maker, m_asset, m_amt, taker, t_asset, t_amt, tx):  # type: ignore[no-untyped-def]
    return ",".join([str(ts), maker, m_asset, str(int(m_amt)), taker, t_asset, str(int(t_amt)), tx])


def _buy(ts, tok, usdc, shares, tx, taker=EX):  # aggressor pays USDC, receives `tok`
    return _row(ts, "0xaggr", "0", usdc * U, taker, tok, shares * U, tx)


def _sell(ts, tok, shares, usdc, tx, taker=EX):  # aggressor gives `tok`, receives USDC
    return _row(ts, "0xaggr", tok, shares * U, taker, "0", usdc * U, tx)


def _maker(ts, tok, usdc, shares, tx):  # a maker's leg: taker is a wallet, not the exchange
    return _row(ts, "0xmaker", "0", usdc * U, "0xtakerwallet", tok, shares * U, tx)


def _write(path: Path, lines: list[str], *, partial: str | None = None) -> None:
    body = ",".join(COLUMNS) + "\n" + "".join(ln + "\n" for ln in lines)
    if partial is not None:
        body += partial  # no trailing newline: still being written
    path.write_bytes(body.encode())


def test_taker_print_decodes_both_sides_and_ignores_maker_rows():
    buy = taker_print(_buy(100, YES, 4.2, 10, "0xa").split(","))
    assert buy == {"timestamp": 100.0, "asset": YES, "side": "BUY", "price": 0.42, "size": 10.0,
                   "transactionHash": "0xa"}
    sell = taker_print(_sell(101, NO, 5, 2.5, "0xb").split(","))
    assert sell is not None and (sell["side"], sell["price"], sell["size"]) == ("SELL", 0.5, 5.0)
    assert taker_print(_maker(100, YES, 4.2, 10, "0xa").split(",")) is None
    assert taker_print(_buy(100, YES, 0, 10, "0xa").split(",")) is None  # zero notional
    assert taker_print(["garbage"]) is None


def test_taker_print_accepts_rows_from_the_neg_risk_exchange():
    # multi-outcome events fill on the neg-risk exchange: same row shape, other taker
    buy = taker_print(_buy(100, YES, 4.2, 10, "0xa", taker=NEG_RISK_EXCHANGE_V2).split(","))
    assert buy is not None and (buy["side"], buy["price"], buy["size"]) == ("BUY", 0.42, 10.0)
    sell = taker_print(_sell(101, NO, 5, 2.5, "0xb", taker=NEG_RISK_EXCHANGE_V2).split(","))
    assert sell is not None and sell["side"] == "SELL"
    assert taker_print(_buy(100, YES, 4.2, 10, "0xa", taker="0xsomewallet").split(",")) is None


def test_extract_reads_neg_risk_prints_through_the_prefilter(meta, tmp_path: Path):
    src = tmp_path / "orderFilled.csv"
    _write(src, [
        _buy(100, YES, 4.2, 10, "0xa", taker=NEG_RISK_EXCHANGE_V2),
        _maker(100, YES, 4.2, 10, "0xa"),
        _sell(101, NO, 5, 2.5, "0xb", taker=NEG_RISK_EXCHANGE_V2),
    ])
    out = extract_taker_prints(src, [meta], tmp_path / "cache")
    rows = out[meta.condition_id]
    assert [(r["asset"], r["side"]) for r in rows] == [(YES, "BUY"), (NO, "SELL")]


def test_extract_keeps_market_tokens_only_and_normalizes_to_yes_space(meta, tmp_path: Path):
    src = tmp_path / "orderFilled.csv"
    _write(src, [
        _maker(10, YES, 4.0, 10, "0x1"),       # maker leg of the print below: dropped
        _buy(10, YES, 4.0, 10, "0x1"),         # taker BUY 10 YES @ 0.40
        _buy(20, OTHER, 1.0, 2, "0x2"),        # another market: dropped
        _buy(30, NO, 3.0, 5, "0x3"),           # taker BUY 5 NO @ 0.60 == SELL YES @ 0.40
        _sell(40, YES, 8, 3.2, "0x4"),         # taker SELL 8 YES @ 0.40
    ])
    out = extract_taker_prints(src, [meta], tmp_path / "cache")
    rows = out[meta.condition_id]
    assert [r["transactionHash"] for r in rows] == ["0x1", "0x3", "0x4"]
    tape = normalize_tape(rows, meta)
    assert [(t.ts, t.aggressor, round(t.price, 6), t.size) for t in tape] == [
        (10.0, Side.BUY, 0.40, 10.0),
        (30.0, Side.SELL, 0.40, 5.0),  # NO print flipped into YES space
        (40.0, Side.SELL, 0.40, 8.0),
    ]
    assert tape[1].raw_side is Side.BUY and tape[1].raw_price == 0.6 and tape[1].token_id == NO


def test_extract_is_incremental_and_leaves_a_partial_line(meta, tmp_path: Path):
    src = tmp_path / "orderFilled.csv"
    cache = tmp_path / "cache"
    half = _buy(50, YES, 1.0, 2, "0x5")
    _write(src, [_buy(10, YES, 4.0, 10, "0x1")], partial=half[: len(half) // 2])
    first = extract_taker_prints(src, [meta], cache)[meta.condition_id]
    assert [r["transactionHash"] for r in first] == ["0x1"]
    # poly_data finishes the batch and appends another print
    _write(src, [_buy(10, YES, 4.0, 10, "0x1"), half, _sell(60, NO, 3, 1.5, "0x6")])
    second = extract_taker_prints(src, [meta], cache)[meta.condition_id]
    assert [r["transactionHash"] for r in second] == ["0x1", "0x5", "0x6"]
    # nothing new -> cache answers without re-reading (same result, no duplicates)
    third = extract_taker_prints(src, [meta], cache)[meta.condition_id]
    assert third == second
    # source rewritten shorter (poly_data discarded an interrupted batch) -> full re-read
    _write(src, [_buy(10, YES, 4.0, 10, "0x1"), _buy(11, YES, 4.0, 10, "0x7")])
    fourth = extract_taker_prints(src, [meta], cache)[meta.condition_id]
    assert [r["transactionHash"] for r in fourth] == ["0x1", "0x7"]


def test_extract_one_pass_serves_several_markets(meta, tmp_path: Path):
    import dataclasses

    from polymaker.domain import TokenMeta

    other = dataclasses.replace(meta, condition_id="0xother",
                                tokens=(TokenMeta("o-yes", "Yes"), TokenMeta("o-no", "No")))
    src = tmp_path / "orderFilled.csv"
    _write(src, [_buy(10, YES, 4.0, 10, "0x1"), _buy(20, "o-no", 1.0, 2, "0x2")])
    out = extract_taker_prints(src, [meta, other], tmp_path / "cache")
    assert [r["asset"] for r in out[meta.condition_id]] == [YES]
    assert [r["asset"] for r in out["0xother"]] == ["o-no"]
    # a market whose cache is behind the other's catches up on the shared pass
    _write(src, [_buy(10, YES, 4.0, 10, "0x1"), _buy(20, "o-no", 1.0, 2, "0x2"),
                 _buy(30, "o-yes", 1.0, 2, "0x3"), _buy(31, NO, 1.0, 2, "0x4")])
    extract_taker_prints(src, [other], tmp_path / "cache")  # only `other` advances
    out = extract_taker_prints(src, [meta, other], tmp_path / "cache")
    assert [r["transactionHash"] for r in out[meta.condition_id]] == ["0x1", "0x4"]
    assert [r["transactionHash"] for r in out["0xother"]] == ["0x2", "0x3"]


def test_extract_rejects_foreign_header(meta, tmp_path: Path):
    src = tmp_path / "x.csv"
    src.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="unexpected header"):
        extract_taker_prints(src, [meta], tmp_path / "cache")
