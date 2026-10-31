"""Taker tape from poly_data's on-chain `OrderFilled` dump (`data/orderFilled.csv`).

The public data-api caps trade history at 10k prints per market; the chain has
every fill since the V2 exchange went live (2026-04-28). poly_data
(github.com/warproxxx/poly_data) streams it into an append-only CSV:

  timestamp,maker,makerAssetId,makerAmountFilled,taker,takerAssetId,takerAmountFilled,transactionHash

The exchange emits one `OrderFilled` per MAKER order matched plus one for the
TAKER order itself, whose `taker` field is the exchange contract. Binary markets
fill on the CTF Exchange V2, multi-outcome (neg-risk) events on the Neg-Risk CTF
Exchange V2; same event, different contract, so both count. Only the latter
is kept: one print per taker order, on the asset the taker traded, at the price
it actually paid and with its whole size - what the live `last_trade_price`
feed and the data-api tape show, and what the sweep detector must see intact.
The maker rows are the other side of the same print (in a mint/merge match they
sit on the complementary token at the makers' own limit prices), so taking them
too would double count.

Per-market prints are cached under <cache_root>/<cid[:12]>/polydata_tape.json
with the byte offset consumed, so a re-run reads only what poly_data appended.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polymaker.domain import MarketMeta
from polymaker.logging import get_logger

log = get_logger("backtest.polydata")

EXCHANGE_V2 = "0xe111180000d2663c0091e4f400237545b87b996b"
NEG_RISK_EXCHANGE_V2 = "0xe2222d279d744050d28e00520010520000310f59"
EXCHANGES = frozenset({EXCHANGE_V2, NEG_RISK_EXCHANGE_V2})
COLUMNS = (
    "timestamp", "maker", "makerAssetId", "makerAmountFilled", "taker", "takerAssetId",
    "takerAmountFilled", "transactionHash",
)
CACHE_NAME = "polydata_tape.json"
_UNIT = 1_000_000  # USDC and CTF outcome tokens both carry 6 decimals

Print = dict[str, Any]  # data-api shaped: timestamp, asset, side, price, size, transactionHash


def taker_print(row: list[str]) -> Print | None:
    """Decode a taker-order row (taker == exchange) into a print; None otherwise.

    On that row `maker` is the aggressor's own order: paying USDC (makerAssetId
    "0") means it BOUGHT `takerAssetId`; paying tokens means it SOLD them.
    """
    if len(row) < 8 or row[4] not in EXCHANGES:
        return None
    try:
        ts = float(row[0])
        if row[2] == "0":
            usdc, token_id, tokens, side = int(row[3]), row[5], int(row[6]), "BUY"
        else:
            token_id, tokens, usdc, side = row[2], int(row[3]), int(row[6]), "SELL"
    except ValueError:
        return None
    if tokens <= 0 or usdc <= 0 or token_id == "0":
        return None
    return {
        "timestamp": ts, "asset": token_id, "side": side,
        "price": round(usdc / tokens, 6), "size": round(tokens / _UNIT, 6),
        "transactionHash": row[7],
    }


@dataclass(slots=True)
class _Slot:
    cid: str
    cache: Path
    offset: int  # bytes of `src` already folded into `rows`
    rows: list[Print]


def _load_cache(cache: Path, src: Path, size: int) -> tuple[int, list[Print]]:
    if not cache.exists():
        return 0, []
    try:
        with cache.open() as fh:
            c = json.load(fh)
        offset = int(c.get("offset", 0))
        if c.get("src") != str(src) or offset <= 0 or offset > size:
            return 0, []
        with src.open("rb") as fh:  # the source must still end a line where we stopped
            fh.seek(offset - 1)
            if fh.read(1) != b"\n":
                return 0, []
        rows: list[Print] = list(c.get("rows", []))
        return offset, rows
    except (OSError, ValueError, TypeError):
        return 0, []


def _write_cache(cache: Path, src: Path, offset: int, rows: list[Print]) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump({"src": str(src), "offset": offset, "rows": rows}, fh)
    tmp.replace(cache)


def extract_taker_prints(
    src: Path, metas: list[MarketMeta], cache_root: Path
) -> dict[str, list[Print]]:
    """One pass over `src` for every market in `metas` -> {condition_id: prints}.

    Append-only source: each market's cache records the byte offset it has
    consumed, and only bytes past the smallest cached offset are read. A
    partial trailing line (a batch poly_data is still writing) is left for the
    next run. A source rewritten under us (offset no longer on a line boundary,
    or the file shrank) is re-read from the top.
    """
    src = Path(src)
    size = src.stat().st_size
    slots: list[_Slot] = []
    by_token: dict[str, list[_Slot]] = {}
    for m in metas:
        cache = Path(cache_root) / m.condition_id[:12] / CACHE_NAME
        offset, rows = _load_cache(cache, src, size)
        s = _Slot(m.condition_id, cache, offset, rows)
        slots.append(s)
        for tok in (m.yes.token_id, m.no.token_id):
            by_token.setdefault(tok, []).append(s)
    if not slots:
        return {}
    start = min(s.offset for s in slots)
    if start >= size:
        return {s.cid: s.rows for s in slots}

    exs = [e.encode() for e in EXCHANGES]
    needles = [t.encode() for t in by_token]
    added = 0
    with src.open("rb") as fh:
        if start == 0:
            header = fh.readline()
            cols = tuple(header.decode(errors="replace").rstrip("\r\n").split(","))
            if cols != COLUMNS:
                raise ValueError(f"{src}: unexpected header {cols!r}")
            pos = len(header)
        else:
            fh.seek(start)
            pos = start
        consumed = pos
        for line in fh:
            line_start = pos
            pos += len(line)
            if not line.endswith(b"\n"):
                break
            consumed = pos
            if not any(e in line for e in exs) or not any(n in line for n in needles):
                continue
            p = taker_print(next(csv.reader([line.decode(errors="replace")])))
            if p is None:
                continue
            for s in by_token.get(p["asset"], ()):
                if line_start >= s.offset:
                    s.rows.append(p)
                    added += 1
    for s in slots:
        s.offset = consumed
        _write_cache(s.cache, src, consumed, s.rows)
    log.info("polydata_tape_extracted", src=str(src), bytes_read=consumed - start,
             prints_added=added, markets=len(slots))
    return {s.cid: s.rows for s in slots}
