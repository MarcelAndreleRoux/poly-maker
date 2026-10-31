"""Historical inputs for the backtester, with a JSON disk cache.

Three public sources, no auth:
  * Gamma `/markets?slug=`          -> MarketMeta (tick, reward params, tokens)
  * data-api `/trades?market=`      -> taker trade tape (max 10k offset, newest first)
  * CLOB `/prices-history`          -> minute mid history (<=14-day windows at fidelity 1)
Or, for the tape, poly_data's on-chain `orderFilled.csv` (every fill since V2
genesis, no cap) via `polydata=`; see backtest/polydata.py.

Everything is cached under <cache_root>/<condition_id[:12]>/ so a re-run is
offline and instant. Trades are normalized to YES space (a taker BUY of NO at q
is a taker SELL of YES at 1-q) because the strategy core thinks in YES prices.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from polymaker.catalog.gamma import fetch_reward_rates, parse_market
from polymaker.domain import MarketMeta, Side
from polymaker.logging import get_logger

log = get_logger("backtest.data")

GAMMA = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

_TRADES_PAGE = 1000
_TRADES_MAX_OFFSET = 10000
_PRICE_WINDOW_S = 13 * 86400  # the API rejects startTs/endTs spans much past 2 weeks
DEFAULT_TAPE_SOURCE = "data-api /trades (newest 11k prints)"


@dataclass(frozen=True, slots=True)
class Tape:
    """One taker print, in YES space."""

    ts: float
    aggressor: Side  # taker side in YES space: BUY lifts YES asks, SELL hits YES bids
    price: float  # YES price
    size: float  # shares
    token_id: str  # asset actually traded on the tape
    raw_side: Side  # taker side on that asset (what the live engine sees)
    raw_price: float  # price on that asset


@dataclass(frozen=True, slots=True)
class Bar:
    ts: float
    mid: float  # YES mid


@dataclass(slots=True)
class MarketHistory:
    meta: MarketMeta
    tape: list[Tape]
    bars: list[Bar]
    start_ts: float
    end_ts: float
    tape_source: str = DEFAULT_TAPE_SOURCE

    @property
    def days(self) -> float:
        return (self.end_ts - self.start_ts) / 86400.0


# ── cache helpers ─────────────────────────────────────────────────────────


def _read(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(obj, fh)


# ── fetchers ──────────────────────────────────────────────────────────────


async def _reward_rates(cache_root: Path, client: httpx.AsyncClient) -> dict[str, float]:
    path = cache_root / "reward_rates.json"
    cached = _read(path)
    if cached and time.time() - float(cached.get("ts", 0)) < 86400:
        return {str(k): float(v) for k, v in cached["rates"].items()}
    rates = await fetch_reward_rates(CLOB)
    _write(path, {"ts": time.time(), "rates": rates})
    return rates


async def fetch_meta(slug: str, cache_root: Path, client: httpx.AsyncClient) -> MarketMeta | None:
    rows: list[dict[str, Any]] = []
    # Gamma hides closed markets unless asked; the on-chain tape is mostly closed ones
    for params in ({"slug": slug}, {"slug": slug, "closed": "true"}):
        r = await client.get(f"{GAMMA}/markets", params=params)
        r.raise_for_status()
        rows = r.json()
        if rows:
            break
    if not rows:
        return None
    raw = rows[0]
    rates = await _reward_rates(cache_root, client)
    # parse_market refuses markets that stopped accepting orders; for a backtest
    # the history is still valid, so relax that single gate
    raw = dict(raw)
    raw.setdefault("acceptingOrders", True)
    if not raw.get("acceptingOrders"):
        raw["acceptingOrders"] = True
    return parse_market(raw, rates)


async def fetch_trades(
    cid: str, start_ts: float, cdir: Path, client: httpx.AsyncClient
) -> list[dict[str, Any]]:
    path = cdir / "trades.json"
    cached = _read(path)
    if cached and float(cached.get("start_ts", 1e18)) <= start_ts and \
            time.time() - float(cached.get("fetched_ts", 0)) < 6 * 3600:
        rows: list[dict[str, Any]] = cached["rows"]
        return [r for r in rows if float(r["timestamp"]) >= start_ts]
    rows = []
    offset = 0
    oldest = time.time()
    while offset <= _TRADES_MAX_OFFSET and oldest >= start_ts:
        r = await client.get(
            f"{DATA_API}/trades",
            params={"market": cid, "limit": _TRADES_PAGE, "offset": offset},
        )
        if r.status_code == 400:
            break
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        rows.extend(page)
        oldest = float(page[-1]["timestamp"])
        offset += _TRADES_PAGE
        if len(page) < _TRADES_PAGE:
            break
    reached = min(oldest, start_ts) if oldest < start_ts else oldest
    _write(path, {"fetched_ts": time.time(), "start_ts": reached, "rows": rows})
    if oldest > start_ts:
        log.warning("tape_truncated_by_api", cid=cid[:10], oldest_reached=int(oldest),
                    note="data-api caps history at offset 10000")
    return [r for r in rows if float(r["timestamp"]) >= start_ts]


async def fetch_prices(
    token_id: str, start_ts: float, end_ts: float, cdir: Path, client: httpx.AsyncClient,
    fidelity: int,
) -> list[tuple[float, float]]:
    path = cdir / f"prices_f{fidelity}.json"
    cached = _read(path)
    pts: dict[int, float] = {}
    if cached:
        pts = {int(k): float(v) for k, v in cached.items()}
    have_lo = min(pts) if pts else None
    have_hi = max(pts) if pts else None
    need: list[tuple[float, float]] = []
    if have_lo is None or have_hi is None:
        need.append((start_ts, end_ts))
    else:
        if start_ts < have_lo - 3600:
            need.append((start_ts, have_lo))
        if end_ts > have_hi + 3600:
            need.append((have_hi, end_ts))
    for lo, hi in need:
        t0 = lo
        while t0 < hi:
            t1 = min(hi, t0 + _PRICE_WINDOW_S)
            r = await client.get(
                f"{CLOB}/prices-history",
                params={"market": token_id, "startTs": int(t0), "endTs": int(t1),
                        "fidelity": fidelity},
            )
            if r.status_code == 200:
                for h in r.json().get("history", []):
                    pts[int(h["t"])] = float(h["p"])
            else:
                log.warning("prices_window_failed", status=r.status_code, body=r.text[:80])
            t0 = t1
    if need:
        _write(path, {str(k): v for k, v in sorted(pts.items())})
    return [(float(t), p) for t, p in sorted(pts.items()) if start_ts <= t <= end_ts]


# ── normalization ─────────────────────────────────────────────────────────


def normalize_tape(rows: list[dict[str, Any]], meta: MarketMeta) -> list[Tape]:
    yes, no = meta.yes.token_id, meta.no.token_id
    out: list[Tape] = []
    for r in rows:
        try:
            asset = str(r["asset"])
            side = Side.SELL if str(r.get("side", "")).upper() == "SELL" else Side.BUY
            price = float(r["price"])
            size = float(r["size"])
            ts = float(r["timestamp"])
        except (KeyError, ValueError, TypeError):
            continue
        if size <= 0 or not (0 < price < 1):
            continue
        if asset == yes:
            out.append(Tape(ts, side, price, size, asset, side, price))
        elif asset == no:
            out.append(Tape(ts, side.opposite, round(1.0 - price, 6), size, asset, side, price))
    out.sort(key=lambda t: t.ts)
    return out


def polydata_rows(meta: MarketMeta, src: str | Path, cache_root: Path) -> list[dict[str, Any]]:
    """Every taker print for `meta` from a poly_data orderFilled.csv (cached, incremental)."""
    from polymaker.backtest.polydata import extract_taker_prints

    return extract_taker_prints(Path(src), [meta], cache_root).get(meta.condition_id, [])


async def prefetch_polydata(src: str | Path, slugs: list[str], cache_root: str | Path) -> None:
    """Warm the per-market caches for every slug in ONE pass over the (large) CSV."""
    from polymaker.backtest.polydata import extract_taker_prints

    root = Path(cache_root)
    metas: list[MarketMeta] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for slug in slugs:
            meta = await fetch_meta(slug, root, client)
            if meta is not None:
                metas.append(meta)
    extract_taker_prints(Path(src), metas, root)


async def load_market_history(
    slug: str, *, days: int, cache_root: str | Path = "data/backtest",
    end_ts: float | None = None, polydata: str | Path | None = None,
) -> MarketHistory | None:
    root = Path(cache_root)
    end = end_ts or time.time()
    fidelity = 1 if days <= 45 else 5
    source = DEFAULT_TAPE_SOURCE
    async with httpx.AsyncClient(timeout=60.0) as client:
        meta = await fetch_meta(slug, root, client)
        if meta is None:
            log.warning("backtest_market_not_found", slug=slug)
            return None
        cdir = root / meta.condition_id[:12]
        if polydata is not None:
            source = f"poly_data on-chain fills ({Path(polydata).name})"
            all_rows = polydata_rows(meta, polydata, root)
            if all_rows:
                last = max(float(r["timestamp"]) for r in all_rows)
                if last < end - 3600:  # backfill still running / collector behind
                    log.warning("polydata_tape_behind", cid=meta.condition_id[:10],
                                tape_end=int(last), hours_behind=round((end - last) / 3600, 1),
                                note="window end clamped to the tape")
                    end = last
            else:
                log.warning("polydata_no_prints", cid=meta.condition_id[:10], slug=slug[:40])
            start = end - days * 86400
            rows = [r for r in all_rows if start <= float(r["timestamp"]) <= end]
        else:
            start = end - days * 86400
            rows = await fetch_trades(meta.condition_id, start, cdir, client)
        pts = await fetch_prices(meta.yes.token_id, start, end, cdir, client, fidelity)
    tape = normalize_tape(rows, meta)
    bars = [Bar(t, p) for t, p in pts if 0 < p < 1]
    if tape:
        start = max(start, min(tape[0].ts, bars[0].ts) if bars else tape[0].ts)
    log.info("history_loaded", slug=slug[:40], trades=len(tape), bars=len(bars),
             days=round((end - start) / 86400, 1), source=source)
    return MarketHistory(meta=meta, tape=tape, bars=bars, start_ts=start, end_ts=end,
                         tape_source=source)
