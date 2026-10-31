"""Competition-aware reward ranking: what would *our* resting size earn, per market,
against the competitors already resting in the reward band right now?

Polymarket's liquidity-rewards formula (docs: "Liquidity Rewards"):

    per order   S(v, s) = ((v - s) / v)^2 * size      (s = distance from midpoint,
                                                       v = rewards_max_spread,
                                                       size >= rewards_min_size)
    per maker   Q_min   = max(min(Q_bid, Q_ask), max(Q_bid, Q_ask) / 3)   if 0.10 <= mid <= 0.90
                        = min(Q_bid, Q_ask)                                 otherwise
    payout      pool * our Q_min / sum of every maker's Q_min, sampled once a minute.

The scanner's `reward_density` ranks by pool / Gamma liquidity, which says nothing
about where that liquidity sits. This module reads the live book, scores the
competitors with the real formula, and asks: with N shares per side at a given
distance, what share of the pool is that, and does it clear the payout floor?

Estimate, not forecast: the book aggregates all competitors into one maker (their
true sum of per-maker Q_min is somewhat higher, so our share is slightly
overstated), the book at scan time is one sample of the day's 1440, and a market
we already quote includes our own resting orders among the "competitors".
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from polymaker.domain import MarketMeta
from polymaker.logging import get_logger

log = get_logger("catalog.competition")

Level = tuple[float, float]  # (price, size in shares)


def order_points(dist: float, v: float, size: float, min_size: float) -> float:
    """Reward points of one resting order `dist` (price units) from the midpoint."""
    if v <= 0 or size < min_size or dist < 0 or dist >= v:
        return 0.0
    return ((v - dist) / v) ** 2 * size


def book_points(levels: list[Level], mid: float, v: float, min_size: float) -> float:
    """Points of one side of the book. Each level is treated as one order; the
    per-order size floor is applied per level (a 60-share level made of two
    30-share orders would really score 0, so this errs toward more competition)."""
    return sum(order_points(abs(mid - p), v, s, min_size) for p, s in levels)


def band_shares(levels: list[Level], mid: float, v: float) -> float:
    """Shares resting strictly inside the reward band on one side."""
    return sum(s for p, s in levels if abs(mid - p) < v)


def qmin(q_bid: float, q_ask: float, mid: float) -> float:
    """Polymarket's two-sidedness rule on a maker's (bid points, ask points)."""
    lo, hi = min(q_bid, q_ask), max(q_bid, q_ask)
    if 0.10 <= mid <= 0.90:
        return max(lo, hi / 3.0)
    return lo


@dataclass(frozen=True, slots=True)
class CompetitionSnapshot:
    condition_id: str
    slug: str
    question: str
    pool_per_day: float
    rewards_min_size: float
    rewards_max_spread_c: float
    mid: float
    spread: float
    comp_q_bid: float
    comp_q_ask: float
    comp_qmin: float
    band_bid_shares: float
    band_ask_shares: float
    our_shares: float
    our_dist: float  # price units; max(requested distance, half the spread)
    our_points: float  # per side (symmetric placement)
    our_share: float  # fraction of the pool at this instant
    usd_per_day: float
    floor_ok: bool  # usd_per_day >= payout floor
    volume_24h: float  # trailing 24h volume (USD) from the scan
    band_turnover: float  # volume_24h / USD value resting in the band: how many times a day the band gets eaten
    # pool / competitor points: the "cheap pool" metric independent of our size
    pool_per_comp_point: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess(
    meta: MarketMeta,
    bids: list[Level],
    asks: list[Level],
    our_shares: float,
    our_dist_c: float,
    floor: float = 1.0,
) -> CompetitionSnapshot | None:
    """Score one market from its YES book with our orders `our_dist_c` cents from the
    midpoint (never inside the spread). None when the book is one-sided/empty."""
    if not bids or not asks:
        return None
    best_bid = max(p for p, _ in bids)
    best_ask = min(p for p, _ in asks)
    if best_ask <= best_bid:
        return None
    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid
    v = meta.rewards_max_spread / 100.0
    mn = meta.rewards_min_size
    qb = book_points(bids, mid, v, mn)
    qa = book_points(asks, mid, v, mn)
    comp = qmin(qb, qa, mid)
    dist = max(our_dist_c / 100.0, spread / 2.0)
    ours_side = order_points(dist, v, our_shares, mn)
    ours = qmin(ours_side, ours_side, mid)
    total = ours + comp
    share = ours / total if total > 0 else 0.0
    usd = meta.rewards_daily_rate * share
    bb, ba = band_shares(bids, mid, v), band_shares(asks, mid, v)
    band_usd = bb * mid + ba * (1.0 - mid)
    return CompetitionSnapshot(
        condition_id=meta.condition_id,
        slug=meta.slug,
        question=meta.question,
        pool_per_day=meta.rewards_daily_rate,
        rewards_min_size=mn,
        rewards_max_spread_c=meta.rewards_max_spread,
        mid=round(mid, 4),
        spread=round(spread, 4),
        comp_q_bid=round(qb, 1),
        comp_q_ask=round(qa, 1),
        comp_qmin=round(comp, 1),
        band_bid_shares=round(bb, 0),
        band_ask_shares=round(ba, 0),
        our_shares=our_shares,
        our_dist=round(dist, 4),
        our_points=round(ours_side, 2),
        our_share=round(share, 5),
        usd_per_day=round(usd, 3),
        floor_ok=usd >= floor,
        volume_24h=round(meta.volume_24hr, 0),
        band_turnover=round(meta.volume_24hr / band_usd, 2) if band_usd > 0 else float("inf"),
        pool_per_comp_point=round(meta.rewards_daily_rate / comp, 4) if comp > 0 else float("inf"),
    )


def parse_book(raw: dict[str, Any]) -> tuple[list[Level], list[Level]]:
    bids = [(float(x["price"]), float(x["size"])) for x in raw.get("bids") or []]
    asks = [(float(x["price"]), float(x["size"])) for x in raw.get("asks") or []]
    return bids, asks


async def fetch_book(client: httpx.AsyncClient, token_id: str) -> tuple[list[Level], list[Level]]:
    r = await client.get("/book", params={"token_id": token_id})
    r.raise_for_status()
    return parse_book(r.json())


async def rank_by_competition(
    metas: list[MarketMeta],
    our_shares: float,
    our_dist_c: float,
    floor: float = 1.0,
    clob_host: str = "https://clob.polymarket.com",
    concurrency: int = 6,
    timeout: float = 20.0,
) -> list[CompetitionSnapshot]:
    """Fetch every market's YES book (bounded concurrency) and rank by usd_per_day."""
    sem = asyncio.Semaphore(concurrency)
    out: list[CompetitionSnapshot] = []

    async with httpx.AsyncClient(base_url=clob_host.rstrip("/"), timeout=timeout) as client:

        async def one(m: MarketMeta) -> None:
            async with sem:
                try:
                    bids, asks = await fetch_book(client, m.yes.token_id)
                except Exception as exc:  # noqa: BLE001 - one bad book must not sink the ranking
                    log.warning("book_fetch_failed", slug=m.slug[:40], err=str(exc)[:80])
                    return
            snap = assess(m, bids, asks, our_shares, our_dist_c, floor)
            if snap is not None:
                out.append(snap)

        await asyncio.gather(*(one(m) for m in metas))

    out.sort(key=lambda s: (s.usd_per_day, s.pool_per_comp_point), reverse=True)
    return out
