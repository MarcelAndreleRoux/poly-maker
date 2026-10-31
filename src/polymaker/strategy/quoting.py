"""Pure quote construction: (market state, inventory, params) -> TargetQuotes.

This is the deterministic core of the strategy. No I/O, no wall-clock reads
except values passed in. Everything here is exercised directly by unit tests.

Model (see the README):
  reservation  r  = FV - skew(inventory),  skew = max(γ·σ, skew_floor)·u
  half-spread  δ  = base + c_vol·σ + c_tox·toxicity   (clamped to reward band in QUIET)
  YES entry bid   = r - δ                       (BUY YES, USDC-collateralized)
  NO  entry bid   = (1 - r) - δ                  (BUY NO; implied YES ask at r + δ)
  exits           = SELL limits on held inventory at r + δ (they lean with the
                    inventory), walked toward the touch by urgency; the opposing
                    entry is shrunk by the exit size so the flatten order, not
                    the pair-adding order, gets the flow
  trade-out       = bounded TAKER exit when a position is under water and the
                    passive exit has had its chance (see StrategyProfile)
  u               = signed capital-at-risk / q_max  (YES at fv, NO at 1-fv)

The BUY-YES + BUY-NO pair is the canonical two-sided quote: both are bids, both
score rewards, and a filled pair merges back to USDC at locked edge 1 - p - q.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from polymaker.config import StrategyProfile
from polymaker.domain import MarketMeta, Position, Quote, Regime, Side, TargetQuotes, TradeOut
from polymaker.marketdata.orderbook import BookView

_EPS = 1e-9


def round_to_tick(price: float, tick: float, decimals: int, *, up: bool) -> float:
    """Snap a price to the tick grid, rounding up or down, clamped to (0,1)."""
    n = price / tick
    n = math.ceil(n - _EPS) if up else math.floor(n + _EPS)
    p = round(n * tick, decimals)
    return min(max(p, tick), 1.0 - tick)


def compute_fair_value(microprice: float, flow_z: float, tick: float, weight: float = 0.5) -> float:
    """Nudge the microprice by bounded signed flow. Clamped to (tick, 1-tick)."""
    fv = microprice + weight * flow_z * tick
    return min(max(fv, tick), 1.0 - tick)


@dataclass(frozen=True, slots=True)
class QuoteInputs:
    meta: MarketMeta
    regime: Regime
    fv: float  # YES fair value in (0,1)
    vol_short: float
    toxicity: float
    yes_view: BookView
    no_view: BookView
    pos_yes: Position
    pos_no: Position
    profile: StrategyProfile
    now: float
    risk_size_scale: float = 1.0  # RiskManager may throttle size in [0,1]
    yes_exit_urgency: float = 0.0  # [0,1]; engine raises with hold time
    no_exit_urgency: float = 0.0
    trade_out_allowed: bool = True  # engine clears this during the trade-out cooldown


def construct_quotes(inp: QuoteInputs) -> TargetQuotes:
    m = inp.meta
    p = inp.profile
    tick = m.tick_size
    dec = m.price_decimals
    cid = m.condition_id

    if inp.regime is Regime.HALTED:
        return TargetQuotes(cid, inp.regime, ())
    if inp.regime is Regime.EVENT and not p.event_keeps_exits:
        return TargetQuotes(cid, inp.regime, ())
    # EVENT: pull entries, keep working the exits (never hold a bag with no exit up)
    entries_allowed = inp.regime not in (Regime.EVENT, Regime.REDUCE_ONLY)

    quotes: list[Quote] = []

    # ── inventory as signed capital-at-risk vs the cap ───────────────────
    net_shares = inp.pos_yes.size - inp.pos_no.size
    u = inventory_util(net_shares, inp.fv, p.q_max_usdc, p.inventory_metric)
    reward_floor = m.rewards_min_size * p.reward_size_mult  # scoring size w/ margin

    skew = max(p.gamma * inp.vol_short, p.skew_min_ticks * tick) * u

    # ── half-spread ─────────────────────────────────────────────────────
    base = p.delta_min_ticks * tick
    delta = base + p.c_vol * inp.vol_short + p.c_tox * inp.toxicity
    reward_band = m.rewards_max_spread / 100.0
    if inp.regime == Regime.QUIET and reward_band > 0:
        delta = _clamp(delta, base, max(base, reward_band))
    delta = max(delta, tick)

    r = _clamp(inp.fv - skew, tick, 1.0 - tick)
    yes_bid_target = r - delta
    no_bid_target = (1.0 - r) - delta

    # ── exits first: SELL held inventory (maker, never cross), optionally leaned
    exits: list[Quote] = []
    anchor = r if p.exit_lean else inp.fv
    _maybe_exit(exits, m.yes.token_id, inp.pos_yes, anchor, delta, inp.yes_view, tick, dec,
                inp.yes_exit_urgency, m, inp.regime, p.reduce_only_urgency)
    _maybe_exit(exits, m.no.token_id, inp.pos_no, 1.0 - anchor, delta, inp.no_view, tick, dec,
                inp.no_exit_urgency, m, inp.regime, p.reduce_only_urgency)
    # an exit only displaces the opposing entry when it scores on its own; a
    # sub-minimum exit must not remove the side's only reward-eligible order
    yes_exit_size = no_exit_size = 0.0
    if p.entry_yields_to_exit:
        yes_exit_size = sum(q.size for q in exits if q.token_id == m.yes.token_id)
        no_exit_size = sum(q.size for q in exits if q.token_id == m.no.token_id)
        if yes_exit_size < m.rewards_min_size:
            yes_exit_size = 0.0
        if no_exit_size < m.rewards_min_size:
            no_exit_size = 0.0

    # ── size scaling ────────────────────────────────────────────────────
    regime_scale = 0.5 if inp.regime == Regime.TRENDING else 1.0
    tox_scale = 1.0 / (1.0 + inp.toxicity * 10.0)
    common_scale = regime_scale * tox_scale * _clamp(inp.risk_size_scale, 0.0, 1.0)

    soft_cap = p.q_soft_frac  # fraction of q_max at which the adding side pulls
    add_yes = entries_allowed and u < soft_cap
    add_no = entries_allowed and u > -soft_cap

    # entry: BUY YES. A SELL NO exit sits on the same side of the book at the same
    # level, so shrink this entry by it: the flatten order gets the flow.
    if add_yes:
        price = _place_bid(yes_bid_target, inp.yes_view, tick, dec, inp.fv, p.min_edge_ticks)
        if price is not None:
            size = _size_shares(p, price, common_scale * (1 - max(u, 0.0)), m)
            _add_layers(quotes, m.yes.token_id, Side.BUY, price, tick, dec,
                        max(0.0, size - no_exit_size),
                        p.layers, p.layer_step_ticks, down=True,
                        exchange_min=m.min_order_size, reward_floor=reward_floor)

    # entry: BUY NO (shrunk by the SELL YES exit, same reasoning)
    if add_no:
        no_fv = 1.0 - inp.fv
        price = _place_bid(no_bid_target, inp.no_view, tick, dec, no_fv, p.min_edge_ticks)
        if price is not None:
            size = _size_shares(p, price, common_scale * (1 - max(-u, 0.0)), m)
            _add_layers(quotes, m.no.token_id, Side.BUY, price, tick, dec,
                        max(0.0, size - yes_exit_size),
                        p.layers, p.layer_step_ticks, down=True,
                        exchange_min=m.min_order_size, reward_floor=reward_floor)

    quotes.extend(exits)

    # ── bounded taker exit (last resort) ────────────────────────────────
    trade_outs: list[TradeOut] = []
    if inp.trade_out_allowed:
        _maybe_trade_out(trade_outs, m.yes.token_id, inp.pos_yes, inp.fv, inp.yes_view, tick, dec,
                         inp.yes_exit_urgency, inp.regime, m, p)
        _maybe_trade_out(trade_outs, m.no.token_id, inp.pos_no, 1.0 - inp.fv, inp.no_view, tick,
                         dec, inp.no_exit_urgency, inp.regime, m, p)

    return TargetQuotes(cid, inp.regime, tuple(quotes), tuple(trade_outs))


# ── helpers ─────────────────────────────────────────────────────────────


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def inventory_util(
    net_shares: float, fv: float, q_max_usdc: float, metric: str = "at_risk"
) -> float:
    """Signed inventory utilisation in [-1, 1].

    "at_risk": capital at risk / cap. Long YES loses `fv` per share if NO
    resolves; long NO (net < 0) loses `1-fv` per share if YES resolves.
    "yes_notional": the pre-fix measure (|net| * fv for both legs), which let a
    NO bag on a 0.19 market reach ~4x the intended dollar cap.
    """
    if q_max_usdc <= 0:
        return 0.0
    if metric == "yes_notional" or net_shares > 0:
        at_risk = net_shares * max(fv, 1e-9)
    else:
        at_risk = net_shares * (1.0 - fv)
    return _clamp(at_risk / q_max_usdc, -1.0, 1.0)


def _place_bid(
    target: float, view: BookView, tick: float, dec: int, fv: float, min_edge_ticks: int
) -> float | None:
    """Position a BUY: join the touch or sit behind, never cross, keep min edge vs FV."""
    price = target
    # never bid above (FV - min_edge*tick): we don't pay through fair value
    price = min(price, fv - min_edge_ticks * tick)
    # join the queue rather than jump it (conservative maker default)
    if view.best_bid is not None and price >= view.best_bid:
        price = view.best_bid
    # never cross the ask
    if view.best_ask is not None and price >= view.best_ask:
        price = view.best_ask - tick
    p = round_to_tick(price, tick, dec, up=False)
    if p <= 0 or p >= 1:
        return None
    return p


def _size_shares(p: StrategyProfile, price: float, scale: float, m: MarketMeta) -> float:
    """Entry size in shares: `base_size_shares` when set, else the USDC notional
    `base_size_usdc` converted at the order price. Per-order minimums are applied
    in _add_layers (reward scoring is per ORDER, so the floor holds per layer)."""
    if p.base_size_shares > 0:
        shares = p.base_size_shares * max(scale, 0.0)
    else:
        shares = (p.base_size_usdc / max(price, m.tick_size)) * max(scale, 0.0)
    return round(shares, 2) if shares > 0 else 0.0


def _add_layers(
    quotes: list[Quote], token_id: str, side: Side, top_price: float, tick: float, dec: int,
    total_size: float, layers: int, step_ticks: int, *, down: bool,
    exchange_min: float = 0.0, reward_floor: float = 0.0,
) -> None:
    """Split size across `layers` price levels stepping away from the touch.

    Each ORDER must meet the exchange min and, when within reach (>=50% of it),
    is bumped to `reward_floor` (the reward min-size × the profile margin) so it
    actually scores — the program scores per order, so a floor applied to the
    total is worthless. Layers that can't reach the floor are consolidated into
    fewer, larger orders rather than resting unscoring dust.
    """
    if total_size <= 0:
        return
    layers = max(1, layers)
    reward_floor = max(reward_floor, exchange_min)
    per = round(total_size / layers, 2)
    # consolidate: if a split layer would fall below half the reward floor,
    # use fewer layers so each resting order can still score
    while layers > 1 and reward_floor > 0 and per < 0.5 * reward_floor:
        layers -= 1
        per = round(total_size / layers, 2)
    if reward_floor > 0 and 0.5 * reward_floor <= per < reward_floor:
        per = reward_floor  # bump each order up to scoring size
    if per < exchange_min or per <= 0:
        return
    for i in range(layers):
        offset = i * step_ticks * tick
        price = top_price - offset if down else top_price + offset
        price = round(price, dec)
        if 0 < price < 1:
            quotes.append(Quote(token_id, side, price, per))


def _maybe_exit(
    quotes: list[Quote], token_id: str, pos: Position, token_fv: float, delta: float,
    view: BookView, tick: float, dec: int, urgency: float, m: MarketMeta, regime: Regime,
    reduce_only_urgency: float = 0.5,
) -> None:
    if pos.size < m.min_order_size:
        return
    # target starts at fv + delta and walks toward best_bid + tick as urgency -> 1
    passive = token_fv + delta
    floor = (view.best_bid + tick) if view.best_bid is not None else passive
    if regime == Regime.REDUCE_ONLY:
        urgency = max(urgency, reduce_only_urgency)
    target = passive * (1.0 - urgency) + floor * urgency
    # never cross down through the bid; never sell below best_bid
    if view.best_bid is not None:
        target = max(target, view.best_bid + tick)
    price = round_to_tick(target, tick, dec, up=True)
    # FLOOR (never round up): selling more than we hold is rejected by the
    # exchange -> the exit silently fails and we stay long. Floor guarantees
    # size <= held.
    size = math.floor(pos.size * 100) / 100
    if 0 < price < 1 and size >= m.min_order_size:
        quotes.append(Quote(token_id, Side.SELL, price, size))


def _maybe_trade_out(
    out: list[TradeOut], token_id: str, pos: Position, token_fv: float, view: BookView,
    tick: float, dec: int, urgency: float, regime: Regime, m: MarketMeta, p: StrategyProfile,
) -> None:
    """Bounded taker exit. Conditions (all must hold):
      * enabled, a real position with a known cost basis, a bid to sell into;
      * under water by >= trade_out_drift_ticks vs. avg cost (fair value, not touch);
      * the passive exit had its chance: REDUCE_ONLY, urgency >= 1, or drift >= 2x.
    Size is capped to a fraction of the best-bid size so we never sell through
    an air pocket; the worst price caps slippage. HALTED never trades."""
    if not p.trade_out_enabled or regime is Regime.HALTED:
        return
    if pos.size < m.min_order_size or pos.avg_price <= 0 or view.best_bid is None:
        return
    adverse_ticks = (pos.avg_price - token_fv) / tick
    if adverse_ticks < p.trade_out_drift_ticks:
        return
    forced = regime is Regime.REDUCE_ONLY or urgency >= 1.0 - _EPS
    if not forced and adverse_ticks < 2.0 * p.trade_out_drift_ticks:
        return
    size = min(pos.size, p.trade_out_depth_frac * view.best_bid_size)
    size = math.floor(size * 100) / 100
    if size < m.min_order_size:
        return
    worst = round_to_tick(view.best_bid - p.trade_out_max_slip_ticks * tick, tick, dec, up=False)
    reason = "reduce_only" if regime is Regime.REDUCE_ONLY else ("urgent" if urgency >= 1.0 - _EPS else "drift")
    out.append(TradeOut(token_id, size, worst, reason))


def fit_entries_to_collateral(quotes: list[Quote], available: float) -> tuple[list[Quote], list[Quote]]:
    """Split `quotes` into (fundable, unfunded). A BUY locks price x size of
    collateral until it fills or cancels; SELLs are exits of shares we hold and
    lock nothing. The exchange rejects a BUY it cannot fund, and one rejected
    order quarantines the whole market, so an entry that does not fit is simply
    not sent (it is retried on the next requote). Plan order is kept."""
    kept: list[Quote] = []
    dropped: list[Quote] = []
    for q in quotes:
        if q.side is Side.BUY:
            need = q.price * q.size
            if need > available + 1e-9:
                dropped.append(q)
                continue
            available -= need
        kept.append(q)
    return kept, dropped
