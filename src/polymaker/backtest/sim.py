"""Tape-replay simulator over the pure strategy core.

Replays a market's taker trade tape + minute mid history through the SAME pure
pieces the live engine uses: compute_fair_value, the estimators, RegimeMachine,
construct_quotes, reconcile, and RiskManager over an in-memory StateStore. The
engine's I/O is replaced by:

  * a synthetic book: best bid/ask = mid -/+ half the assumed spread, constant
    touch depth (no historical L2 exists; see the report's assumptions page);
  * a volume-ahead queue model: an order joining a level starts behind
    `touch_depth` shares; prints at that level drain the queue before we fill,
    prints THROUGH the level fill us fully;
  * exact Polymarket reward scoring per sampled minute (Q = S*((v-s)/v)^2, Qmin
    with c=3) so we know when a quote actually scores.

`engine_bugs=True` replicates three live-engine defects found in review (markout
recorded in token space vs YES fair value, unflipped flow sign on NO-token
prints, exit urgency never ramped) so their cost can be measured A/B. The profile passed in carries the defensive knobs (skew floor,
bounded trade-out); the CLI runs a third scenario with them enabled.
"""

from __future__ import annotations

import bisect
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from polymaker.backtest.data import Bar, MarketHistory, Tape
from polymaker.config import RiskConfig, StrategyProfile
from polymaker.domain import Fill, OpenOrder, OrderState, Regime, Side
from polymaker.execution.reconciler import reconcile
from polymaker.marketdata.orderbook import BookView
from polymaker.risk.manager import RiskManager
from polymaker.state.store import StateStore
from polymaker.strategy.estimators import (
    FlowEstimator,
    MarketEstimators,
    MarkoutTracker,
    VolEstimator,
)
from polymaker.strategy.quoting import (
    QuoteInputs,
    compute_fair_value,
    construct_quotes,
    inventory_util,
    round_to_tick,
)
from polymaker.strategy.regime import RegimeInputs, RegimeMachine

_EPS = 1e-9


@dataclass(slots=True)
class SimParams:
    spread_ticks: int = 2  # assumed book spread (ticks) around the minute mid
    touch_depth: float = 300.0  # shares ahead of us when we join a level
    sweep_depth: float = 500.0  # assumed near-touch depth for the sweep detector
    engine_bugs: bool = False  # replicate the live engine's known defects
    markout_horizons: tuple[int, ...] = (60, 300, 1800)
    reward_shares: tuple[float, ...] = (0.05, 0.10, 0.20)
    # USDC collateral on the funder; BUYs it cannot fund are not placed (what the
    # engine does live). None = unconstrained (the pre-existing behaviour).
    bankroll: float | None = None


@dataclass(slots=True)
class SimOrder:
    oid: str
    token_id: str
    side: Side
    price: float  # token-space price (what we'd post)
    size: float
    placed_ts: float
    queue_ahead: float
    yes_price: float  # YES-space level
    is_bid: bool  # True = sits on the YES bid side (BUY YES or SELL NO)


@dataclass(slots=True)
class SimFill:
    ts: float
    token_id: str
    side: Side
    price: float
    size: float
    yes_price: float
    bought_yes: bool  # YES-equivalent direction (BUY YES / SELL NO = True)
    fv: float
    regime: str
    taker: bool = False  # a bounded trade-out (we crossed the spread)
    fee_usd: float = 0.0
    rebate_usd: float = 0.0  # maker rebate credited on this fill (fee-enabled markets)
    markout_usd: dict[int, float] = field(default_factory=dict)  # horizon -> $
    markout_ticks: dict[int, float] = field(default_factory=dict)


@dataclass(slots=True)
class SimResult:
    slug: str
    question: str
    profile_name: str
    scenario: str
    params: SimParams
    start_ts: float
    end_ts: float
    n_trades: int
    n_bars: int
    fills: list[SimFill]
    equity: list[tuple[float, float]]
    price: list[tuple[float, float]]
    inventory: list[tuple[float, float, float]]  # ts, pos_yes, pos_no
    regime_seconds: dict[str, float]
    placements: int
    cancels: int
    order_lifetimes_s: list[float]
    reward_minutes_total: int
    reward_minutes_scored: int
    reward_minutes_one_sided: int
    reward_unscored: dict[str, int]
    reward_unscored_samples: list[tuple[float, list[tuple[bool, float, float]]]]
    reward_qmin_sum: float
    reward_daily_rate: float
    final_cash: float
    final_inv_value: float
    final_pos_yes: float
    final_pos_no: float
    max_drawdown: float
    max_net_shares: float
    max_inv_value: float
    daily_pnl: list[tuple[str, float]]
    halt_minutes: int
    tox_series: list[tuple[float, float]]

    # ── derived ───────────────────────────────────────────────────────────
    @property
    def days(self) -> float:
        return max(1e-9, (self.end_ts - self.start_ts) / 86400.0)

    @property
    def total_pnl(self) -> float:
        return self.final_cash + self.final_inv_value

    @property
    def fill_notional(self) -> float:
        return sum(f.price * f.size for f in self.fills)

    @property
    def fill_shares(self) -> float:
        return sum(f.size for f in self.fills)

    @property
    def trade_outs(self) -> list[SimFill]:
        return [f for f in self.fills if f.taker]

    @property
    def fees_usd(self) -> float:
        return sum(f.fee_usd for f in self.fills)

    @property
    def rebates_usd(self) -> float:
        """Maker rebates credited on our maker fills (already inside total_pnl)."""
        return sum(f.rebate_usd for f in self.fills)

    def markout_mean_usd(self, h: int) -> float:
        xs = [f.markout_usd[h] for f in self.fills if h in f.markout_usd]
        return sum(xs) / len(xs) if xs else 0.0

    def markout_total_usd(self, h: int) -> float:
        return sum(f.markout_usd[h] for f in self.fills if h in f.markout_usd)

    def markout_mean_ticks(self, h: int) -> float:
        """Size-weighted mean markout in ticks (total $ / shares / tick), so one
        large adverse fill is not hidden behind many small benign ones."""
        pairs = [(f.markout_ticks[h], f.size) for f in self.fills if h in f.markout_ticks]
        w = sum(s for _, s in pairs)
        return sum(t * s for t, s in pairs) / w if w > 0 else 0.0

    def adverse_fraction(self, h: int) -> float:
        xs = [f.markout_usd[h] for f in self.fills if h in f.markout_usd]
        return sum(1 for x in xs if x < 0) / len(xs) if xs else 0.0

    @property
    def scored_fraction(self) -> float:
        return self.reward_minutes_scored / self.reward_minutes_total if self.reward_minutes_total else 0.0

    def rewards_usd(self, share: float) -> float:
        """Estimated liquidity rewards if we capture `share` of the daily pool
        while scoring. Share is an assumption (competition is not observable)."""
        return self.reward_daily_rate * self.days * self.scored_fraction * share

    @property
    def regime_share(self) -> dict[str, float]:
        tot = sum(self.regime_seconds.values()) or 1.0
        return {k: v / tot for k, v in self.regime_seconds.items()}

    @property
    def avg_order_lifetime_s(self) -> float:
        return sum(self.order_lifetimes_s) / len(self.order_lifetimes_s) if self.order_lifetimes_s else 0.0

    @property
    def sharpe_daily(self) -> float:
        xs = [p for _, p in self.daily_pnl]
        if len(xs) < 3:
            return 0.0
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return m / math.sqrt(v) * math.sqrt(365) if v > 0 else 0.0


# ── the simulator ─────────────────────────────────────────────────────────


def run_sim(
    hist: MarketHistory, profile: StrategyProfile, profile_name: str,
    risk_cfg: RiskConfig, params: SimParams, scenario: str,
) -> SimResult:
    meta = hist.meta
    tick = meta.tick_size
    dec = meta.price_decimals
    yes, no = meta.yes.token_id, meta.no.token_id
    half_spread = params.spread_ticks * tick / 2.0

    store = StateStore(":memory:")
    risk = RiskManager(risk_cfg, store)
    est = MarketEstimators(
        vol=VolEstimator(profile.vol_short_halflife_s, profile.vol_long_halflife_s),
        flow=FlowEstimator(profile.flow_ewma_halflife_s),
        # pre-fix engine: toxicity never aged out between fills
        markout=MarkoutTracker(
            decay_halflife_s=profile.tox_decay_halflife_s if profile.tox_decay_halflife_s > 0 else None),
    )
    fee_rate = meta.taker_fee_bps / 10000.0 if meta.fees_enabled else 0.0
    last_trade_out: dict[str, float] = {}
    rm = RegimeMachine()

    orders: dict[str, SimOrder] = {}
    next_oid = 0
    cash = 0.0
    fills: list[SimFill] = []
    equity: list[tuple[float, float]] = []
    price: list[tuple[float, float]] = []
    inventory: list[tuple[float, float, float]] = []
    tox_series: list[tuple[float, float]] = []
    regime_seconds: dict[str, float] = defaultdict(float)
    placements = cancels = 0
    lifetimes: list[float] = []
    rw_total = rw_scored = rw_one_sided = 0
    rw_qsum = 0.0
    rw_unscored: dict[str, int] = defaultdict(int)
    rw_samples: list[tuple[float, list[tuple[bool, float, float]]]] = []
    halt_minutes = 0
    pos_open_ts: dict[str, float] = {}
    sweep = False
    mid: float | None = None
    last_regime: Regime | None = None
    last_ts: float | None = None
    peak = 0.0
    max_dd = 0.0
    max_net = 0.0
    max_inv = 0.0
    day_key: str | None = None
    day_start_eq = 0.0
    daily: list[tuple[str, float]] = []

    # merged event stream: bars (kind 0) before trades (kind 1) at equal ts
    events: list[tuple[float, int, int]] = [(b.ts, 0, i) for i, b in enumerate(hist.bars)]
    events += [(t.ts, 1, i) for i, t in enumerate(hist.tape)]
    events.sort()
    bar_ts = [b.ts for b in hist.bars]

    def mark_equity(m: float) -> float:
        py, pn = store.position(yes).size, store.position(no).size
        return cash + py * m + pn * (1.0 - m)

    def apply(fill_token: str, side: Side, price_tok: float, size: float, ts: float,
              yes_px: float, fv: float, regime: Regime, *, taker: bool = False) -> None:
        nonlocal cash
        prior = store.position(fill_token).size
        fill = Fill(fill_token, side, price_tok, size, f"sim-{ts:.3f}-{len(fills)}", ts,
                    is_maker=not taker)
        if not store.apply_fill(fill):
            return
        risk.note_fill(fill)
        cash += price_tok * size * (1 if side is Side.SELL else -1)
        # V2 taker fee per share = rate * p * (1 - p); makers pay nothing. On a
        # fee-enabled market the maker is rebated `rebate_rate` of the fee the
        # taker paid on that fill (0 on fee-free markets such as politics).
        per_share = fee_rate * price_tok * (1.0 - price_tok)
        fee = per_share * size if taker else 0.0
        rebate = per_share * meta.rebate_rate * size if not taker else 0.0
        cash -= fee
        cash += rebate
        if side is Side.BUY and prior <= 0:
            pos_open_ts[fill_token] = ts
        if store.position(fill_token).size <= 0:
            pos_open_ts.pop(fill_token, None)
        # markout bookkeeping, exactly as the engine does it (bug toggle)
        if params.engine_bugs:
            token_fv = fv if fill_token == yes else 1.0 - fv
            est.markout.record_fill(side, token_fv, ts)
        else:
            side_yes = side if fill_token == yes else side.opposite
            est.markout.record_fill(side_yes, fv, ts)
        bought_yes = (fill_token == yes) == (side is Side.BUY)
        fills.append(SimFill(ts, fill_token, side, price_tok, size, yes_px, bought_yes, fv,
                             regime.value, taker=taker, fee_usd=fee, rebate_usd=rebate))

    def recompute(now: float) -> None:
        nonlocal next_oid, placements, cancels, sweep, last_regime, last_ts
        assert mid is not None
        bb = round_to_tick(mid - half_spread, tick, dec, up=False)
        ba = round_to_tick(mid + half_spread, tick, dec, up=True)
        if ba <= bb:
            ba = round(bb + tick, dec)
        d = params.touch_depth
        yes_view = BookView(bb, d, ba, d, round(bb - tick, dec), round(ba + tick, dec), d, d)
        no_view = BookView(round(1 - ba, dec), d, round(1 - bb, dec), d,
                           round(1 - ba - tick, dec), round(1 - bb + tick, dec), d, d)

        est.flow.decay_to(now)
        fv = compute_fair_value(mid, est.flow.z, tick)
        prev_fv = est.last_fv
        est.on_fair_value(fv, now)
        risk.update_mark(yes, fv)
        risk.update_mark(no, 1.0 - fv)

        pos_yes, pos_no = store.position(yes), store.position(no)
        inv_util = abs(inventory_util(pos_yes.size - pos_no.size, fv, profile.q_max_usdc,
                                      profile.inventory_metric))
        rd = risk.evaluate(meta, ws_stale=False, event_group_cost=0.0)
        regime = rm.decide(RegimeInputs(
            now=now, tick=tick, fv=fv, prev_fv=prev_fv, vol_ratio=est.vol.ratio,
            flow_z=est.flow.z, inventory_util=inv_util, hours_to_end=None,
            sweep_flagged=sweep, risk_halt=rd.halt, risk_reduce_only=rd.reduce_only,
        ), profile)
        sweep = False

        if last_regime is not None and last_ts is not None:
            regime_seconds[last_regime.value] += max(0.0, now - last_ts)
        last_regime, last_ts = regime, now

        def urgency(tok: str) -> float:
            if params.engine_bugs or tok not in pos_open_ts or profile.exit_urgency_s <= 0:
                return 0.0
            return min(1.0, max(0.0, (now - pos_open_ts[tok]) / profile.exit_urgency_s))

        allowed = all(now - last_trade_out.get(t, 0.0) >= profile.trade_out_cooldown_s
                      for t in (yes, no))
        tq = construct_quotes(QuoteInputs(
            meta=meta, regime=regime, fv=fv, vol_short=est.vol.short,
            toxicity=est.markout.toxicity, yes_view=yes_view, no_view=no_view,
            pos_yes=pos_yes, pos_no=pos_no, profile=profile, now=now,
            risk_size_scale=rd.size_scale,
            yes_exit_urgency=urgency(yes), no_exit_urgency=urgency(no),
            trade_out_allowed=allowed,
        ))
        if tq.trade_outs:
            # bounded taker exit: fills at the synthetic best bid of that token
            # (the core already capped size to a fraction of the touch), subject
            # to the worst price; then cool down. Resting exits on the token are
            # pulled first, exactly as the engine does.
            for t in tq.trade_outs:
                last_trade_out[t.token_id] = now
                v = yes_view if t.token_id == yes else no_view
                assert v.best_bid is not None
                px = v.best_bid
                if px < t.worst_price - _EPS:
                    continue
                held = store.position(t.token_id).size
                size = min(t.size, held)
                if size < meta.min_order_size:
                    continue
                for o in [o for o in orders.values() if o.token_id == t.token_id]:
                    orders.pop(o.oid, None)
                    lifetimes.append(now - o.placed_ts)
                    cancels += 1
                yes_px = px if t.token_id == yes else round(1.0 - px, dec)
                apply(t.token_id, Side.SELL, px, size, now, yes_px, fv, regime, taker=True)
            return  # requote on the next event with the new position
        live = [OpenOrder(o.oid, o.token_id, o.side, o.price, o.size, OrderState.LIVE)
                for o in orders.values()]
        plan = reconcile(tq, live, tick=tick, reprice_ticks=profile.reprice_ticks,
                         resize_frac=profile.resize_frac,
                         min_live_size=meta.rewards_min_size if profile.replace_below_reward_min else 0.0)
        for oid in plan.to_cancel:
            o = orders.pop(oid, None)
            if o is not None:
                cancels += 1
                lifetimes.append(now - o.placed_ts)
        for q in plan.to_place:
            if params.bankroll is not None and q.side is Side.BUY:
                open_buys = sum(o.price * o.size for o in orders.values() if o.side is Side.BUY)
                if q.price * q.size > params.bankroll + cash - open_buys + _EPS:
                    continue  # the engine never sends an entry it cannot fund
            if q.token_id == yes:
                yes_px, is_bid = q.price, q.side is Side.BUY
            else:
                yes_px, is_bid = round(1.0 - q.price, dec), q.side is Side.SELL
            # queue: joining/behind the touch -> full depth ahead; inside the spread -> front
            if is_bid:
                ahead = d if yes_px <= bb + _EPS else 0.0
            else:
                ahead = d if yes_px >= ba - _EPS else 0.0
            next_oid += 1
            orders[f"o{next_oid}"] = SimOrder(f"o{next_oid}", q.token_id, q.side, q.price,
                                              q.size, now, ahead, yes_px, is_bid)
            placements += 1

    def on_trade(t: Tape) -> None:
        nonlocal sweep
        assert mid is not None
        # flow: the live engine feeds the raw aggressor (bug); fixed flips NO prints
        est.flow.update(t.raw_side if params.engine_bugs else t.aggressor, t.size, t.ts)
        base = profile.base_size_usdc / max(t.raw_price, 0.01)
        if t.size >= profile.event_sweep_mult * base and \
                t.size >= profile.event_sweep_frac * params.sweep_depth:
            sweep = True
        fv = est.last_fv if est.last_fv is not None else mid
        regime = last_regime or Regime.QUIET
        remaining = t.size
        if t.aggressor is Side.SELL:
            book = sorted((o for o in orders.values() if o.is_bid), key=lambda o: -o.yes_price)
            for o in book:
                if remaining <= 0:
                    break
                if t.price < o.yes_price - _EPS:  # traded through our level
                    take = min(o.size, remaining)
                elif abs(t.price - o.yes_price) <= _EPS:  # at our level: drain the queue first
                    ahead = o.queue_ahead
                    o.queue_ahead = max(0.0, ahead - remaining)
                    avail = remaining - ahead
                    take = min(o.size, avail) if avail > 0 else 0.0
                    remaining -= min(ahead, remaining)
                else:
                    break
                if take <= 0:
                    continue
                apply(o.token_id, o.side, o.price, take, t.ts, o.yes_price, fv, regime)
                remaining -= take
                o.size = round(o.size - take, 6)
                if o.size <= _EPS:
                    orders.pop(o.oid, None)
                    lifetimes.append(t.ts - o.placed_ts)
        else:
            book = sorted((o for o in orders.values() if not o.is_bid), key=lambda o: o.yes_price)
            for o in book:
                if remaining <= 0:
                    break
                if t.price > o.yes_price + _EPS:
                    take = min(o.size, remaining)
                elif abs(t.price - o.yes_price) <= _EPS:
                    ahead = o.queue_ahead
                    o.queue_ahead = max(0.0, ahead - remaining)
                    avail = remaining - ahead
                    take = min(o.size, avail) if avail > 0 else 0.0
                    remaining -= min(ahead, remaining)
                else:
                    break
                if take <= 0:
                    continue
                apply(o.token_id, o.side, o.price, take, t.ts, o.yes_price, fv, regime)
                remaining -= take
                o.size = round(o.size - take, 6)
                if o.size <= _EPS:
                    orders.pop(o.oid, None)
                    lifetimes.append(t.ts - o.placed_ts)

    def sample_rewards(m: float) -> None:
        nonlocal rw_total, rw_scored, rw_one_sided, rw_qsum
        v = meta.rewards_max_spread
        if v <= 0:
            return
        qb = qa = 0.0
        n_small = n_out = 0
        for o in orders.values():
            if o.size + _EPS < meta.rewards_min_size:
                n_small += 1
                continue
            s = (m - o.yes_price) * 100.0 if o.is_bid else (o.yes_price - m) * 100.0
            if s < 0 or s > v:
                n_out += 1
                continue
            q = o.size * ((v - s) / v) ** 2
            if o.is_bid:
                qb += q
            else:
                qa += q
        # Official rule (docs.polymarket.com/market-makers/liquidity-rewards): inside the
        # 10c-90c band a one-sided book scores a third, a balanced one scores min(Qone, Qtwo);
        # outside the band only the two-sided minimum counts.
        if 0.10 <= m <= 0.90:
            qmin = max(min(qb, qa), max(qb, qa) / 3.0)
        else:
            qmin = min(qb, qa)
        rw_total += 1
        if qmin > 0:
            rw_scored += 1
            rw_qsum += qmin
            if min(qb, qa) <= 0:
                rw_one_sided += 1
        else:
            if orders and len(rw_samples) < 24 and rw_total % 997 == 0:
                rw_samples.append((m, [(o.is_bid, o.yes_price, o.size) for o in orders.values()]))
            if not orders:
                rw_unscored["no_orders"] += 1
            elif n_out and not n_small:
                rw_unscored["outside_max_spread"] += 1
            elif n_small and not n_out:
                rw_unscored["below_min_size"] += 1
            elif n_small or n_out:
                rw_unscored["mixed"] += 1
            else:
                rw_unscored["one_sided_out_of_band"] += 1

    for ts, kind, idx in events:
        if kind == 0:
            b: Bar = hist.bars[idx]
            mid = b.mid
            eq = mark_equity(mid)
            equity.append((ts, eq))
            py, pn = store.position(yes).size, store.position(no).size
            inventory.append((ts, py, pn))
            price.append((ts, mid))
            tox_series.append((ts, est.markout.toxicity))
            peak = max(peak, eq)
            max_dd = max(max_dd, peak - eq)
            max_net = max(max_net, abs(py - pn))
            max_inv = max(max_inv, py * mid + pn * (1 - mid))
            dk = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
            if dk != day_key:
                if day_key is not None:
                    daily.append((day_key, eq - day_start_eq))
                day_key, day_start_eq = dk, eq
                risk.reset_day()
            sample_rewards(mid)
            if (last_regime is Regime.HALTED):
                halt_minutes += 1
            recompute(ts)
        else:
            if mid is None:
                continue
            on_trade(hist.tape[idx])
            recompute(ts)

    if day_key is not None and equity:
        daily.append((day_key, equity[-1][1] - day_start_eq))

    # post-hoc markouts from the mid series (exact, independent of the estimator)
    for f in fills:
        for h in params.markout_horizons:
            i = bisect.bisect_left(bar_ts, f.ts + h)
            if i >= len(hist.bars):
                continue
            m_h = hist.bars[i].mid
            move = (m_h - f.yes_price) * (1.0 if f.bought_yes else -1.0)
            f.markout_usd[h] = move * f.size
            f.markout_ticks[h] = move / tick

    final_mid = hist.bars[-1].mid if hist.bars else (mid or 0.5)
    py, pn = store.position(yes).size, store.position(no).size
    store.close()
    return SimResult(
        slug=meta.slug, question=meta.question, profile_name=profile_name, scenario=scenario,
        params=params, start_ts=hist.start_ts, end_ts=hist.end_ts,
        n_trades=len(hist.tape), n_bars=len(hist.bars), fills=fills, equity=equity,
        price=price, inventory=inventory, regime_seconds=dict(regime_seconds),
        placements=placements, cancels=cancels, order_lifetimes_s=lifetimes,
        reward_minutes_total=rw_total, reward_minutes_scored=rw_scored, reward_minutes_one_sided=rw_one_sided,
        reward_unscored=dict(rw_unscored), reward_unscored_samples=rw_samples, reward_qmin_sum=rw_qsum,
        reward_daily_rate=meta.rewards_daily_rate, final_cash=cash,
        final_inv_value=py * final_mid + pn * (1 - final_mid), final_pos_yes=py, final_pos_no=pn,
        max_drawdown=max_dd, max_net_shares=max_net, max_inv_value=max_inv, daily_pnl=daily,
        halt_minutes=halt_minutes, tox_series=tox_series,
    )


def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d %H:%M UTC")


def now_ts() -> float:
    return time.time()
