"""RiskManager: pre-trade gates and circuit breakers (see the README).

Consulted by the engine before every quote set. Returns a per-market decision
(size scale / reduce-only / halt) and owns the global kill switches. Position
and order data come from the StateStore; fair-value marks are pushed in by the
engine so PnL is always current.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polymaker.config import RiskConfig
from polymaker.domain import Fill, MarketMeta, Side
from polymaker.logging import get_logger
from polymaker.state.store import StateStore

log = get_logger("risk.manager")


@dataclass(frozen=True, slots=True)
class RiskDecision:
    halt: bool  # HALTED regime for this market
    reduce_only: bool  # REDUCE_ONLY regime for this market
    size_scale: float  # multiply quote sizes by this [0,1]
    reason: str = ""


class RiskManager:
    def __init__(self, cfg: RiskConfig, store: StateStore) -> None:
        self._cfg = cfg
        self._store = store
        self._marks: dict[str, float] = {}  # token_id -> fair value
        self._net_cash = 0.0  # cumulative signed cash from fills (+sell, -buy)
        self._day_start_equity = 0.0
        self._killed = False
        self._kill_reason = "manual_kill"
        self._total_loss = ""
        self._tl_strikes = 0
        self._tl_first_ts = 0.0
        self._order_attempts = 0
        self._order_errors = 0

    # ── PnL bookkeeping ─────────────────────────────────────────────────
    def note_fill(self, fill: Fill) -> None:
        self._net_cash += (fill.price * fill.size) * (1 if fill.side is Side.SELL else -1)

    def update_mark(self, token_id: str, fv: float) -> None:
        self._marks[token_id] = fv

    def _inventory_value(self) -> float:
        total = 0.0
        for tok, pos in self._store.positions.items():
            if pos.size > 0:
                total += pos.size * self._marks.get(tok, pos.avg_price)
        return total

    @property
    def net_cash(self) -> float:
        return self._net_cash

    @property
    def inventory_value(self) -> float:
        return self._inventory_value()

    @property
    def equity(self) -> float:
        return self._net_cash + self._inventory_value()

    @property
    def daily_pnl(self) -> float:
        return self.equity - self._day_start_equity

    def reset_day(self) -> None:
        self._day_start_equity = self.equity

    # ── error-rate breaker ──────────────────────────────────────────────
    def note_order_result(self, ok: bool) -> None:
        self._order_attempts += 1
        if not ok:
            self._order_errors += 1

    @property
    def error_rate(self) -> float:
        return self._order_errors / self._order_attempts if self._order_attempts >= 20 else 0.0

    # ── global kill switch ──────────────────────────────────────────────
    def global_halt(self) -> tuple[bool, str]:
        if self._killed:
            return True, self._kill_reason
        if self._total_loss:
            return True, self._total_loss
        if self.daily_pnl <= -self._cfg.daily_loss_kill_usdc:
            return True, f"daily_loss {self.daily_pnl:.0f}"
        if self.error_rate >= self._cfg.max_order_error_rate:
            return True, f"error_rate {self.error_rate:.2f}"
        return False, ""

    def kill(self, reason: str = "manual_kill") -> None:
        self._killed = True
        self._kill_reason = reason
        log.critical("kill_switch_engaged", reason=reason)

    # a breach must hold on this many consecutive checks spanning at least this
    # long before the stop fires: one bad API read (2026-10-29: a disconnect read
    # as a 0 balance) must not halt the engine
    TOTAL_LOSS_CONFIRM_CHECKS = 3
    TOTAL_LOSS_CONFIRM_SPAN_S = 30.0

    def check_total_loss(self, real_equity: float, now: float | None = None) -> str:
        """Cumulative stop. `real_equity` = funder collateral + inventory at fair
        value. Once it sits `total_loss_kill_usdc` below `start_equity_usdc` on
        TOTAL_LOSS_CONFIRM_CHECKS consecutive checks spanning TOTAL_LOSS_CONFIRM_SPAN_S,
        every market goes HALTED (all orders pulled, inventory left for a human) and
        the reason sticks for the life of the process. Returns the reason, '' if fine."""
        cfg = self._cfg
        if cfg.total_loss_kill_usdc <= 0 or cfg.start_equity_usdc <= 0:
            return ""
        if self._total_loss:
            return self._total_loss
        now = time.time() if now is None else now
        drawdown = cfg.start_equity_usdc - real_equity
        if drawdown < cfg.total_loss_kill_usdc:
            if self._tl_strikes:
                log.warning("total_loss_cleared", strikes=self._tl_strikes,
                            equity=round(real_equity, 2))
            self._tl_strikes = 0
            return ""
        if self._tl_strikes == 0:
            self._tl_first_ts = now
        self._tl_strikes += 1
        span = now - self._tl_first_ts
        if (self._tl_strikes < self.TOTAL_LOSS_CONFIRM_CHECKS
                or span < self.TOTAL_LOSS_CONFIRM_SPAN_S):
            log.warning("total_loss_strike", strike=self._tl_strikes, span_s=round(span, 1),
                        drawdown=round(drawdown, 2), equity=round(real_equity, 2))
            return ""
        self._total_loss = (f"total_loss {drawdown:.0f} "
                            f"(equity {real_equity:.0f} of {cfg.start_equity_usdc:.0f})")
        log.critical("total_loss_kill", drawdown=round(drawdown, 2),
                     equity=round(real_equity, 2), strikes=self._tl_strikes)
        return self._total_loss

    # ── per-market evaluation ───────────────────────────────────────────
    def evaluate(
        self, meta: MarketMeta, *, ws_stale: bool, event_group_cost: float
    ) -> RiskDecision:
        halted, why = self.global_halt()
        if halted:
            return RiskDecision(True, False, 0.0, why)
        if ws_stale:
            return RiskDecision(True, False, 0.0, "ws_stale")

        market_notional = self._market_notional(meta)
        total_exposure = self._total_exposure()

        # hard caps -> reduce only
        if market_notional >= self._cfg.max_market_notional_usdc:
            return RiskDecision(False, True, 1.0, "market_cap")
        if event_group_cost >= self._cfg.max_event_group_loss_usdc:
            return RiskDecision(False, True, 1.0, "event_group_cap")
        if total_exposure >= self._cfg.max_total_exposure_usdc:
            return RiskDecision(False, True, 1.0, "total_exposure_cap")

        # soft scaling: taper size as any cap is approached (worst-binding wins)
        scale = min(
            _headroom(market_notional, self._cfg.max_market_notional_usdc),
            _headroom(total_exposure, self._cfg.max_total_exposure_usdc),
            _headroom(event_group_cost, self._cfg.max_event_group_loss_usdc),
        )
        return RiskDecision(False, False, scale, "")

    def _market_notional(self, meta: MarketMeta) -> float:
        """Filled-inventory notional for this market. Deliberately does NOT count
        our own resting BUY orders: those are the quotes we're about to replace,
        and counting them makes the size taper collapse the moment we place a full
        quote (self-reinforcing cancel/replace churn). Worst-case fill is bounded
        instead by small per-quote sizes + the position cap that this drives."""
        total = 0.0
        for tok in (meta.yes.token_id, meta.no.token_id):
            pos = self._store.position(tok)
            total += pos.size * self._marks.get(tok, pos.avg_price or 0.5)
        return total

    def _total_exposure(self) -> float:
        total = 0.0
        for tok, pos in self._store.positions.items():
            if pos.size > 0:
                total += pos.size * self._marks.get(tok, pos.avg_price or 0.5)
        return total


def _headroom(current: float, cap: float) -> float:
    """1.0 well below the cap, tapering to 0 as we approach it (from 70%)."""
    if cap <= 0:
        return 1.0
    frac = current / cap
    if frac <= 0.7:
        return 1.0
    return max(0.0, (1.0 - frac) / 0.3)


class KillFile:
    """Persistent stop marker next to the state db. Written when the total-loss
    stop trips; every later engine start halts while it exists. Delete it only
    after reading why it was written."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def read(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {"reason": "unreadable kill file"}
        except Exception:  # noqa: BLE001 - a corrupt file must still halt
            return {"reason": "unreadable kill file"}

    def write(self, reason: str, equity: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "reason": reason, "equity": round(equity, 4), "ts": time.time(),
            "when": datetime.now(UTC).isoformat(timespec="seconds"),
        }, indent=2) + "\n")
