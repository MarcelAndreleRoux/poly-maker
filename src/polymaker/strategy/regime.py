"""Per-market regime decision (see the README).

Priority order, highest first:
  HALTED       kill switch / stale data / resolved / past halt-before window /
               inside a scheduled catalyst window
  EVENT        active cooloff, or a fresh sweep / fair-value jump (or a catalyst
               window when the profile keeps exits up through it)
  REDUCE_ONLY  inventory at hard cap, inside the reduce-only end-date window,
               in the flatten lead-in before a catalyst, or fair value outside
               the profile's fv_band (no new entries near 0 / 1)
  TRENDING     persistent one-sided flow or elevated short/long vol
  QUIET        default farming posture
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from polymaker.config import CatalystWindow, StrategyProfile
from polymaker.domain import Regime


@dataclass(frozen=True, slots=True)
class RegimeInputs:
    now: float
    tick: float
    fv: float
    prev_fv: float | None
    vol_ratio: float
    flow_z: float
    inventory_util: float  # |net notional| / q_max, >=0
    hours_to_end: float | None
    sweep_flagged: bool = False
    market_resolved: bool = False
    ws_stale: bool = False
    risk_halt: bool = False
    risk_reduce_only: bool = False
    catalyst_active: bool = False  # now inside a scheduled catalyst window
    catalyst_flatten: bool = False  # in the lead-in before one: work the bag down


@dataclass(frozen=True, slots=True)
class CatalystState:
    active: bool
    flatten: bool
    note: str
    next_change_s: float  # seconds until the next boundary (inf if none ahead)


def catalyst_state(
    windows: Iterable[CatalystWindow], now: float, default_flatten_hours: float
) -> CatalystState:
    """Where `now` sits relative to the scheduled catalyst windows.

    active   = inside some [start, end]; flatten = inside some [start - lead, start).
    next_change_s lets the quoter wake exactly on the next boundary instead of
    on its slow baseline tick.
    """
    active = flatten = False
    note = ""
    nxt = math.inf
    for w in windows:
        lead = (w.flatten_hours if w.flatten_hours is not None else default_flatten_hours) * 3600.0
        lead_start = w.start_ts - max(0.0, lead)
        if w.start_ts <= now < w.end_ts:
            if not active:
                note = w.note
            active = True
            nxt = min(nxt, w.end_ts - now)
        elif lead_start <= now < w.start_ts:
            if not active and not flatten:
                note = w.note
            flatten = True
            nxt = min(nxt, w.start_ts - now)
        elif now < lead_start:
            nxt = min(nxt, lead_start - now)
    return CatalystState(active, flatten and not active, note, nxt)


class RegimeMachine:
    """Stateful regime decider for one market (tracks the EVENT cooloff)."""

    __slots__ = ("_event_until",)

    def __init__(self) -> None:
        self._event_until: float = 0.0

    def decide(self, inp: RegimeInputs, p: StrategyProfile) -> Regime:
        # 1. hard halts
        if inp.risk_halt or inp.ws_stale or inp.market_resolved:
            return Regime.HALTED
        if inp.hours_to_end is not None and inp.hours_to_end <= p.halt_before_hours:
            return Regime.HALTED
        if inp.catalyst_active and not p.catalyst_keeps_exits:
            return Regime.HALTED

        # 2. events (sweep / jump / active cooloff / catalyst window keeping exits)
        if inp.catalyst_active:
            return Regime.EVENT  # no cool-off: the window itself bounds it
        jump_ticks = abs(inp.fv - inp.prev_fv) / inp.tick if inp.prev_fv is not None else 0.0
        if inp.sweep_flagged or jump_ticks >= p.event_jump_ticks:
            self._event_until = inp.now + p.event_cooloff_s
            return Regime.EVENT
        if inp.now < self._event_until:
            return Regime.EVENT

        # 3. reduce-only
        if inp.risk_reduce_only or inp.inventory_util >= 1.0:
            return Regime.REDUCE_ONLY
        if inp.hours_to_end is not None and inp.hours_to_end <= p.reduce_only_hours:
            return Regime.REDUCE_ONLY
        if inp.catalyst_flatten:
            return Regime.REDUCE_ONLY
        if not (p.fv_band_min <= inp.fv <= p.fv_band_max):
            return Regime.REDUCE_ONLY

        # 4. trending
        if abs(inp.flow_z) >= p.trend_flow_z or inp.vol_ratio >= p.trend_vol_ratio:
            return Regime.TRENDING

        # 5. default
        return Regime.QUIET

    @property
    def in_cooloff(self) -> bool:
        return self._event_until > 0.0

    def cooloff_remaining(self, now: float) -> float:
        """Seconds until the EVENT cool-off expires (0 if not cooling off)."""
        return max(0.0, self._event_until - now)
