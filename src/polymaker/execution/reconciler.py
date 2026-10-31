"""Pure reconciliation: desired TargetQuotes vs live orders -> minimal actions.

The strategy emits a target quote set; this computes the smallest cancel/place
set to reach it, applying churn tolerances so we don't burn queue position for
sub-tick or sub-threshold size changes (v1's should_cancel instinct, generalized).
No I/O — the gateway executes the returned plan.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from polymaker.domain import OpenOrder, Quote, TargetQuotes

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class ReconcilePlan:
    to_cancel: list[str] = field(default_factory=list)  # order ids
    to_place: list[Quote] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        return not self.to_cancel and not self.to_place


def reconcile(
    targets: TargetQuotes,
    live: list[OpenOrder],
    *,
    tick: float,
    reprice_ticks: int,
    resize_frac: float,
    min_live_size: float = 0.0,
) -> ReconcilePlan:
    """Diff targets against live orders. Keep live orders that already satisfy a
    target within tolerance; cancel the rest; place targets with no match.

    `min_live_size` (the reward program's per-order minimum): a partially-filled
    remnant below it has stopped scoring, so it never counts as a match for a
    target that is itself at scoring size, even inside `resize_frac`."""
    live_by_key: dict[tuple[str, str], list[OpenOrder]] = defaultdict(list)
    for o in live:
        live_by_key[(o.token_id, o.side.value)].append(o)

    keep: set[str] = set()
    to_place: list[Quote] = []
    price_tol = reprice_ticks * tick + _EPS

    for q in targets.quotes:
        candidates = live_by_key.get((q.token_id, q.side.value), [])
        match: OpenOrder | None = None
        for o in candidates:
            if o.order_id in keep:
                continue
            if min_live_size > 0 and q.size >= min_live_size and o.size + _EPS < min_live_size:
                continue  # remnant no longer scores -> replace it
            price_close = abs(o.price - q.price) <= price_tol
            size_close = q.size <= 0 or abs(o.size - q.size) <= resize_frac * q.size + _EPS
            if price_close and size_close:
                match = o
                break
        if match is not None:
            keep.add(match.order_id)
        else:
            to_place.append(q)

    to_cancel = [o.order_id for o in live if o.order_id not in keep]
    return ReconcilePlan(to_cancel=to_cancel, to_place=to_place)
