"""Polymarket's own view of our liquidity rewards.

`get_earnings_for_user_for_day(day)` returns, per market we quoted, the earnings
and `earning_percentage` (our real share of that day's pool) — the one number no
backtest can know. Snapshot it to JSON lines for later evaluation.

(`get_user_earnings_and_markets_config` and `get_current_rewards` page through every
reward market on the exchange — thousands of requests, minutes per call — so they are
deliberately not used; pool configs come from Gamma instead.)
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from polymaker.config import Config


def _client(cfg: Config) -> Any:
    """Same construction as the execution gateway (L2 creds derived from the PK)."""
    from py_clob_client_v2.client import ClobClient

    sec = cfg.secrets
    if not sec.has_wallet:
        raise RuntimeError("no wallet configured (set PK and BROWSER_ADDRESS in .env)")
    client = ClobClient(
        host=cfg.wallet.clob_host, chain_id=cfg.wallet.chain_id, key=sec.pk,
        signature_type=cfg.wallet.signature_type, funder=sec.browser_address,
    )
    client.set_api_creds(client.create_or_derive_api_key())
    return client


def snapshot_rewards(
    cfg: Config, days: int, client: Any | None = None, with_balances: bool = True
) -> list[dict[str, Any]]:
    """One row per UTC day: today (running) plus the last `days` settled days.
    Raw API payloads are kept verbatim; a failed call is recorded, not raised.
    Each row also carries the wallet balances at snapshot time (see `reconcile`)."""
    client = client or _client(cfg)
    now = datetime.now(UTC)
    balances = wallet_balances(cfg, client) if with_balances else None
    rows: list[dict[str, Any]] = []
    for back in range(days + 1):
        day = (now - timedelta(days=back)).strftime("%Y-%m-%d")
        row: dict[str, Any] = {"ts": time.time(), "day": day, "settled": back > 0}
        if balances is not None:
            row["balances"] = balances
        try:
            row["markets"] = client.get_earnings_for_user_for_day(day)
        except Exception as exc:  # noqa: BLE001 - keep collecting the other days
            row["markets_error"] = str(exc)
        try:
            row["total"] = client.get_total_earnings_for_user_for_day(day)
        except Exception as exc:  # noqa: BLE001
            row["total_error"] = str(exc)
        rows.append(row)
    return rows


def summarize(row: dict[str, Any]) -> str:
    parts = [row["day"], "settled" if row.get("settled") else "running"]
    if "total" in row:
        parts.append(f"total={row['total']}")
    for m in row.get("markets") or []:
        if isinstance(m, dict):
            share = m.get("earning_percentage")
            parts.append(f"{str(m.get('market') or m.get('condition_id') or m.get('question', '?'))[:28]}: "
                         f"earn={m.get('earnings')} share={share}")
    for k in ("markets_error", "total_error"):
        if k in row:
            parts.append(f"{k}={row[k][:80]}")
    return " | ".join(str(p) for p in parts)


# ── payout reconciliation ─────────────────────────────────────────────────────
#
# The earnings API says what we earned; it does not say whether it was paid.
# Third-party guides claim a 1 USD minimum daily payout (per market, no rollover);
# the official docs are silent. Snapshot the wallet alongside each earnings poll
# and compare balances across the 00:00 UTC settlement boundary.

PUSD = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"  # Polymarket pUSD (CLOB collateral, chain 137)
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # bridged USDC.e (the asset_address on earnings rows)
USDC = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"  # native USDC
TRACKED_TOKENS = {"pusd": PUSD, "usdce": USDC_E, "usdc": USDC}
BALANCE_KEYS = ("clob_collateral", "pusd", "usdce", "usdc")


def erc20_balance(rpc: str, token: str, holder: str, decimals: int = 6, timeout: float = 15.0) -> float:
    """ERC-20 balanceOf(holder) via raw JSON-RPC eth_call (no web3 import)."""
    import httpx

    data = "0x70a08231" + holder.lower().removeprefix("0x").rjust(64, "0")
    payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_call",
               "params": [{"to": token, "data": data}, "latest"]}
    r = httpx.post(rpc, json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise RuntimeError(str(body["error"])[:160])
    return int(body["result"], 16) / 10**decimals


def wallet_balances(cfg: Config, client: Any) -> dict[str, Any]:
    """CLOB collateral plus on-chain pUSD / USDC.e / USDC of the funder. Each read
    fails independently (recorded as `<key>_error`), never raises."""
    out: dict[str, Any] = {}
    try:
        from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams

        from polymaker.execution.gateway import parse_collateral

        out["clob_collateral"] = parse_collateral(
            client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        )
    except Exception as exc:  # noqa: BLE001
        out["clob_collateral_error"] = str(exc)[:120]
    rpc = cfg.secrets.polygon_rpc or cfg.wallet.polygon_rpc
    holder = cfg.secrets.browser_address or ""
    for name, token in TRACKED_TOKENS.items():
        try:
            out[name] = erc20_balance(rpc, token, holder)
        except Exception as exc:  # noqa: BLE001
            out[f"{name}_error"] = str(exc)[:120]
    return out


def day_earnings(row: dict[str, Any]) -> float | None:
    """Earned USD for the row's day from the API payloads (total first, then markets)."""
    for key in ("total", "markets"):
        items = row.get(key)
        if isinstance(items, list):
            vals = [float(t.get("earnings") or 0) for t in items if isinstance(t, dict)]
            if vals or key == "markets":
                return sum(vals)
    return None


def _midnight_after(day: str) -> float:
    d = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)
    return d.timestamp()


def reconcile(
    rows: list[dict[str, Any]],
    now: float | None = None,
    floor: float = 1.0,
    grace_h: float = 24.0,
    fill_ts: list[float] | None = None,
    tolerance: float = 0.8,
    extra_snaps: list[tuple[float, dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """One record per settled day with earnings: did the wallet grow by ~that much
    after the following 00:00 UTC?

    status: credited | pending (settlement boundary < grace_h ago; payouts have been
            seen to lag midnight by hours) | not_credited | no_data (no balance
            snapshot on one side of the boundary).
    extra_snaps: additional (ts, balances) samples, e.g. the engine journal's
            collateral readings (`journal_collateral_snaps`), which cover the time
            before this tracker existed and sample far more often.
    floor_suspect: not credited and earnings < floor — the payout floor explanation.
    fills_in_window: fills between the two snapshots move collateral too; > 0 means
    the balance delta is not a clean payout signal.
    """
    now = time.time() if now is None else now
    merged: dict[float, dict[str, Any]] = {ts: dict(b) for ts, b in (extra_snaps or [])}
    for r in rows:
        if isinstance(r.get("balances"), dict):
            merged[r["ts"]] = r["balances"]
    snaps = sorted(merged.items())
    latest_by_day: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.get("settled") and "day" in r:
            prev = latest_by_day.get(r["day"])
            if prev is None or r["ts"] >= prev["ts"]:
                latest_by_day[r["day"]] = r
    out: list[dict[str, Any]] = []
    for day in sorted(latest_by_day):
        row = latest_by_day[day]
        earned = day_earnings(row)
        if earned is None or earned <= 0:
            continue
        boundary = _midnight_after(day)
        before = [s for s in snaps if s[0] < boundary]
        after = [s for s in snaps if s[0] >= boundary]
        rec: dict[str, Any] = {"day": day, "earned": round(earned, 6), "boundary_ts": boundary,
                               "floor_suspect": False, "fills_in_window": 0}
        if not before or not after:
            rec["status"] = "no_data"
            rec["missing"] = "before" if not before else "after"
            out.append(rec)
            continue
        b_ts, b = before[-1]
        a_ts, a = after[-1]
        rec["before_ts"], rec["after_ts"] = b_ts, a_ts
        deltas = {k: round(float(a[k]) - float(b[k]), 6) for k in BALANCE_KEYS
                  if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float))}
        rec["deltas"] = deltas
        rec["fills_in_window"] = sum(1 for t in (fill_ts or []) if b_ts <= t <= a_ts)
        hit = [k for k, d in deltas.items() if d >= tolerance * earned]
        if hit:
            rec["status"] = "credited"
            rec["credited_to"] = hit
        elif now < boundary + grace_h * 3600:
            rec["status"] = "pending"
        else:
            rec["status"] = "not_credited"
            rec["floor_suspect"] = earned < floor
        out.append(rec)
    return out


def journal_fill_ts(journal_dir: Path) -> list[float]:
    """Timestamps of our fills (`user_trade` records) across the journal files."""
    out: list[float] = []
    if not journal_dir.is_dir():
        return out
    for p in sorted(journal_dir.glob("*.jsonl")):
        with p.open() as fh:
            for line in fh:
                if '"user_trade"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "user_trade":
                    out.append(float(r["ts"]))
    return out


def journal_collateral_snaps(journal_dir: Path, every_s: float = 600.0) -> list[tuple[float, dict[str, Any]]]:
    """(ts, {"clob_collateral": x}) from the engine journal's `pnl` / `collateral`
    records, thinned to one sample per `every_s` (the engine writes one per reconcile)."""
    out: list[tuple[float, dict[str, Any]]] = []
    if not journal_dir.is_dir():
        return out
    last = -float("inf")
    for p in sorted(journal_dir.glob("*.jsonl")):
        with p.open() as fh:
            for line in fh:
                if '"collateral"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = r.get("kind")
                data = r.get("data") or {}
                if kind == "pnl":
                    val = data.get("collateral")
                elif kind == "collateral":
                    val = data.get("pusd")
                else:
                    continue
                ts = float(r.get("ts", 0))
                if not isinstance(val, (int, float)) or ts - last < every_s:
                    continue
                last = ts
                out.append((ts, {"clob_collateral": float(val)}))
    return out


def format_reconcile(rec: dict[str, Any]) -> str:
    parts = [rec["day"], f"earned={rec['earned']:.4f}", rec["status"]]
    if rec.get("deltas"):
        parts.append("Δ " + " ".join(f"{k}={v:+.4f}" for k, v in rec["deltas"].items()))
    if rec.get("credited_to"):
        parts.append(f"credited_to={','.join(rec['credited_to'])}")
    if rec.get("missing"):
        parts.append(f"missing={rec['missing']}")
    if rec.get("fills_in_window"):
        parts.append(f"fills_in_window={rec['fills_in_window']} (delta not a clean payout signal)")
    if rec.get("floor_suspect"):
        parts.append("FLOOR? earned < 1 USD and never credited")
    return " | ".join(parts)
