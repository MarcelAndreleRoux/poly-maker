"""polymaker command-line interface.

  polymaker scan                 sweep Gamma for political markets -> SQLite
  polymaker markets              rank/browse the catalog
  polymaker markets-add <slug>   append a market to config/markets.toml
  polymaker status               positions / open orders / PnL (reads SQLite)
  polymaker doctor               preflight: wallet auth, balances, WS reachability
  polymaker run [--paper]        start the market maker
  polymaker cancel-all           panic button
  polymaker backtest             tape-replay backtest -> PDF report
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from polymaker import __version__
from polymaker.config import Config

app = typer.Typer(
    name="polymaker",
    help="Maker-only market maker for Polymarket CLOB V2.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the polymaker version."""
    console.print(f"polymaker {__version__}")


@app.command()
def scan(
    config_dir: str = typer.Option("config", help="config directory"),
    min_liquidity: float = typer.Option(1000.0, help="minimum market liquidity (USDC)"),
    all_markets: bool = typer.Option(False, "--all", help="include non-rewards markets"),
    tag: str = typer.Option("politics", help="Gamma tag slug to sweep (politics, sports, crypto, ...)"),
) -> None:
    """Sweep Gamma for the markets under a tag (politics by default), score, persist."""
    from polymaker.catalog.scanner import ScanConfig, run_scan
    from polymaker.catalog.store import CatalogStore

    cfg = Config.load(config_dir)
    store = CatalogStore(cfg.paths.db)

    async def _go() -> int:
        metas = await run_scan(store, ScanConfig(tag_slug=tag, min_liquidity=min_liquidity,
                                                 rewards_only=not all_markets))
        return len(metas)

    n = asyncio.run(_go())
    csv_path = Path(config_dir).parent / "markets.csv"
    written = store.export_csv(csv_path)
    console.print(f"[green]Scanned and stored {n} markets.[/green] "
                  f"Wrote [bold]{csv_path}[/bold] ({written} rows) — open it, pick markets, "
                  f"then `polymaker markets-add <slug>`.")
    store.close()


@app.command()
def markets(
    config_dir: str = typer.Option("config", help="config directory"),
    limit: int = typer.Option(25, help="rows to show"),
) -> None:
    """Show the top scored markets from the catalog."""
    from polymaker.catalog.store import CatalogStore

    cfg = Config.load(config_dir)
    store = CatalogStore(cfg.paths.db)
    rows = store.top(limit)
    if not rows:
        console.print("[yellow]Catalog empty. Run `polymaker scan` first.[/yellow]")
        raise typer.Exit()

    table = Table(title="Political markets by score")
    for col in ("score", "reward/day", "rebate/day", "spread", "tick", "neg", "question"):
        table.add_column(col, justify="right" if col != "question" else "left")
    for meta, sc in rows:
        table.add_row(
            f"{sc.score:.2f}", f"{meta.rewards_daily_rate:.0f}", f"{sc.rebate_potential:.0f}",
            f"{sc.spread:.3f}", f"{meta.tick_size:g}", "Y" if meta.neg_risk else "-",
            meta.question[:60],
        )
    console.print(table)
    console.print("\nAdd one with: [bold]polymaker markets-add <slug>[/bold]  (slugs are in the catalog)")


@app.command()
def rank(
    config_dir: str = typer.Option("config", help="config directory"),
    shares: float = typer.Option(100.0, help="our resting size per side (shares)"),
    dist_c: float = typer.Option(2.5, "--dist", help="our distance from the midpoint in cents (never inside the spread)"),
    top: int = typer.Option(60, help="catalog rows (by scan score) whose live books to read"),
    floor: float = typer.Option(1.0, help="daily payout floor (USD) to flag"),
    out: str = typer.Option("", help="CSV path (default: markets-competition.csv next to markets.csv)"),
) -> None:
    """Rank the scanned catalog by what OUR size would earn today against the competitors
    actually resting in each reward band (live books, Polymarket's own score formula),
    instead of by pool size or 24h volume. Run `polymaker scan` first (per tag)."""
    import csv

    from polymaker.catalog.competition import rank_by_competition
    from polymaker.catalog.store import CatalogStore

    cfg = Config.load(config_dir)
    store = CatalogStore(cfg.paths.db)
    metas = [m for m, _ in store.top(top)]
    store.close()
    if not metas:
        console.print("[yellow]Catalog empty. Run `polymaker scan` first.[/yellow]")
        raise typer.Exit()

    snaps = asyncio.run(rank_by_competition(metas, shares, dist_c, floor=floor, clob_host=cfg.wallet.clob_host))
    table = Table(title=f"Reward share for {shares:g} shares/side {dist_c:g}c from mid ({len(snaps)} books). "
                        "turn = times/day the band's liquidity trades (fill risk)")
    for col in ("$/day", "share", "pool", "comp", "band b/a", "turn", "mid", "sprd", "slug"):
        table.add_column(col, justify="left" if col == "slug" else "right", no_wrap=True)
    for s in snaps:
        table.add_row(
            f"{s.usd_per_day:.2f}", f"{100 * s.our_share:.1f}%", f"{s.pool_per_day:.0f}", f"{s.comp_qmin:.0f}",
            f"{s.band_bid_shares:.0f}/{s.band_ask_shares:.0f}", f"{s.band_turnover:.1f}", f"{s.mid:.2f}",
            f"{100 * s.spread:.1f}c",
            s.slug[:44],
        )
    console.print(table)
    path = Path(out) if out else Path(config_dir).parent / "markets-competition.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(snaps[0].as_dict().keys()) if snaps else ["condition_id"])
        w.writeheader()
        for s in snaps:
            w.writerow(s.as_dict())
    console.print(f"[dim]wrote {path}. Estimate from one book sample; books of markets we quote include our own orders.[/dim]")


@app.command(name="markets-add")
def markets_add(
    slug: str,
    profile: str = typer.Option("political-longdated", help="strategy profile"),
    config_dir: str = typer.Option("config", help="config directory"),
) -> None:
    """Append a market (by slug) to config/markets.toml."""
    from polymaker.catalog.store import CatalogStore

    cfg = Config.load(config_dir)
    store = CatalogStore(cfg.paths.db)
    meta = store.get_by_slug(slug)
    store.close()
    if meta is None:
        console.print(f"[red]No market with slug {slug!r} in the catalog. Run `polymaker scan`.[/red]")
        raise typer.Exit(1)

    path = Path(config_dir) / "markets.toml"
    block = f'\n[[markets]]\nslug    = "{slug}"\nprofile = "{profile}"\nenabled = true\n'
    with path.open("a") as fh:
        fh.write(block)
    console.print(f"[green]Added[/green] {meta.question[:60]!r} to {path}")


@app.command()
def status(config_dir: str = typer.Option("config", help="config directory")) -> None:
    """Show positions, open orders, and marks from the local state DB."""
    from polymaker.state.store import StateStore

    cfg = Config.load(config_dir)
    store = StateStore(cfg.paths.db)
    snap = store.snapshot()
    console.print(f"[bold]Open orders:[/bold] {snap['open_orders']}")
    positions: dict[str, Any] = snap["positions"]  # type: ignore[assignment]
    if not positions:
        console.print("[dim]No open positions.[/dim]")
    else:
        table = Table(title="Positions")
        table.add_column("token")
        table.add_column("size", justify="right")
        table.add_column("avg", justify="right")
        for tok, p in positions.items():
            table.add_row(tok[:16] + "…", f"{p['size']:.2f}", f"{p['avg_price']:.3f}")
        console.print(table)
    store.close()


@app.command()
def pnl(config_dir: str = typer.Option("config", help="config directory")) -> None:
    """Show PnL from the recorded snapshots (equity, daily PnL, fills)."""
    import sqlite3

    cfg = Config.load(config_dir)
    conn = sqlite3.connect(cfg.paths.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT ts, equity, net_cash, inventory_value, daily_pnl FROM pnl_snapshots "
        "ORDER BY ts DESC LIMIT 1"
    ).fetchall()
    if not rows:
        console.print("[yellow]No PnL snapshots yet (run the engine first).[/yellow]")
    else:
        r = rows[0]
        color = "green" if r["daily_pnl"] >= 0 else "red"
        console.print(f"[bold]equity:[/bold] {r['equity']:.4f}  "
                      f"[bold]inventory:[/bold] {r['inventory_value']:.4f}  "
                      f"[bold]net cash:[/bold] {r['net_cash']:.4f}")
        console.print(f"[bold]daily PnL:[/bold] [{color}]{r['daily_pnl']:+.4f}[/{color}] pUSD")
    nfills = conn.execute("SELECT COUNT(*) n FROM fills").fetchone()["n"]
    console.print(f"[dim]total fills recorded: {nfills}[/dim]")
    conn.close()


@app.command(name="export-csv")
def export_csv(
    config_dir: str = typer.Option("config", help="config directory"),
    out: str = typer.Option("markets.csv", help="output CSV path"),
    limit: int = typer.Option(500, help="max rows"),
) -> None:
    """Export the scored market catalog to a CSV for easy picking."""
    from polymaker.catalog.store import CatalogStore

    cfg = Config.load(config_dir)
    store = CatalogStore(cfg.paths.db)
    n = store.export_csv(out, limit)
    store.close()
    console.print(f"[green]Wrote {n} markets to {out}.[/green]")


@app.command()
def doctor(config_dir: str = typer.Option("config", help="config directory")) -> None:
    """Preflight checks: config, wallet auth, balance/allowance, WS reachability."""
    from polymaker.doctor import run_doctor

    cfg = Config.load(config_dir)
    ok = asyncio.run(run_doctor(cfg, console))
    raise typer.Exit(0 if ok else 1)


@app.command()
def run(
    config_dir: str = typer.Option("config", help="config directory"),
    paper: bool = typer.Option(False, "--paper", help="paper mode: full pipeline, no orders posted"),
) -> None:
    """Start the market maker."""
    from polymaker.engine import Engine
    from polymaker.logging import configure

    cfg = Config.load(config_dir)
    configure(json_file=Path(cfg.paths.log_dir) / ("paper.jsonl" if paper else "live.jsonl"))
    if cfg.engine.loop == "uvloop":
        try:
            import uvloop

            uvloop.install()
        except Exception:  # noqa: BLE001
            pass

    engine = Engine(cfg, paper=paper)

    async def _go() -> None:
        try:
            await engine.run_forever()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await engine.shutdown()

    console.print(f"[bold green]Starting polymaker[/bold green] ({'PAPER' if paper else 'LIVE'})…")
    try:
        asyncio.run(_go())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")


@app.command()
def livetest(
    config_dir: str = typer.Option("config", help="config directory"),
    notional: float = typer.Option(5.0, help="order notional in USDC"),
) -> None:
    """Live wallet round-trip: place a deep post-only order and cancel it (~$5)."""
    from polymaker.livetest import run_livetest

    cfg = Config.load(config_dir)
    ok = asyncio.run(run_livetest(cfg, console, notional))
    raise typer.Exit(0 if ok else 1)


@app.command()
def moneydoctor(
    config_dir: str = typer.Option("config", help="config directory"),
) -> None:
    """LIVE trading self-test: rest a limit, then market buy + sell (spends a little)."""
    from polymaker.moneydoctor import run_moneydoctor

    cfg = Config.load(config_dir)
    ok = asyncio.run(run_moneydoctor(cfg, console))
    raise typer.Exit(0 if ok else 1)


@app.command()
def rewards(
    config_dir: str = typer.Option("config", help="config directory"),
    days: int = typer.Option(2, help="settled UTC days to fetch besides today"),
    out: str = typer.Option("", help="append JSON lines here (default: rewards.jsonl next to the state db)"),
    loop: int = typer.Option(0, help="repeat every N seconds (0 = once)"),
    reconcile_only: bool = typer.Option(False, "--reconcile", help="no API call: reconcile the file as it is"),
) -> None:
    """Snapshot Polymarket's own reward accounting for our wallet: per-market earnings and
    earning_percentage (our real share of each pool) for today and the last N settled days,
    plus the wallet balances at that moment. After each snapshot, reconcile: did every
    settled day's earnings actually land in the wallet after 00:00 UTC (or did the 1 USD
    payout floor eat them)? This is the only ground truth a backtest has to assume."""
    import json
    import time as _time

    from polymaker.rewards import (
        format_reconcile,
        journal_collateral_snaps,
        journal_fill_ts,
        reconcile,
        snapshot_rewards,
        summarize,
    )

    cfg = Config.load(config_dir)
    path = Path(out) if out else Path(cfg.paths.db).with_name("rewards.jsonl")
    jdir = Path(cfg.paths.journal_dir)
    if not jdir.is_absolute():
        jdir = Path(cfg.paths.db).parent / jdir

    def _reconcile() -> None:
        if not path.exists():
            return
        rows = []
        with path.open() as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        recs = reconcile(rows, fill_ts=journal_fill_ts(jdir), extra_snaps=journal_collateral_snaps(jdir))
        if not recs:
            console.print("[dim]reconcile: no settled day with earnings yet[/dim]")
        for rec in recs:
            colour = {"credited": "green", "pending": "yellow", "no_data": "dim"}.get(rec["status"], "red")
            console.print(f"[{colour}]reconcile: {format_reconcile(rec)}[/{colour}]")

    if reconcile_only:
        _reconcile()
        return
    while True:
        try:
            rows = snapshot_rewards(cfg, days)
            with path.open("a") as fh:
                for r in rows:
                    fh.write(json.dumps(r, default=str) + "\n")
            for r in rows:
                console.print(summarize(r))
            bal = rows[0].get("balances") if rows else None
            if bal:
                console.print("balances: " + " ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                                      for k, v in bal.items()))
            console.print(f"[dim]{len(rows)} rows appended to {path}[/dim]")
            _reconcile()
        except Exception as exc:  # noqa: BLE001 - a loop must survive one bad poll
            console.print(f"[red]rewards snapshot failed: {exc}[/red]")
            if not loop:
                raise typer.Exit(1) from exc
        if not loop:
            break
        _time.sleep(loop)


@app.command(name="cancel-all")
def cancel_all(config_dir: str = typer.Option("config", help="config directory")) -> None:
    """Cancel all open orders for the wallet (panic button)."""
    from polymaker.execution.gateway import ExecutionGateway

    cfg = Config.load(config_dir)
    gw = ExecutionGateway(cfg)

    async def _go() -> None:
        await gw.connect()
        await gw.cancel_all()

    asyncio.run(_go())
    console.print("[green]Sent cancel-all.[/green]")


if __name__ == "__main__":
    app()


@app.command()
def backtest(
    config_dir: str = typer.Option("config", help="config directory"),
    market: list[str] | None = typer.Option(  # noqa: B008 - typer idiom
        None, "--market", "-m",
        help="slug or slug=profile (repeatable). Default: enabled entries in markets.toml",
    ),
    days: int = typer.Option(30, help="lookback window in days"),
    out: str = typer.Option("reports/backtest.pdf", help="output PDF path"),
    spread_ticks: int = typer.Option(2, help="assumed book spread in ticks (no historical L2)"),
    touch_depth: float = typer.Option(300.0, help="assumed shares queued ahead of us at a level"),
    bankroll: float | None = typer.Option(
        None, "--bankroll",
        help="USDC collateral on the funder: BUY quotes it cannot fund are not placed, as the "
             "engine does live. Default: unconstrained",
    ),
    reward_share: float = typer.Option(0.10, help="assumed share of the daily reward pool"),
    cache_dir: str = typer.Option("data/backtest", help="historical data cache directory"),
    sensitivity: bool = typer.Option(True, help="also run one-knob variants of the defended profile"),
    polydata: str | None = typer.Option(
        None, "--polydata",
        help="poly_data orderFilled.csv: replay the full on-chain fill tape instead of the "
             "data-api's newest 10k prints (e.g. ../poly_data/data/orderFilled.csv)",
    ),
) -> None:
    """Replay public trade history through the strategy core and write a PDF report."""
    from polymaker.backtest.data import load_market_history, prefetch_polydata
    from polymaker.backtest.report import AS_IS, DEFENDED, FIXED, build_report
    from polymaker.backtest.sim import SimParams, run_sim
    from polymaker.logging import configure

    # strategy behaviour before the defenses were added (the pure-core knobs off)
    LEGACY_CORE = {"skew_min_ticks": 0.0, "trade_out_enabled": False, "exit_lean": False,
                   "entry_yields_to_exit": False, "event_keeps_exits": False,
                   "replace_below_reward_min": False, "inventory_metric": "yes_notional"}
    configure(level="WARNING")  # the store logs every fill at INFO; keep the console quiet
    cfg = Config.load(config_dir)
    targets: list[tuple[str, str]] = []
    default_profile = cfg.enabled_markets[0].profile if cfg.enabled_markets else ""
    if market:
        for spec in market:
            slug, _, prof = spec.partition("=")
            targets.append((slug, prof or default_profile))
    else:
        targets = [(e.slug or "", e.profile) for e in cfg.enabled_markets if e.slug]
    if not targets:
        console.print("[red]no markets: pass --market slug=profile or enable entries in markets.toml[/red]")
        raise typer.Exit(1)

    async def _load(slug: str):  # type: ignore[no-untyped-def]
        return await load_market_history(slug, days=days, cache_root=cache_dir, polydata=polydata)

    if polydata:
        if not Path(polydata).is_file():
            console.print(f"[red]--polydata: {polydata} not found[/red]")
            raise typer.Exit(1)
        console.print(f"[cyan]poly_data[/cyan]: scanning {polydata} for {len(targets)} market(s) ...")
        asyncio.run(prefetch_polydata(polydata, [s for s, _ in targets], cache_dir))

    runs = []
    sens: dict[str, list] = {}  # type: ignore[type-arg]
    for slug, prof_name in targets:
        if prof_name not in cfg.profiles:
            console.print(f"[yellow]{slug}: unknown profile {prof_name!r}, skipped[/yellow]")
            continue
        console.print(f"[cyan]{slug}[/cyan]: loading {days}d of history ...")
        hist = asyncio.run(_load(slug))
        if hist is None or not hist.bars or not hist.tape:
            console.print(f"[yellow]{slug}: no history available (resolved/unknown market), skipped[/yellow]")
            continue
        console.print(f"  tape: {len(hist.tape)} prints over {hist.days:.1f}d from {hist.tape_source}")
        profile = cfg.profiles[prof_name]
        # as-is / fixed run the profile WITHOUT the defensive knobs so the third
        # scenario isolates their effect
        undefended = profile.with_overrides(LEGACY_CORE)
        scen = {}
        for name, bugs, prof in ((AS_IS, True, undefended), (FIXED, False, undefended),
                                 (DEFENDED, False, profile)):
            params = SimParams(spread_ticks=spread_ticks, touch_depth=touch_depth, engine_bugs=bugs,
                               bankroll=bankroll)
            r = run_sim(hist, prof, prof_name, cfg.risk, params, name)
            scen[name] = r
            console.print(f"  {name:17s} fills={len(r.fills):4d}  pnl={r.total_pnl:+8.2f}  "
                          f"markout5m={r.markout_total_usd(300):+7.2f}  scored={r.scored_fraction:.0%}  "
                          f"maxinv={r.max_inv_value:6.0f}  tradeouts={len(r.trade_outs):3d}  "
                          f"rebates={r.rebates_usd:+.2f}  "
                          f"rewards@{reward_share:.0%}={r.rewards_usd(reward_share):+.2f}")
        if sensitivity:
            variants = [
                ("c_tox x2", {"c_tox": profile.c_tox * 2}),
                ("skew floor 4 ticks", {"skew_min_ticks": 4.0}),
                ("trade-out drift 5", {"trade_out_drift_ticks": 5.0}),
                ("trade-out drift 20", {"trade_out_drift_ticks": 20.0}),
                ("no trade-out", {"trade_out_enabled": False}),
                ("tox decay 6h", {"tox_decay_halflife_s": 6 * 3600.0}),
                ("entry yields to exit", {"entry_yields_to_exit": True}),
            ]
            for label, over in variants:
                params = SimParams(spread_ticks=spread_ticks, touch_depth=touch_depth, engine_bugs=False,
                                   bankroll=bankroll)
                r = run_sim(hist, profile.with_overrides(over), prof_name, cfg.risk, params, label)
                sens.setdefault(hist.meta.condition_id, []).append(r)
                console.print(f"    {label:22s} pnl={r.total_pnl:+8.2f}  scored={r.scored_fraction:.0%}  "
                              f"maxdd={r.max_drawdown:5.2f}  total@{reward_share:.0%}="
                              f"{r.total_pnl + r.rewards_usd(reward_share):+.2f}")
        runs.append((hist, scen))
    if not runs:
        console.print("[red]nothing to report[/red]")
        raise typer.Exit(1)
    path = build_report(runs, Path(out), days=days, config_dir=config_dir, share=reward_share,
                        sensitivity=sens)
    console.print(f"[green]report written:[/green] [bold]{path.resolve()}[/bold]")
