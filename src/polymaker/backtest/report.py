"""PDF report for backtest runs: matplotlib charts embedded in a reportlab doc.

One summary page, one assumptions page, two pages per market, one A/B page
comparing the engine-as-is scenario with the fixed scenario.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from polymaker.backtest.data import MarketHistory  # noqa: E402
from polymaker.backtest.sim import SimResult  # noqa: E402

# ── palette (validated: dataviz reference instance, light mode) ─────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"  # slot 1: fixed scenario / positive
ORANGE = "#eb6834"  # slot 2: engine as-is
AQUA = "#1baf7a"  # slot 3: fixed + defenses (sub-3:1 on light surface -> always direct-labeled)
RED = "#e34948"  # diverging negative pole
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue, 250..650
AS_IS = "engine as-is"
FIXED = "fixed"
DEFENDED = "fixed + defenses"
SCENARIOS = (AS_IS, FIXED, DEFENDED)
SCEN_COLOR = {AS_IS: ORANGE, FIXED: BLUE, DEFENDED: AQUA}
REGIMES = ["QUIET", "TRENDING", "EVENT", "REDUCE_ONLY", "HALTED"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 8,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 1.4,
})

_W = 6.6  # inches, fits A4 text width


def _img(path: Path, width_pt: float) -> Image:
    """reportlab Image scaled to a width, keeping the PNG's aspect ratio."""
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    return Image(str(path), width=width_pt, height=width_pt * h / w)


def _png(fig: plt.Figure, tmp: Path, name: str) -> Path:
    p = tmp / f"{name}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return p


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, UTC)


def _date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


# ── charts ───────────────────────────────────────────────────────────────


def fig_price_fills(r: SimResult, tmp: Path, key: str) -> Path:
    fig, ax = plt.subplots(figsize=(_W, 2.3))
    xs = [_dt(t) for t, _ in r.price]
    ax.plot(xs, [p for _, p in r.price], color=MUTED, linewidth=1.0, label="YES mid")
    buys = [(f.ts, f.yes_price) for f in r.fills if f.bought_yes and not f.taker]
    sells = [(f.ts, f.yes_price) for f in r.fills if not f.bought_yes and not f.taker]
    outs = [(f.ts, f.yes_price) for f in r.fills if f.taker]
    if buys:
        ax.scatter([_dt(t) for t, _ in buys], [p for _, p in buys], s=14, color=BLUE,
                   edgecolor=SURFACE, linewidth=0.6, zorder=3, label=f"bought YES-equiv ({len(buys)})")
    if sells:
        ax.scatter([_dt(t) for t, _ in sells], [p for _, p in sells], s=14, color=ORANGE,
                   edgecolor=SURFACE, linewidth=0.6, zorder=3, label=f"sold YES-equiv ({len(sells)})")
    if outs:
        ax.scatter([_dt(t) for t, _ in outs], [p for _, p in outs], s=34, marker="x", color=RED,
                   linewidth=1.2, zorder=4, label=f"trade-out, taker ({len(outs)})")
    ax.set_title(f"YES mid and our fills  ({r.scenario})")
    ax.set_ylabel("price")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    _date_axis(ax)
    return _png(fig, tmp, f"price_{key}")


def fig_equity(runs: dict[str, SimResult], tmp: Path, key: str) -> Path:
    fig, ax = plt.subplots(figsize=(_W, 2.3))
    ax.axhline(0, color=BASELINE, linewidth=0.8)
    for scen, r in runs.items():
        xs = [_dt(t) for t, _ in r.equity]
        ys = [e for _, e in r.equity]
        ax.plot(xs, ys, color=SCEN_COLOR[scen], label=scen)
        if xs:
            ax.annotate(f"{scen}  {ys[-1]:+.2f}", (xs[-1], ys[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=7.5, color=INK2, va="center")
    ax.set_title("Mark-to-mid equity (cash + inventory at mid), USD, excludes rewards")
    ax.set_ylabel("USD")
    ax.legend(loc="best", ncol=2)
    _date_axis(ax)
    ax.margins(x=0.12)
    return _png(fig, tmp, f"equity_{key}")


def fig_markout(runs: dict[str, SimResult], tmp: Path, key: str) -> Path:
    fig, ax = plt.subplots(figsize=(_W, 1.45))
    horizons = next(iter(runs.values())).params.markout_horizons
    labels = [f"{h // 60} min" for h in horizons]
    n = len(runs)
    w = 0.8 / n
    ax.axhline(0, color=BASELINE, linewidth=0.8)
    for i, (scen, r) in enumerate(runs.items()):
        vals = [r.markout_mean_ticks(h) for h in horizons]
        xs = [j + (i - (n - 1) / 2) * w for j in range(len(horizons))]
        bars = ax.bar(xs, vals, width=w * 0.92, color=SCEN_COLOR[scen], label=scen)
        for b, v in zip(bars, vals, strict=False):
            ax.annotate(f"{v:+.1f}", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3 if v >= 0 else -9), textcoords="offset points",
                        ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(horizons)), labels)
    ax.set_title("Size-weighted post-fill markout (ticks, + = price moved our way)")
    ax.set_ylabel("ticks")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    return _png(fig, tmp, f"markout_{key}")


def fig_regime(runs: dict[str, SimResult], tmp: Path, key: str) -> Path:
    fig, ax = plt.subplots(figsize=(_W, 1.3))
    for i, (_scen, r) in enumerate(runs.items()):
        share = r.regime_share
        left = 0.0
        for j, reg in enumerate(REGIMES):
            v = share.get(reg, 0.0)
            if v <= 0:
                continue
            ax.barh(i, v, left=left, color=RAMP[j], edgecolor=SURFACE, linewidth=1.5,
                    label=reg if i == 0 else None)
            if v >= 0.08:
                ax.text(left + v / 2, i, f"{reg} {v:.0%}", ha="center", va="center",
                        fontsize=7, color="#ffffff" if j >= 2 else INK)
            left += v
    ax.set_yticks(range(len(runs)), list(runs))
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(False)
    ax.set_title("Time in regime")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=5)
    return _png(fig, tmp, f"regime_{key}")


def fig_daily(runs: dict[str, SimResult], tmp: Path, key: str) -> Path:
    fig, ax = plt.subplots(figsize=(_W, 1.45))
    ax.axhline(0, color=BASELINE, linewidth=0.8)
    n = len(runs)
    w = 0.8 / n
    for i, (scen, r) in enumerate(runs.items()):
        xs = [j + (i - (n - 1) / 2) * w for j in range(len(r.daily_pnl))]
        ax.bar(xs, [p for _, p in r.daily_pnl], width=w * 0.92, color=SCEN_COLOR[scen], label=scen)
    days = [d for d, _ in next(iter(runs.values())).daily_pnl]
    step = max(1, len(days) // 8)
    ax.set_xticks(range(0, len(days), step), [d[5:] for d in days[::step]])
    ax.set_title("Daily mark-to-mid PnL, USD (excludes rewards)")
    ax.legend(loc="upper left", ncol=2)
    return _png(fig, tmp, f"daily_{key}")


def fig_toxicity(runs: dict[str, SimResult], tmp: Path, key: str) -> Path:
    fig, ax = plt.subplots(figsize=(_W, 1.3))
    for scen, r in runs.items():
        ax.plot([_dt(t) for t, _ in r.tox_series], [v for _, v in r.tox_series],
                color=SCEN_COLOR[scen], label=scen)
    ax.set_title("Estimator toxicity as the quoter saw it (drives spread widening and size cuts)")
    ax.set_ylabel("toxicity")
    ax.legend(loc="upper left", ncol=2)
    _date_axis(ax)
    return _png(fig, tmp, f"tox_{key}")


# ── document ─────────────────────────────────────────────────────────────


def _styles() -> dict[str, ParagraphStyle]:
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontSize=20, leading=24, textColor=INK,
                                alignment=0, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=9, textColor=INK2, spaceAfter=10),
        "h1": ParagraphStyle("h1", parent=ss["Heading2"], fontSize=13, textColor=INK, spaceBefore=6,
                             spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=ss["Heading3"], fontSize=10.5, textColor=INK,
                             spaceBefore=6, spaceAfter=3),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontSize=8.8, leading=12, textColor=INK),
        "small": ParagraphStyle("sm", parent=ss["Normal"], fontSize=7.5, leading=10, textColor=INK2),
        "bullet": ParagraphStyle("bl", parent=ss["Normal"], fontSize=8.6, leading=11.5,
                                 textColor=INK, leftIndent=10, bulletIndent=0),
    }


def _table(rows: list[list[str]], col_widths: list[float] | None = None, *, font: float = 7.8,
           align_right_from: int = 1, pad: float = 2.5) -> Table:
    t = Table(rows, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font),
        ("FONT", (0, 1), (-1, -1), "Helvetica", font),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efec")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(BASELINE)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(GRID)),
        ("ALIGN", (align_right_from, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (align_right_from, 0), (-1, 0), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
    ]
    t.setStyle(TableStyle(style))
    return t


def _usd(x: float) -> str:
    return f"{x:+,.2f}"


def _verdict(runs: list[tuple[MarketHistory, dict[str, SimResult]]], share: float) -> str:
    parts: list[str] = []
    for hist, scen in runs:
        a, f, d = scen.get(AS_IS), scen.get(FIXED), scen.get(DEFENDED)
        if f is None or a is None:
            continue
        name = hist.meta.question[:60]
        txt = (
            f"<b>{name}</b>: trading PnL {_usd(f.total_pnl)} fixed vs {_usd(a.total_pnl)} as-is "
            f"over {f.days:.0f} days"
        )
        if d is not None:
            txt += (f", {_usd(d.total_pnl)} with defenses ({len(d.trade_outs)} trade-outs, "
                    f"max inventory ${d.max_inv_value:,.0f} vs ${f.max_inv_value:,.0f})")
        best = d or f
        txt += (
            f". With rewards at a {share:.0%} pool share the best total is "
            f"{_usd(best.total_pnl + best.rewards_usd(share))} vs {_usd(a.total_pnl + a.rewards_usd(share))} "
            f"as-is. Size-weighted 5-minute markout {best.markout_mean_ticks(300):+.1f} ticks; quotes "
            f"scored rewards {best.scored_fraction:.0%} of sampled minutes (as-is {a.scored_fraction:.0%})."
        )
        parts.append(txt)
    return " ".join(parts) if parts else "No completed runs."


def _metric_rows(
    scen: dict[str, SimResult], horizons: tuple[int, ...], shares: tuple[float, ...], share: float,
) -> list[list[str]]:
    names = [n for n in SCENARIOS if n in scen]
    rows: list[list[str]] = [["Metric", *names]]

    def row(label: str, fn) -> None:  # type: ignore[no-untyped-def]
        rows.append([label, *[fn(scen[n]) for n in names]])

    row("Fills (count / shares / USD)",
        lambda r: f"{len(r.fills)} / {r.fill_shares:,.0f} / {r.fill_notional:,.0f}")
    row("Fill rate vs tape (fills per 100 prints)",
        lambda r: f"{100 * len(r.fills) / max(1, r.n_trades):.1f}")
    row("Trading PnL (cash + inventory at mid)", lambda r: _usd(r.total_pnl))
    row("  of which cash", lambda r: _usd(r.final_cash))
    row("  of which inventory value", lambda r: _usd(r.final_inv_value))
    row("Final position YES / NO (shares)", lambda r: f"{r.final_pos_yes:,.0f} / {r.final_pos_no:,.0f}")
    row("Max net inventory (shares) / max inventory USD",
        lambda r: f"{r.max_net_shares:,.0f} / {r.max_inv_value:,.0f}")
    row("Max drawdown (mark-to-mid)", lambda r: f"{r.max_drawdown:,.2f}")
    row("Daily PnL Sharpe (annualized, no rewards)", lambda r: f"{r.sharpe_daily:.2f}")
    for h in horizons:
        row(f"Markout {h // 60}m: size-wtd ticks / total USD / adverse share",
            lambda r, h=h: f"{r.markout_mean_ticks(h):+.2f} / {r.markout_total_usd(h):+.2f} / {r.adverse_fraction(h):.0%}")
    row("Trade-outs (count / shares / taker fees USD)",
        lambda r: f"{len(r.trade_outs)} / {sum(f.size for f in r.trade_outs):,.0f} / {r.fees_usd:,.2f}")
    row("Maker rebates credited (USD, inside PnL)", lambda r: f"{r.rebates_usd:,.2f}")
    row("Orders placed / cancelled", lambda r: f"{r.placements:,} / {r.cancels:,}")
    row("Avg order lifetime (min)", lambda r: f"{r.avg_order_lifetime_s / 60:.1f}")
    row("Minutes scoring rewards (Qmin > 0)",
        lambda r: f"{r.reward_minutes_scored:,} / {r.reward_minutes_total:,} ({r.scored_fraction:.0%})")
    row("Mean Qmin while scoring (shares)",
        lambda r: f"{(r.reward_qmin_sum / r.reward_minutes_scored) if r.reward_minutes_scored else 0:,.0f}")
    for s in shares:
        row(f"Est. rewards at {s:.0%} pool share", lambda r, s=s: _usd(r.rewards_usd(s)))
    row(f"Total PnL incl. rewards @{share:.0%}", lambda r: _usd(r.total_pnl + r.rewards_usd(share)))
    row("Minutes halted (kill switch / caps)", lambda r: f"{r.halt_minutes:,}")
    return rows


def build_report(
    runs: list[tuple[MarketHistory, dict[str, SimResult]]], out: Path, *,
    days: int, config_dir: str, share: float = 0.10,
    sensitivity: dict[str, list[SimResult]] | None = None,
) -> Path:
    st = _styles()
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title="poly-maker backtest", author="polymaker backtest")
    story: list = []
    gen = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    tmpdir = Path(tempfile.mkdtemp(prefix="polymaker-report-"))

    # ── page 1: summary ────────────────────────────────────────────────
    story.append(Paragraph("poly-maker backtest", st["title"]))
    story.append(Paragraph(
        f"Tape replay of the configured strategy over the last {days} days. Generated {gen}. "
        f"Config: {config_dir}/. Three scenarios per market: <b>{AS_IS}</b> replicates the live "
        f"engine's known defects; <b>{FIXED}</b> has them corrected; <b>{DEFENDED}</b> adds the "
        f"inventory-lean floor and the bounded taker trade-out from the configured profile. "
        f"Rewards are an estimate at an assumed {share:.0%} share of each market's daily pool.",
        st["sub"]))
    story.append(Paragraph("Headline", st["h1"]))
    hdr = ["Market", "Scenario", "Fills", "Fill $", "Trading PnL", "Markout 5m", "Max DD",
           "Scored", f"Rew @{share:.0%}", "Total"]
    rows: list[list[str]] = [hdr]
    for hist, scen in runs:
        for name in SCENARIOS:
            r = scen.get(name)
            if r is None:
                continue
            rows.append([
                hist.meta.question[:26], name, str(len(r.fills)), f"{r.fill_notional:,.0f}",
                _usd(r.total_pnl), _usd(r.markout_total_usd(300)), f"{r.max_drawdown:,.2f}",
                f"{r.scored_fraction:.0%}", _usd(r.rewards_usd(share)),
                _usd(r.total_pnl + r.rewards_usd(share)),
            ])
    story.append(_table(rows, [38 * mm, 24 * mm, 9 * mm, 13 * mm, 17 * mm, 17 * mm, 13 * mm,
                               12 * mm, 16 * mm, 16 * mm], font=7.2, align_right_from=2))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Verdict", st["h1"]))
    story.append(Paragraph(_verdict(runs, share), st["body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Reading the table: <b>Trading PnL</b> is cash from fills plus inventory marked at the final "
        "mid, so it includes unrealized inventory risk. <b>Markout 5m</b> sums, over all fills, how "
        "far the mid moved for or against us five minutes after each fill; a negative number is "
        "adverse selection paid. <b>Scored</b> is the share of sampled minutes in which our resting "
        "quotes met the reward program's size and band rules on both sides (Qmin > 0). "
        "<b>Rewards</b> = daily pool x days x scored fraction x assumed share; see the sensitivity "
        "table on each market page.", st["small"]))
    story.append(PageBreak())

    # ── page 2: method ─────────────────────────────────────────────────
    story.append(Paragraph("Method and assumptions", st["h1"]))
    bullets = [
        "<b>Strategy code under test is the repo's own pure core:</b> compute_fair_value, "
        "Vol/Flow/Markout estimators, RegimeMachine, construct_quotes, reconcile, RiskManager over "
        "a StateStore. The engine's I/O is replaced by the replay harness; nothing in strategy/ was "
        "modified for the backtest.",
        "<b>Data:</b> taker trade tape from " + ", ".join(sorted({h.tape_source for h, _ in runs}))
        + " and the CLOB /prices-history minute series for the YES token. Trades on "
        "the NO token are mapped to YES space (taker BUY NO at q = taker SELL YES at 1 - q). "
        + ("The on-chain tape holds every fill since V2 genesis, one print per taker order on the "
           "asset it traded (maker legs of the same match are not double counted)."
           if all(h.tape_source.startswith("poly_data") for h, _ in runs) else
           "The public data-api tape carries only roughly 25-40% of the 24h volume Gamma reports "
           "for these markets (checked on the run date), so the replay sees FEWER prints than the "
           "live book: fill counts and fill-driven PnL here are conservative, reward scoring is not "
           "affected. Pass --polydata to replay the full on-chain tape instead."),
        "<b>Synthetic book:</b> no historical L2 exists, so best bid/ask = minute mid -/+ half the "
        "assumed spread (parameter spread_ticks) with a constant touch depth. Microprice = mid. "
        "Sub-minute book jitter, dust at the touch, and air pockets are therefore NOT modeled; the "
        "live regime machine will fire more often than this replay shows.",
        "<b>Fills (volume-ahead queue):</b> an order that joins or sits behind the touch starts with "
        "touch_depth shares ahead of it; prints at its level drain that queue before we fill; prints "
        "through the level fill us fully. Our presence is assumed not to change the tape (standard "
        "maker-backtest assumption; optimistic on thin markets).",
        "<b>Mint/merge matching is honored:</b> a taker SELL YES fills our BUY YES and our SELL NO "
        "exits; a taker BUY YES fills our SELL YES and our BUY NO. Pairs are marked at 1.00 total "
        "(merge not simulated; capital lock-up is reported, not penalized).",
        "<b>Rewards:</b> each minute we compute Q = S x ((v - s)/v)^2 per in-band order of at least "
        "the reward min size, Qmin = max(min(Qb, Qa/3), min(Qb/3, Qa)) for mids in [0.10, 0.90]. "
        "Competition is unobservable, so earned rewards = pool x scored fraction x an assumed share; "
        "three shares are shown per market.",
        "<b>Fees/rebates:</b> trade-outs pay the market's taker fee (rate x p x (1-p) per share); "
        "maker fills pay nothing and, on fee-enabled markets, are credited rebate_rate x the "
        "taker's fee (0 on fee-free markets such as politics).",
        "<b>Scenario 'engine as-is'</b> replicates: (1) markout recorded in token space against YES "
        "fair value, (2) raw aggressor sign on NO-token prints feeding the flow EWMA, (3) exit "
        "urgency never ramped. <b>'fixed'</b> corrects all three and runs the strategy core as it "
        "was before the defenses (every defense knob off). <b>'fixed + defenses'</b> runs the "
        "configured profile: inventory-lean floor, exits priced off the leaned reservation price, "
        "EVENT keeps exits alive, inventory measured as capital at risk, sub-minimum remnants "
        "replaced, and the bounded taker trade-out.",
        "<b>Toxicity memory:</b> an earlier candidate fix let adverse-selection toxicity age out over "
        "30 minutes. Measured on both markets it re-engaged full size after every bad fill and "
        "roughly quadrupled the trading loss, so the shipped default keeps a long memory (no "
        "decay); the sensitivity table shows a 6-hour half-life for reference.",
        "<b>Risk:</b> the config's [risk] caps and daily-loss kill are active, reset at each UTC day "
        "boundary. Neg-risk group cost is zero (single markets).",
        "<b>Not modeled:</b> WS staleness, heartbeat gaps, post-only rejections, partial batch "
        "failures, data-API lag, gas or relayer costs, and competition reacting to our quotes.",
    ]
    for b in bullets:
        story.append(Paragraph(b, st["bullet"], bulletText="-"))
    story.append(PageBreak())

    # ── per market ─────────────────────────────────────────────────────
    for mi, (hist, scen) in enumerate(runs):
        m = hist.meta
        key = f"m{mi}"
        any_r = next(iter(scen.values()))
        lo = min(p for _, p in any_r.price) if any_r.price else 0
        hi = max(p for _, p in any_r.price) if any_r.price else 0
        vol = sum(t.size * t.price for t in hist.tape)
        story.append(Paragraph(m.question, st["h1"]))
        story.append(Paragraph(
            f"slug {m.slug} - profile <b>{any_r.profile_name}</b> - tick {m.tick_size:g} - "
            f"reward min size {m.rewards_min_size:g} sh, band {m.rewards_max_spread:g}c, pool "
            f"${m.rewards_daily_rate:,.0f}/day - taker fee {m.taker_fee_bps / 100:.1f}%, rebate "
            f"{m.rebate_rate:.0%}", st["small"]))
        story.append(Paragraph(
            f"Window {_dt(any_r.start_ts):%Y-%m-%d} to {_dt(any_r.end_ts):%Y-%m-%d} "
            f"({any_r.days:.1f} days): {any_r.n_trades:,} taker prints, ${vol:,.0f} tape notional, "
            f"{any_r.n_bars:,} mid samples, YES mid range {lo:.3f} to {hi:.3f}.", st["small"]))
        story.append(Spacer(1, 4))
        shown = scen.get(DEFENDED) or scen.get(FIXED) or any_r
        story.append(_img(fig_price_fills(shown, tmpdir, key), _W * 72))
        story.append(_img(fig_equity(scen, tmpdir, key), _W * 72))
        story.append(_img(fig_regime(scen, tmpdir, key), _W * 72))
        story.append(PageBreak())

        story.append(Paragraph(f"{m.question[:70]} - detail", st["h2"]))
        story.append(_img(fig_markout(scen, tmpdir, key), _W * 72))
        story.append(_img(fig_daily(scen, tmpdir, key), _W * 72))
        story.append(_img(fig_toxicity(scen, tmpdir, key), _W * 72))
        story.append(Spacer(1, 4))
        det = _metric_rows(scen, any_r.params.markout_horizons, any_r.params.reward_shares, share)
        ncol = len(det[0]) - 1
        story.append(_table(det, [66 * mm, *([112 * mm / ncol] * ncol)], font=7.0, pad=1.5))
        story.append(PageBreak())

    # ── A/B page ───────────────────────────────────────────────────────
    story.append(Paragraph("Engine defects: measured cost", st["h1"]))
    story.append(Paragraph(
        "Two steps per market: the engine bug fixes (fixed minus as-is) and the defenses on top "
        "(defended minus fixed). Positive PnL/markout = the step helps; negative max-inventory "
        "and drawdown = less risk carried. The replay sees only minute mids and a synthetic book, "
        "so treat magnitudes as directional.",
        st["small"]))
    ab: list[list[str]] = [["Market", "Step", "Trading PnL", "Markout 5m", "Scored min",
                            "Fills", "Max inv $", "Max DD"]]
    for hist, scen in runs:
        a, f, d = scen.get(AS_IS), scen.get(FIXED), scen.get(DEFENDED)
        steps = [("bug fixes (fixed - as-is)", f, a), ("defenses (defended - fixed)", d, f)]
        for label, x, y in steps:
            if x is None or y is None:
                continue
            ab.append([hist.meta.question[:30], label, _usd(x.total_pnl - y.total_pnl),
                       _usd(x.markout_total_usd(300) - y.markout_total_usd(300)),
                       f"{x.reward_minutes_scored - y.reward_minutes_scored:+,}",
                       f"{len(x.fills) - len(y.fills):+}",
                       f"{x.max_inv_value - y.max_inv_value:+,.0f}",
                       f"{x.max_drawdown - y.max_drawdown:+,.2f}"])
    story.append(_table(ab, [46 * mm, 40 * mm, 18 * mm, 18 * mm, 17 * mm, 11 * mm, 15 * mm, 13 * mm],
                        font=7.2, align_right_from=2))
    if sensitivity:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Sensitivity: one knob at a time on top of 'fixed + defenses'", st["h2"]))
        story.append(Paragraph(
            "Each row changes exactly one profile value. Total = trading PnL + rewards at the "
            f"assumed {share:.0%} share. One month of one market is not enough to adopt a tuning "
            "from this table; it shows which levers matter.", st["small"]))
        for hist, scen in runs:
            rows_s = sensitivity.get(hist.meta.condition_id)
            d = scen.get(DEFENDED)
            if not rows_s or d is None:
                continue
            tbl: list[list[str]] = [["Variant", "Fills", "Trading PnL", "Markout 5m", "Scored",
                                     "Max DD", "Trade-outs", f"Total @{share:.0%}"]]
            base_row = ["configured (defended)", str(len(d.fills)), _usd(d.total_pnl),
                        _usd(d.markout_total_usd(300)), f"{d.scored_fraction:.0%}",
                        f"{d.max_drawdown:,.2f}", str(len(d.trade_outs)),
                        _usd(d.total_pnl + d.rewards_usd(share))]
            tbl.append(base_row)
            for v in rows_s:
                tbl.append([v.scenario, str(len(v.fills)), _usd(v.total_pnl),
                            _usd(v.markout_total_usd(300)), f"{v.scored_fraction:.0%}",
                            f"{v.max_drawdown:,.2f}", str(len(v.trade_outs)),
                            _usd(v.total_pnl + v.rewards_usd(share))])
            story.append(Paragraph(hist.meta.question[:70], st["small"]))
            story.append(_table(tbl, [40 * mm, 12 * mm, 22 * mm, 22 * mm, 14 * mm, 16 * mm,
                                      18 * mm, 22 * mm], font=7.2))
            story.append(Spacer(1, 6))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Caveats before acting on this", st["h2"]))
    for b in [
        "A trades-only replay cannot see queue competition, dust at the touch, pulled liquidity, or "
        "gaps. On thin markets the live engine will be picked off more than shown here.",
        "The public tape under-reports flow (25-40% of Gamma's 24h volume on the run date), so "
        "expect more fills live than in this replay, in both directions.",
        "Rewards dominate the economics of both configured markets. The share assumption is the "
        "single largest uncertainty; measure it live by comparing paid rewards to the scored "
        "fraction over a few days.",
        "Window is one month of recent history. Re-run with --days and other markets before "
        "generalizing; the data cache makes re-runs offline.",
        "The trade-out is modeled as an immediate fill at the synthetic best bid for the capped "
        "size. Live, a FAK can fill partially or not at all when the bid has moved; the engine "
        "then waits out the cooldown rather than chasing.",
        "Remaining review findings not addressed by any scenario: the flow z-score bound (flow "
        "TRENDING trigger unreachable), the consecutive-recompute jump detector, own orders "
        "polluting the microprice, and the never-improve-the-touch placement rule.",
    ]:
        story.append(Paragraph(b, st["bullet"], bulletText="-"))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"polymaker backtest - generated {gen}", st["small"]))

    doc.build(story)
    return out
