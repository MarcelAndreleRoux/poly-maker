# poly-maker

A maker-only market-making bot for **Polymarket CLOB V2**, focused on political
markets. Single async process, local-file config, typed and
tested.

> [!WARNING]
> Market making on Polymarket is competitive and can lose money. This is a
> reference implementation and a research harness, not a guaranteed-profitable
> product. Test in `--paper` mode first; go live with small size.

## What it does

- Discovers political markets via the **Gamma API** (seconds) and ranks them by
  reward + rebate income vs. volatility/spread risk.
- Maintains a live order book per token from the **market WebSocket**.
- Quotes **maker-only** — every order is post-only. Fair-value + inventory-skew
  strategy that posts BUY-YES and BUY-NO as a two-sided quote, with live
  volatility/toxicity estimation and a regime machine that pulls quotes during
  news events (see [Strategy](#strategy)).
- Reconciles a target quote set against live orders with churn tolerances; runs
  the exchange **heartbeat** dead-man switch; enforces risk caps and a daily-loss
  kill switch.
- Config, market selection, and state are **local files + SQLite**. An operator
  with the repo, a `.env`, and a funded wallet is a complete deployment.

## Install

Uses [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync --extra dev          # install deps + dev tools
uv run polymaker --help
```

## Configure

```bash
cp .env.example .env         # then edit two values:
```

- `PK` — the private key of your signer wallet
- `BROWSER_ADDRESS` — your Polymarket address (shown on the profile / developer page)

Everything else is TOML under [`config/`](config/):

- `config.toml` — wallet/engine/risk/execution settings
- `strategy.toml` — named parameter profiles (`political-longdated`, `political-hot`)
- `markets.toml` — the trade list (populated via the CLI below)

## Use

```bash
# 1. discover + rank the markets under a Gamma tag (politics by default; writes to state.db).
#    Augmented neg-risk "Other" placeholders are skipped (Polymarket says never trade them).
uv run polymaker scan                  # --tag sports / crypto for the fee-enabled categories
uv run polymaker markets

# 2. add markets to the trade list
uv run polymaker markets-add <slug> --profile political-longdated

# 3. backtest the configured markets on public history -> PDF report
uv run polymaker backtest --days 30 --out reports/backtest.pdf
#    (or any market: -m <slug>=<profile>; data is cached under data/backtest/)
#    The data-api tape stops at the newest 10k prints. For the full on-chain
#    fill history (every taker print since V2 genesis, 2026-04-28) point it at
#    poly_data's dump (github.com/warproxxx/poly_data, run alongside):
uv run polymaker backtest --days 180 --polydata ../poly_data/data/orderFilled.csv

# 4. dry run: full pipeline against the live feed, no orders posted
uv run polymaker run --paper

# 5. preflight the wallet before going live
uv run polymaker doctor

# 6. self-tests: a deep post-only order (free), then a real fill round-trip (~cents)
uv run polymaker livetest      # place a deep post-only order + cancel (no fill)
uv run polymaker moneydoctor   # limit rest + market buy + market sell, auto-flattens

# 7. go live
uv run polymaker run

# ops
uv run polymaker status        # positions / open orders
uv run polymaker cancel-all    # panic button
```

## Docker

Runs the engine as a background container with every dependency pinned inside the
image. State (`state.db`, `journal/`, `logs/`) lives on the host under `data/`,
config is bind-mounted, secrets come from `.env`.

```bash
cp .env.example .env                        # PK + BROWSER_ADDRESS
docker compose --profile cli build
docker compose run --rm cli scan            # any one-off CLI command
docker compose run --rm cli doctor
docker compose --profile paper up -d        # paper engine, no orders posted
docker compose --profile live  up -d        # LIVE engine (real money)
docker compose logs -f live
docker compose --profile live down          # clean stop: cancels all resting orders
```

`data/live` is shared by the `live` engine and `cli`; `data/paper` is isolated.
The container is stopped with SIGINT so `engine.shutdown()` runs `cancel_all`
before exit; the exchange heartbeat still covers a hard crash. Run exactly one
live engine per wallet.

## Architecture

```
market WS ─▶ OrderBook ─▶ (wake) ─▶ Quoter ─▶ strategy (pure) ─▶ reconcile ─▶ ExecutionGateway
user WS   ─▶ StateStore                                         RiskManager ┘   (post-only, heartbeat)
Gamma     ─▶ Catalog/scanner ─▶ SQLite            periodic REST reconcile ┘
```

One async event loop. The strategy layer is a pure function `(book, inventory,
params, clock) → TargetQuotes` — deterministic and unit-tested. The engine owns
all I/O and state around it; the `ExecutionGateway` wraps `py-clob-client-v2`
(which handles the V2 EIP-712 signing) and offloads its blocking calls to a
thread pool so the hot path never stalls. State (positions, orders, PnL, catalog)
lives in one SQLite file; raw WS/order events are journaled to `journal/` for
replay.

## Strategy

Maker-only, quoting both sides of each market as USDC-collateralized bids:

- **Fair value** — depth-weighted microprice off the live book, nudged by an
  EWMA of signed trade flow.
- **Quote construction** — reservation price `r = FV − skew(inventory)`;
  half-spread `δ = base + c_vol·σ + c_tox·toxicity`. Post **BUY-YES at `r − δ`**
  and **BUY-NO at `(1 − r) − δ`**. Because both legs are bids that sum below 1,
  a filled pair merges back to USDC at locked edge `1 − p − q` — a maker-only
  exit that never crosses the spread.
- **Inventory skew** — net position leans both quotes: long YES → bid YES lower,
  bid NO higher (acquire the offsetting leg). Size tapers as inventory approaches
  a soft cap, then the adding side is pulled entirely. The lean has a floor in
  ticks (`skew_min_ticks`) so it survives a dead-quiet tape, and inventory is
  measured as **capital at risk** (YES at fv, NO at 1 − fv), not YES-notional.
- **Exits** — SELL limits on held inventory priced off the *leaned* reservation
  price, walked toward the touch as hold time approaches `exit_urgency_s`. They
  keep working through an EVENT (only entries are pulled). As a last resort a
  **bounded taker trade-out** fires when a position is under water by
  `trade_out_drift_ticks` and the passive exit has had its chance: a FAK sell
  into the near bids only (`trade_out_depth_frac` of the best-bid size, never
  worse than `trade_out_max_slip_ticks`), then a cooldown. It is the one
  deliberate exception to maker-only; `trade_out_enabled = false` restores it.
- **Volatility / toxicity** — realized-vol and per-fill markout (adverse
  selection) EWMAs widen the spread and shrink size in markets that pick us off.
  Toxicity keeps a long memory by default (`tox_decay_halflife_s = 0`): letting
  it age out over 30 minutes re-engaged full size after every bad fill and
  roughly quadrupled backtested losses.
- **Regime machine** — per market: `QUIET` (farm rewards in-band), `TRENDING`
  (lean + widen + half size), `EVENT` (sweep/jump detected → pull quotes, cool
  off), `REDUCE_ONLY` (inventory cap / near end date → exits only), `HALTED`
  (stale data / resolved / kill switch → cancel all).
- **Catalyst calendar** — the jump/sweep detector is reactive: the first print
  of a news event lands before it fires. For *known* catalysts (debates, court
  rulings, data releases, election night) list `[[markets.catalysts]]` windows
  per market, or global `[[catalysts]]`, in `markets.toml`. Inside a window the
  market is `HALTED` (every order pulled; `catalyst_keeps_exits = true` makes it
  `EVENT` instead, exits keep working). For `catalyst_flatten_hours` before the
  window it is `REDUCE_ONLY`, so inventory is worked down before the news. The
  quoter wakes exactly on each boundary.
- **Rewards + rebates** — quotes stay inside the liquidity-rewards band in QUIET;
  the market selector also scores the new maker-rebate program (a share of taker
  fees rebated to makers), and the backtester credits that rebate on every
  simulated maker fill (`rebates=` in the console summary; 0 on fee-free markets).
- **Risk** — per-market notional cap, neg-risk event-group worst-case cap, total
  exposure cap, daily-loss kill switch, WS-staleness halt, and a fair-value band
  (`fv_band_min` / `fv_band_max`, default 0.10–0.90): outside it the market is
  `REDUCE_ONLY`, so no new entries near 0 / 1 while exits keep working.
- **Collateral-aware** — the engine tracks the funder's pUSD (read at startup and
  every reconcile, adjusted on fills; in paper mode the exposure cap stands in)
  and never sends a BUY it cannot fund once every resting BUY is counted: the
  exchange would reject it and one rejection quarantines the market. The
  backtester applies the same rule with `--bankroll N`. When a USDC budget would
  leave the expensive side under the reward floor, size entries in shares with
  `base_size_shares` (the `fifty` profile: 50 YES + 50 NO shares < 50 USD).

Tune it all via profiles in `config/strategy.toml`.

## Develop

```bash
uv sync --extra dev --extra report   # `report` = matplotlib + reportlab for the backtest PDF
uv run pytest                 # unit suite (offline)
POLYMAKER_LIVE=1 uv run pytest tests/test_live_marketdata.py   # live WS integration
uv run ruff check src tests   # lint
uv run mypy src               # types (strict)
```

## Status

Implemented and live-verified end to end (auth → book → strategy → sign → post →
cancel): config, catalog/scanner, order book + analytics, strategy (FV,
vol/toxicity, regime, quoting), state store + lifecycle, execution gateway +
reconciler + heartbeat, market/user websockets, risk manager, merger (EOA path),
engine, CLI, paper mode, journal capture. 83 tests; ruff + mypy strict clean.

A tape-replay backtester (`polymaker backtest`) replays the public taker trade tape
and minute price history through the pure strategy core with a synthetic book and
a volume-ahead queue model, and writes a PDF report (`src/polymaker/backtest/`).
It runs three scenarios per market (engine "as-is" with its pre-fix defects, "fixed",
and "fixed + defenses" = the configured profile) plus a one-knob sensitivity table.
Every defense is a profile knob so it can be switched off and measured. With
`--polydata` the tape comes from poly_data's on-chain `orderFilled.csv` instead of the
capped data-api (one print per taker order; per-market caches are incremental, so a
re-run only reads what the collector appended). The dump must cover both V2 exchange
contracts: upstream poly_data only watched the CTF Exchange, which drops every neg-risk
(multi-outcome) market; the local poly_data copy also watches the Neg-Risk CTF Exchange.
Not yet built: a replay over captured L2 journals, and external data feeds (polls /
news / cross-venue). Merging through a Safe/proxy wallet routes a
tx via the relayer and isn't wired yet — until then inventory exits via limit
sells rather than merging.

## License

MIT
