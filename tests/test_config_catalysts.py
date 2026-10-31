"""markets.toml catalyst calendar: parsing, validation, merge with global windows."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from polymaker.config import CatalystWindow, Config, MarketEntry


def _write_cfg(tmp_path: Path, markets_toml: str) -> Config:
    (tmp_path / "config.toml").write_text("")
    (tmp_path / "strategy.toml").write_text(
        "[profiles.p]\ncatalyst_flatten_hours = 2.0\n"
    )
    (tmp_path / "markets.toml").write_text(markets_toml)
    return Config.load(tmp_path, load_env=False)


def test_per_market_and_global_catalysts_parse_and_merge(tmp_path: Path):
    cfg = _write_cfg(tmp_path, """
[[catalysts]]
start = 2026-11-03T23:00:00Z
end   = 2026-11-04T12:00:00Z
note  = "election night"

[[markets]]
slug    = "m1"
profile = "p"
gamma   = 0.9

  [[markets.catalysts]]
  start = "2026-10-30T18:00:00Z"          # ISO string form also accepted
  end   = "2026-10-30T20:00:00Z"
  note  = "debate"
  flatten_hours = 1.0

  [[markets.catalysts]]
  start = 2026-10-28 14:00:00             # naive -> UTC
  end   = 2026-10-28 15:00:00

[[markets]]
slug    = "m2"
profile = "p"
""")
    m1, m2 = cfg.markets
    assert [c.note for c in m1.catalysts] == ["debate", ""]
    assert m1.catalysts[1].start == datetime(2026, 10, 28, 14, tzinfo=UTC)
    assert m1.catalysts[0].flatten_hours == 1.0 and m1.catalysts[1].flatten_hours is None
    # `catalysts` is a reserved key, never a profile override; real overrides survive
    assert m1.overrides == {"gamma": 0.9}
    assert cfg.profile_for(m1).gamma == 0.9
    assert cfg.profile_for(m1).catalyst_flatten_hours == 2.0
    # global window applies to every market, merged soonest-first
    assert [c.note for c in cfg.catalysts_for(m1)] == ["", "debate", "election night"]
    assert [c.note for c in cfg.catalysts_for(m2)] == ["election night"]
    assert m2.catalysts == []


def test_catalyst_window_validation():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="end must be after start"):
        CatalystWindow(start=t0, end=t0)
    with pytest.raises(ValueError, match="flatten_hours"):
        CatalystWindow(start=t0, end=datetime(2026, 1, 2, tzinfo=UTC), flatten_hours=-1)
    with pytest.raises(ValueError):  # extra="forbid": typos don't pass silently
        CatalystWindow(start=t0, end=datetime(2026, 1, 2, tzinfo=UTC), nite="x")  # type: ignore[call-arg]
    # non-UTC offsets normalise to UTC
    w = CatalystWindow(start="2026-01-01T12:00:00+02:00", end="2026-01-01T13:00:00+02:00")  # type: ignore[arg-type]
    assert w.start == datetime(2026, 1, 1, 10, tzinfo=UTC)
    assert w.end_ts - w.start_ts == 3600.0


def test_market_entry_without_catalysts_is_unchanged():
    e = MarketEntry(slug="x", profile="p", c_vol=2.0)
    assert e.catalysts == [] and e.overrides == {"c_vol": 2.0}


def test_reload_markets_refreshes_catalysts(tmp_path: Path):
    cfg = _write_cfg(tmp_path, '[[markets]]\nslug="m"\nprofile="p"\n')
    assert cfg.catalysts == []
    (tmp_path / "markets.toml").write_text(
        '[[catalysts]]\nstart = 2030-01-01T00:00:00Z\nend = 2030-01-02T00:00:00Z\n'
        '[[markets]]\nslug="m"\nprofile="p"\n'
    )
    cfg.reload_markets()
    assert len(cfg.catalysts) == 1
