"""Download and normalise English Premier League match data.

Source: football-data.co.uk, which publishes one CSV per season containing
full-time results plus match statistics (shots, shots on target, corners,
fouls, cards) and closing bookmaker odds. No API key is required.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import DATA_RAW, FIRST_SEASON, LAST_SEASON

log = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Raw column -> tidy name. Columns missing from a season are filled with NaN.
COLUMN_MAP = {
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}

# Closing odds. Pinnacle (PSC*) is sharpest; Bet365 (B365C*) and the market
# average (AvgC*) are fallbacks, in that order of preference.
ODDS_CANDIDATES = [
    ("PSCH", "PSCD", "PSCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
    ("AvgH", "AvgD", "AvgA"),
    ("BbAvH", "BbAvD", "BbAvA"),
]

# Bookmakers that publish BOTH an early price and a closing price, best first.
# Pinnacle is the sharpest and has the longest history (2012-13 on); Bet365
# and the market average cover 2019-20 on, including the current season.
OPEN_CLOSE_PAIRS = [
    (("PSH", "PSD", "PSA"), ("PSCH", "PSCD", "PSCA")),
    (("B365H", "B365D", "B365A"), ("B365CH", "B365CD", "B365CA")),
    (("AvgH", "AvgD", "AvgA"), ("AvgCH", "AvgCD", "AvgCA")),
]

# Best price available across bookmakers ("line shopping"). Published as
# Max*/MaxC* from 2019-20; the older BbMx* series is deliberately not mixed in
# because it is a pre-closing maximum and would change what the column means
# halfway through the history.
BEST_PRICE_COLS = {
    "best_close": ("MaxCH", "MaxCD", "MaxCA"),
    "best_open": ("MaxH", "MaxD", "MaxA"),
}

# Team naming drifts a little across seasons and in the fixtures feed.
TEAM_ALIASES = {
    "Middlesboro": "Middlesbrough",
    "Sheffield Weds": "Sheffield Weds",
    "Nott'm Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Man United": "Man United",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Wolverhampton": "Wolves",
    "Tottenham Hotspur": "Tottenham",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Hull City": "Hull",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Cardiff City": "Cardiff",
    "Norwich City": "Norwich",
    "Birmingham City": "Birmingham",
    "Blackburn Rovers": "Blackburn",
    "Bolton Wanderers": "Bolton",
    "Charlton Athletic": "Charlton",
    "Ipswich Town": "Ipswich",
    "Luton Town": "Luton",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Brighton & Hove Albion": "Brighton",
    "AFC Bournemouth": "Bournemouth",
    "Sheffield United": "Sheffield United",
}


def season_code(start_year: int) -> str:
    """2024 -> '2425' (the code football-data.co.uk uses in its URLs)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2024 -> '2024-25'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def download_season(start_year: int, *, refresh: bool = False) -> Path:
    """Fetch one season CSV into data/raw, returning its path.

    The current season is always re-fetched (``refresh`` or not) because new
    results are appended to it week by week.
    """
    path = DATA_RAW / f"E0_{season_code(start_year)}.csv"
    is_current = start_year >= LAST_SEASON
    if path.exists() and not refresh and not is_current:
        return path

    url = BASE_URL.format(code=season_code(start_year))
    log.info("downloading %s", url)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        # The source goes down from time to time. A scheduled rebuild should
        # fall back to the last good copy rather than fail outright; only a
        # season we have never fetched is a genuine error.
        if path.exists():
            log.warning("%s unavailable (%s); using the cached copy", url, exc)
            return path
        raise
    path.write_bytes(resp.content)
    return path


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a football-data CSV that may be slightly malformed.

    Two quirks show up in the archive: some seasons are latin-1 rather than
    utf-8, and some rows carry extra trailing commas (2003-04 has 45 such
    rows, which a strict parser would silently drop). Rows are padded or
    truncated to the header width instead.
    """
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with path.open(encoding=encoding, newline="") as fh:
                rows = list(csv.reader(fh))
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 decodes any byte string
        raise ValueError(f"could not decode {path.name}")

    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        return pd.DataFrame()
    header = [c.strip() for c in rows[0]]
    width = len(header)
    body = [(r + [""] * width)[:width] for r in rows[1:]]
    df = pd.DataFrame(body, columns=header)
    return df.replace("", pd.NA)


def _pick_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Add odds_h/odds_d/odds_a from the best bookmaker columns available."""
    out = pd.DataFrame(index=df.index, columns=["odds_h", "odds_d", "odds_a"], dtype="float64")
    for cols in ODDS_CANDIDATES:
        if not all(c in df.columns for c in cols):
            continue
        vals = df[list(cols)].apply(pd.to_numeric, errors="coerce")
        vals.columns = ["odds_h", "odds_d", "odds_a"]
        out = out.where(out.notna().all(axis=1), vals)
    return out


def _pick_open_close(df: pd.DataFrame) -> pd.DataFrame:
    """Opening and closing prices from the best bookmaker that has both.

    Both sides of the pair must be present on the same row, otherwise the
    movement between them would be comparing two different bookmakers.
    """
    names = ["open_h", "open_d", "open_a", "close_h", "close_d", "close_a"]
    out = pd.DataFrame(index=df.index, columns=names, dtype="float64")
    for open_cols, close_cols in OPEN_CLOSE_PAIRS:
        cols = list(open_cols) + list(close_cols)
        if not all(c in df.columns for c in cols):
            continue
        vals = df[cols].apply(pd.to_numeric, errors="coerce")
        vals.columns = names
        complete = vals.notna().all(axis=1) & (vals > 1.0).all(axis=1)
        take = out.isna().all(axis=1) & complete
        out.loc[take] = vals.loc[take]
    return out


def _pick_best_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Best available price across bookmakers, opening and closing."""
    out = pd.DataFrame(index=df.index)
    for prefix, cols in BEST_PRICE_COLS.items():
        names = [f"{prefix}_{k}" for k in ("h", "d", "a")]
        if all(c in df.columns for c in cols):
            vals = df[list(cols)].apply(pd.to_numeric, errors="coerce")
            vals.columns = names
            # A "best price" below evens on every outcome is a data error.
            vals = vals.where(vals > 1.0)
            out[names] = vals
        else:
            out[names] = np.nan
    return out


def _parse_dates(raw: pd.Series) -> pd.Series:
    """football-data uses dd/mm/yy in old seasons and dd/mm/yyyy in new ones."""
    s = raw.astype("string").str.strip()
    four = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    two = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return four.fillna(two)


def load_season(start_year: int, *, refresh: bool = False) -> pd.DataFrame:
    """Load one season as a tidy frame, one row per match."""
    path = download_season(start_year, refresh=refresh)
    raw = _read_csv_robust(path)
    if "HomeTeam" not in raw.columns:
        raise ValueError(f"unexpected layout in {path.name}")
    raw = raw[raw["HomeTeam"].notna() & raw["AwayTeam"].notna()]

    df = pd.DataFrame(index=raw.index)
    for src, dst in COLUMN_MAP.items():
        df[dst] = raw[src] if src in raw.columns else pd.NA

    df["date"] = _parse_dates(df["date"])
    numeric = [c for c in df.columns if c not in ("date", "time", "home_team", "away_team", "result")]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    df = pd.concat(
        [df, _pick_odds(raw), _pick_open_close(raw), _pick_best_prices(raw)], axis=1
    )

    for col in ("home_team", "away_team"):
        df[col] = df[col].astype("string").str.strip().replace(TEAM_ALIASES)

    df["season"] = season_label(start_year)
    df["season_start"] = start_year
    # Matches with no result yet (current season rows published early) are dropped.
    df = df[df["result"].isin(["H", "D", "A"]) & df["date"].notna()]
    return df.sort_values(["date", "home_team"]).reset_index(drop=True)


def load_matches(
    first_season: int = FIRST_SEASON,
    last_season: int = LAST_SEASON,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load every season in [first_season, last_season] into one frame."""
    frames = []
    for year in range(first_season, last_season + 1):
        try:
            frames.append(load_season(year, refresh=refresh))
        except Exception as exc:  # a season may be missing or malformed
            log.warning("skipping %s: %s", season_label(year), exc)
    if not frames:
        raise RuntimeError("no seasons could be loaded")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)
    df["match_id"] = df.index
    return df


def load_fixtures(*, refresh: bool = True) -> pd.DataFrame:
    """Fetch upcoming fixtures (all leagues) and keep the Premier League ones."""
    path = DATA_RAW / "fixtures.csv"
    if refresh or not path.exists():
        try:
            resp = requests.get(FIXTURES_URL, timeout=60)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        except requests.RequestException as exc:
            if not path.exists():
                log.warning("fixtures unavailable (%s) and nothing cached", exc)
                return pd.DataFrame(columns=["date", "home_team", "away_team"])
            log.warning("fixtures unavailable (%s); using the cached copy", exc)
    raw = _read_csv_robust(path)
    raw = raw[raw.get("Div", pd.Series(dtype=object)) == "E0"]
    if raw.empty:
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    df = pd.DataFrame(
        {
            "date": _parse_dates(raw["Date"]),
            "time": raw["Time"] if "Time" in raw.columns else pd.NA,
            "home_team": raw["HomeTeam"].astype("string").str.strip().replace(TEAM_ALIASES),
            "away_team": raw["AwayTeam"].astype("string").str.strip().replace(TEAM_ALIASES),
        }
    )
    df = pd.concat([df, _pick_odds(raw)], axis=1)
    return df[df["date"].notna()].sort_values("date").reset_index(drop=True)
