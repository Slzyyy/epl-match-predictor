"""Expected-goals data from Understat.

football-data.co.uk only began publishing xG in 2026-27, which is far too
little history to train on. Understat covers the Premier League from 2014-15
and exposes the same data the league pages render, one JSON document per
season, at ``/getLeagueData/EPL/<year>``.

Each season gives, per match: expected goals and non-penalty expected goals for
both sides, PPDA (passes allowed per defensive action, a pressing-intensity
measure) and "deep" completions (passes completed within ~20 yards of goal).
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
import requests

from .config import DATA_RAW
from .data import TEAM_ALIASES, season_label

log = logging.getLogger(__name__)

LEAGUE_URL = "https://understat.com/getLeagueData/EPL/{year}"
PAGE_URL = "https://understat.com/league/EPL/{year}"

# Understat's xG history starts with the 2014-15 season.
FIRST_XG_SEASON = 2014

# Understat spells out two names that football-data abbreviates; everything
# else is already covered by data.TEAM_ALIASES.
UNDERSTAT_ALIASES = {
    **TEAM_ALIASES,
    "Queens Park Rangers": "QPR",
    "Wolverhampton Wanderers": "Wolves",
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "X-Requested-With": "XMLHttpRequest",
}


def download_season_xg(year: int, *, refresh: bool = False, pause: float = 0.3) -> dict:
    """Fetch one season of Understat data, caching the raw JSON."""
    path = DATA_RAW / f"understat_EPL_{year}.json"
    is_current = year >= FIRST_XG_SEASON and not path.exists()
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    log.info("downloading Understat EPL %s", year)
    try:
        resp = requests.get(
            LEAGUE_URL.format(year=year),
            headers={**_HEADERS, "Referer": PAGE_URL.format(year=year)},
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        if path.exists():
            log.warning("Understat %s unavailable (%s); using the cached copy", year, exc)
            return json.loads(path.read_text(encoding="utf-8"))
        raise
    path.write_text(json.dumps(payload), encoding="utf-8")
    del is_current
    time.sleep(pause)  # be gentle with a free data source
    return payload


def _team_history(payload: dict) -> pd.DataFrame:
    """Per team-match rows carrying the stats that `dates` leaves out."""
    rows = []
    for team in payload.get("teams", {}).values():
        title = team["title"]
        for h in team.get("history", []):
            ppda = h.get("ppda") or {}
            ppda_allowed = h.get("ppda_allowed") or {}
            rows.append(
                {
                    "team": title,
                    "datetime": h["date"],
                    "npxg": float(h["npxG"]),
                    "npxga": float(h["npxGA"]),
                    # PPDA: opponent passes divided by our defensive actions.
                    # Low means we press hard. Undefined when a side made no
                    # defensive actions at all, which is why this is guarded.
                    "ppda": (ppda["att"] / ppda["def"]) if ppda.get("def") else None,
                    "ppda_allowed": (
                        (ppda_allowed["att"] / ppda_allowed["def"])
                        if ppda_allowed.get("def")
                        else None
                    ),
                    "deep": float(h.get("deep", 0)),
                    "deep_allowed": float(h.get("deep_allowed", 0)),
                }
            )
    return pd.DataFrame(rows)


def load_season_xg(year: int, *, refresh: bool = False) -> pd.DataFrame:
    """One row per played match, with both sides' xG and pressing stats."""
    payload = download_season_xg(year, refresh=refresh)
    played = [m for m in payload.get("dates", []) if m.get("isResult")]
    if not played:
        return pd.DataFrame()

    base = pd.DataFrame(
        {
            "datetime": [m["datetime"] for m in played],
            "home_team": [m["h"]["title"] for m in played],
            "away_team": [m["a"]["title"] for m in played],
            "home_xg": [float(m["xG"]["h"]) for m in played],
            "away_xg": [float(m["xG"]["a"]) for m in played],
            "us_home_goals": [int(m["goals"]["h"]) for m in played],
            "us_away_goals": [int(m["goals"]["a"]) for m in played],
        }
    )

    hist = _team_history(payload)
    for side in ("home", "away"):
        part = hist.rename(
            columns={
                "team": f"{side}_team",
                "npxg": f"{side}_npxg",
                "npxga": f"{side}_npxga_conceded",
                "ppda": f"{side}_ppda",
                "ppda_allowed": f"{side}_ppda_allowed",
                "deep": f"{side}_deep",
                "deep_allowed": f"{side}_deep_allowed",
            }
        )
        base = base.merge(part, on=["datetime", f"{side}_team"], how="left")

    base["date"] = pd.to_datetime(base["datetime"]).dt.normalize()
    base["season_start"] = year
    base["season"] = season_label(year)
    for col in ("home_team", "away_team"):
        base[col] = base[col].astype("string").str.strip().replace(UNDERSTAT_ALIASES)
    return base.drop(columns=["datetime"]).sort_values("date").reset_index(drop=True)


def load_xg(
    first_season: int = FIRST_XG_SEASON, last_season: int = 2026, *, refresh: bool = False
) -> pd.DataFrame:
    """Every available season of Understat xG, concatenated."""
    frames = []
    for year in range(max(first_season, FIRST_XG_SEASON), last_season + 1):
        try:
            part = load_season_xg(year, refresh=refresh)
        except Exception as exc:
            log.warning("skipping Understat %s: %s", year, exc)
            continue
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


XG_MATCH_COLS = [
    "home_xg", "away_xg", "home_npxg", "away_npxg",
    "home_ppda", "away_ppda", "home_deep", "away_deep",
]


def attach_xg(matches: pd.DataFrame, xg: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join xG onto the match table on season and the two team names.

    The team pair is unique within a season (each meets once per venue), so no
    date comparison is needed — which avoids spurious misses when a fixture is
    rescheduled and the two sources disagree about the date.
    """
    if xg is None:
        xg = load_xg()
    out = matches.copy()
    if xg.empty:
        for col in XG_MATCH_COLS:
            out[col] = pd.NA
        return out

    key = ["season_start", "home_team", "away_team"]
    dupes = xg.duplicated(key).sum()
    if dupes:
        log.warning("dropping %d duplicate xG rows", dupes)
        xg = xg.drop_duplicates(key)

    merged = out.merge(xg[key + XG_MATCH_COLS], on=key, how="left", validate="many_to_one")

    in_range = merged["season_start"] >= FIRST_XG_SEASON
    missing = int(merged.loc[in_range, "home_xg"].isna().sum())
    if missing:
        log.warning("%d matches in the xG era have no Understat row", missing)
    return merged


def load_season_fixtures(year: int, *, refresh: bool = False) -> pd.DataFrame:
    """Fixtures in a season that have not been played yet.

    The same Understat payload that supplies xG also carries the full 380-match
    calendar, flagged with ``isResult``. The unplayed half is the rest of the
    season's fixture list, which is why no separate fixtures feed is needed:
    football-data's publishes only the next few days and goes empty between
    rounds, whereas this is complete from August to May.
    """
    payload = download_season_xg(year, refresh=refresh)
    upcoming = [m for m in payload.get("dates", []) if not m.get("isResult")]
    if not upcoming:
        return pd.DataFrame(columns=["date", "time", "home_team", "away_team"])

    when = pd.to_datetime([m["datetime"] for m in upcoming], errors="coerce")
    df = pd.DataFrame(
        {
            "date": when.normalize(),
            "time": when.strftime("%H:%M"),
            "home_team": [m["h"]["title"] for m in upcoming],
            "away_team": [m["a"]["title"] for m in upcoming],
            "season_start": year,
        }
    )
    for col in ("home_team", "away_team"):
        df[col] = df[col].astype("string").str.strip().replace(UNDERSTAT_ALIASES)
    return df[df["date"].notna()].sort_values("date").reset_index(drop=True)


def load_upcoming_fixtures(
    first_season: int = FIRST_XG_SEASON, last_season: int = 2026, *, refresh: bool = True
) -> pd.DataFrame:
    """Unplayed fixtures across the seasons available, newest season last."""
    frames = []
    for year in range(max(first_season, FIRST_XG_SEASON), last_season + 1):
        try:
            part = load_season_fixtures(year, refresh=refresh)
        except Exception as exc:
            log.warning("no Understat fixtures for %s: %s", year, exc)
            continue
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["date", "time", "home_team", "away_team"])
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def top_up_from_understat(matches: pd.DataFrame, xg: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add results Understat has but football-data does not yet.

    football-data is the primary source, but it goes down and it lags. Understat
    carries the score for every played match, so any fixture it has settled that
    is missing here is appended with its goals and result. The columns
    football-data would have supplied (shots, corners, cards, odds) stay NaN,
    which the trees handle natively.

    Without this, a football-data outage silently freezes the form features
    behind every prediction, which is worse than a few rows with missing
    columns.
    """
    if xg is None:
        xg = load_xg()
    if xg.empty:
        return matches

    have = set(map(tuple, matches[["season_start", "home_team", "away_team"]]
                   .astype({"season_start": "int64"}).astype(str).to_numpy()))
    cand = xg[xg["us_home_goals"].notna() & xg["us_away_goals"].notna()].copy()
    keys = cand[["season_start", "home_team", "away_team"]].astype(
        {"season_start": "int64"}).astype(str).to_numpy()
    missing = cand[[tuple(k) not in have for k in keys]]
    if missing.empty:
        return matches

    log.info("topping up %d match(es) from Understat", len(missing))
    hg = missing["us_home_goals"].to_numpy(dtype=float)
    ag = missing["us_away_goals"].to_numpy(dtype=float)
    extra = pd.DataFrame(
        {
            "date": missing["date"].to_numpy(),
            "season": missing["season"].to_numpy(),
            "season_start": missing["season_start"].to_numpy(),
            "home_team": missing["home_team"].to_numpy(),
            "away_team": missing["away_team"].to_numpy(),
            "home_goals": hg,
            "away_goals": ag,
            "result": np.where(hg > ag, "H", np.where(hg == ag, "D", "A")),
        }
    )
    # reindex (rather than assigning pd.NA per column) so the columns we do not
    # have arrive as float NaN and keep the numeric dtypes of the frame we are
    # concatenating onto.
    extra = extra.reindex(columns=matches.columns)
    out = pd.concat([matches, extra], ignore_index=True)
    out = out.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)
    # match_id is positional, so it has to be reissued after inserting rows.
    out["match_id"] = out.index
    return out
