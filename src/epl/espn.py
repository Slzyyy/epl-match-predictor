"""Results from ESPN's public scoreboard.

A third results source, independent of the other two. football-data.co.uk and
Understat have both been unreachable at the same time, and when that happens
the table freezes at whatever the last good build saw: football-data is the
primary feed and Understat is what tops it up, so losing both leaves nothing
to move the season on. ESPN answered throughout that outage, so it is asked
last, for whatever the other two are still missing.

It carries the score and nothing else -- no xG, no shots, no odds -- so it is
strictly a way of keeping the results current, not a replacement for either
of the others.

The scoreboard takes one calendar day per request (a ``dates`` range is
rejected), so a sweep costs one request per day. Days whose matches have all
finished can never change, so they are kept and never asked for again; only
the tail of the window is refetched.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time

import numpy as np
import pandas as pd
import requests

from .config import DATA_RAW
from .data import TEAM_ALIASES, season_label

log = logging.getLogger(__name__)

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"

# ESPN spells out two names that neither football-data nor TEAM_ALIASES covers.
# Every other name it uses is either a key there already or is spelt the way
# this project spells it, in which case the replace below is a no-op.
ESPN_ALIASES = {
    **TEAM_ALIASES,
    "Coventry City": "Coventry",
    "Wolverhampton Wanderers": "Wolves",
}

# A day this close to now may still be in play, so it is always refetched even
# when the stored copy says every match on it had finished.
UNSETTLED_DAYS = 2


def _today() -> dt.date:
    """Indirected so a test can fix the day the sweep thinks it is."""
    return dt.date.today()


def _cache_path(year: int):
    return DATA_RAW / f"espn_EPL_{year}.json"


def _match(event: dict) -> dict | None:
    """One completed match, or None if the event is not one.

    Anything unexpected in the shape is treated as "not a completed match"
    rather than raised: this feed is the fallback, so a single odd event
    should not cost us the rest of the day's results.
    """
    try:
        comp = event["competitions"][0]
        if not comp["status"]["type"].get("completed"):
            return None
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        home, away = sides["home"], sides["away"]
        hg, ag = int(home["score"]), int(away["score"])
        return {
            "date": str(event["date"])[:10],
            "home_team": str(home["team"]["displayName"]),
            "away_team": str(away["team"]["displayName"]),
            "home_goals": hg,
            "away_goals": ag,
        }
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _usable_day(payload: object) -> bool:
    """Whether a response is the scoreboard rather than something else.

    A day with no fixtures is a legitimate empty answer, so the test is on the
    envelope, not on the number of events. Without this an error page that
    happens to be JSON would be cached as "no matches that day" and, because
    settled days are never re-asked, would stay that way.
    """
    return isinstance(payload, dict) and isinstance(payload.get("events"), list)


def _load_cache(year: int) -> dict:
    path = _cache_path(year)
    if not path.exists():
        return {"days": {}}
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        log.warning("ESPN cache for %d is not readable; starting it again", year)
        return {"days": {}}
    days = stored.get("days")
    if not isinstance(days, dict):
        return {"days": {}}
    return {"days": {k: v for k, v in days.items() if isinstance(v, dict)}}


def _save_cache(year: int, cache: dict) -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    _cache_path(year).write_text(json.dumps(cache, indent=1), encoding="utf-8")


def fetch_day(day: dt.date, *, timeout: float = 20.0) -> list[dict]:
    """Every completed match ESPN lists for one calendar day."""
    resp = requests.get(
        SCOREBOARD,
        params={"dates": day.strftime("%Y%m%d")},
        headers={"Accept": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not _usable_day(payload):
        raise ValueError("response is not the scoreboard")
    return [m for m in (_match(e) for e in payload["events"]) if m]


def download_season_results(
    year: int,
    *,
    start: dt.date | None = None,
    end: dt.date | None = None,
    budget: int = 60,
    max_seconds: float = 90.0,
    pause: float = 0.2,
    refresh: bool = False,
) -> dict[str, list[dict]]:
    """Sweep a window of days, returning day -> completed matches.

    Bounded three ways on purpose, because this runs inside a scheduled build:
    a request budget, a wall-clock deadline, and an abort after a run of
    failures. A sweep that runs out of any of them keeps what it has rather
    than failing the build -- partial results still move the table forward.
    """
    today = _today()
    start = start or dt.date(year, 8, 1)
    end = min(end or today, today)
    cache = _load_cache(year)
    days = cache["days"]
    if refresh:
        days = cache["days"] = {}

    deadline = time.monotonic() + max_seconds
    spent = failures = 0
    cutoff = today - dt.timedelta(days=UNSETTLED_DAYS)

    day = start
    while day <= end:
        key = day.strftime("%Y%m%d")
        stored = days.get(key)
        settled = bool(stored and stored.get("final"))
        if settled and not refresh:
            day += dt.timedelta(days=1)
            continue
        if spent >= budget or time.monotonic() > deadline or failures >= 5:
            log.warning(
                "ESPN sweep stopped early at %s (%d request(s), %d failure(s))",
                key, spent, failures,
            )
            break
        try:
            matches = fetch_day(day)
            spent += 1
            failures = 0
        except (requests.RequestException, ValueError) as exc:
            spent += 1
            failures += 1
            log.warning("ESPN %s unavailable (%s)", key, exc)
            day += dt.timedelta(days=1)
            continue
        # A day is only final once it is behind us: asked about today, ESPN
        # answers with fixtures that have not kicked off yet.
        days[key] = {"final": day < cutoff, "matches": matches}
        if pause:
            time.sleep(pause)
        day += dt.timedelta(days=1)

    if spent:
        _save_cache(year, cache)
    return {k: v.get("matches", []) for k, v in days.items()}


def espn_results(year: int, **kwargs) -> pd.DataFrame:
    """Completed results for one season as a frame, newest sweep included."""
    days = download_season_results(year, **kwargs)
    rows = [m for day in sorted(days) for m in days[day]]
    cols = ["date", "season", "season_start", "home_team", "away_team",
            "home_goals", "away_goals", "result"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("home_team", "away_team"):
        df[col] = df[col].astype("string").str.strip().replace(ESPN_ALIASES)
    df["season"] = season_label(year)
    df["season_start"] = year
    # float, not int: these rows are concatenated onto a frame whose goal
    # columns carry NaN for unplayed matches, and an int column cannot.
    hg = df["home_goals"].to_numpy(dtype=float)
    ag = df["away_goals"].to_numpy(dtype=float)
    df["home_goals"], df["away_goals"] = hg, ag
    df["result"] = np.where(hg > ag, "H", np.where(hg == ag, "D", "A"))
    df = df[df["date"].notna() & df["home_team"].notna() & df["away_team"].notna()]
    df = df.drop_duplicates(["season_start", "home_team", "away_team"], keep="last")
    return df[cols].sort_values("date").reset_index(drop=True)


def top_up_from_espn(matches: pd.DataFrame, results: pd.DataFrame | None = None,
                     *, year: int | None = None, **kwargs) -> pd.DataFrame:
    """Add results ESPN has that neither of the other two sources supplied.

    Asked for the current season only: the earlier ones are complete in
    football-data and nothing would be added.
    """
    if matches.empty and results is None:
        return matches
    if year is None:
        if matches.empty:
            return matches
        year = int(matches["season_start"].max())
    if results is None:
        # Start the sweep where what we already have runs out, so a build in
        # the ordinary case asks about the last day or two and no more. An
        # explicit start wins, so a caller can widen the window.
        start = kwargs.pop("start", None)
        if start is None:
            known = matches.loc[matches["season_start"] == year, "date"]
            if not known.empty:
                last = pd.to_datetime(known).max()
                if pd.notna(last):
                    start = last.date()
        try:
            results = espn_results(year, start=start, **kwargs)
        except requests.RequestException as exc:
            log.warning("ESPN unavailable (%s); leaving the table as it is", exc)
            return matches
    if results.empty:
        return matches

    have = set(map(tuple, matches[["season_start", "home_team", "away_team"]]
                   .astype({"season_start": "int64"}).astype(str).to_numpy()))
    keys = results[["season_start", "home_team", "away_team"]].astype(
        {"season_start": "int64"}).astype(str).to_numpy()
    missing = results[[tuple(k) not in have for k in keys]]
    if missing.empty:
        return matches

    log.info("topping up %d match(es) from ESPN", len(missing))
    extra = missing.reindex(columns=matches.columns)
    out = pd.concat([matches, extra], ignore_index=True)
    out = out.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)
    # match_id is positional, so it has to be reissued after inserting rows.
    out["match_id"] = out.index
    return out
