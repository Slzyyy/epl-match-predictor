"""Per-match player rosters from Understat, and squad-availability features.

There is no free historical injury feed with match-level Premier League
coverage. What is available is who actually took the field: Understat's match
endpoint lists every player who appeared, with minutes, position and xG
contribution, from 2014-15 onward.

Availability is derived from that, using **past line-ups only** — a player who
has missed the team's last two matches is treated as probably unavailable.
That proxy is deliberately lagged so the features are known days before
kick-off, exactly like every other feature in this project. It conflates
injury with suspension and rotation, which is the price of the data being free.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_RAW
from .data import season_label
from .xg import FIRST_XG_SEASON, UNDERSTAT_ALIASES, download_season_xg

log = logging.getLogger(__name__)

MATCH_URL = "https://understat.com/getMatchData/{match_id}"
MATCH_PAGE = "https://understat.com/match/{match_id}"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "X-Requested-With": "XMLHttpRequest",
}

# Fields kept from each roster entry; the rest is match detail we do not need.
_KEEP = ("player_id", "player", "position", "positionOrder", "time",
         "goals", "assists", "own_goals", "xG", "xA", "xGChain",
         "yellow_card", "red_card")


def _roster_path(year: int) -> Path:
    return DATA_RAW / f"rosters_EPL_{year}.jsonl"


def season_match_index(year: int) -> pd.DataFrame:
    """Match ids for one season, with teams and dates, from the league feed."""
    payload = download_season_xg(year)
    rows = [
        {
            "understat_id": int(m["id"]),
            "date": pd.to_datetime(m["datetime"]).normalize(),
            "home_team": m["h"]["title"],
            "away_team": m["a"]["title"],
            "season_start": year,
        }
        for m in payload.get("dates", [])
        if m.get("isResult")
    ]
    idx = pd.DataFrame(rows)
    for col in ("home_team", "away_team"):
        idx[col] = idx[col].astype("string").str.strip().replace(UNDERSTAT_ALIASES)
    return idx


def fetch_match_roster(
    match_id: int, session: requests.Session, *, timeout: float = 20.0
) -> list[dict]:
    """One match's roster rows, both sides, trimmed to the fields we use.

    The timeout is short on purpose: this runs in a loop over hundreds of
    matches, so a minute spent waiting on one unreachable request is a minute
    multiplied by however many are left.
    """
    resp = session.get(
        MATCH_URL.format(match_id=match_id),
        headers={**_HEADERS, "Referer": MATCH_PAGE.format(match_id=match_id)},
        timeout=timeout,
    )
    resp.raise_for_status()
    rosters = resp.json().get("rosters", {})
    out = []
    for side, players in rosters.items():
        for entry in players.values():
            row = {k: entry.get(k) for k in _KEEP}
            row["understat_id"] = int(match_id)
            row["is_home"] = side == "h"
            out.append(row)
    return out


def missing_roster_matches(year: int) -> list[int]:
    """Match ids for a season that have not been downloaded yet.

    Separate from the download so a caller can size a run before making any
    requests: the archive is one request per match against a free site.
    """
    path = _roster_path(year)
    done: set[int] = set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["understat_id"])
    try:
        index = season_match_index(year)
    except Exception as exc:  # the feed is down; nothing to do this run
        log.warning("season %s index unavailable (%s)", year, exc)
        return []
    return [m for m in index["understat_id"].tolist() if m not in done]


def download_season_rosters(
    year: int,
    *,
    pause: float = 0.3,
    limit: int | None = None,
    deadline: float | None = None,
    max_failures: int = 5,
) -> int:
    """Fetch match rosters for a season, resumably. Returns how many landed.

    Already-downloaded matches are skipped, so an interrupted run can simply be
    restarted. One request per match against a free site, hence the pause.

    Bounded three ways, because this runs unattended in CI: ``limit`` caps the
    requests, ``deadline`` (a ``time.monotonic`` value) caps the wall clock,
    and ``max_failures`` consecutive errors abandons the run. Without the last
    two, an unreachable source turns a bounded number of requests into an
    unbounded amount of time - which is exactly what it did.
    """
    path = _roster_path(year)
    todo = missing_roster_matches(year)
    if limit:
        todo = todo[:limit]
    if not todo:
        return 0

    log.info("season %s: %d matches to fetch", year, len(todo))
    session = requests.Session()
    fetched = 0
    failures = 0
    with path.open("a", encoding="utf-8") as fh:
        for i, match_id in enumerate(todo, 1):
            if deadline is not None and time.monotonic() >= deadline:
                log.info("  %s: out of time after %d", year, fetched)
                break
            try:
                rows = fetch_match_roster(match_id, session)
            except Exception as exc:
                failures += 1
                log.warning("match %s failed: %s", match_id, exc)
                if failures >= max_failures:
                    log.warning("  %s: giving up after %d consecutive failures",
                                year, failures)
                    break
                time.sleep(1.0)
                continue
            failures = 0
            fh.write(json.dumps({"understat_id": match_id, "rows": rows}) + "\n")
            fh.flush()
            fetched += 1
            if i % 50 == 0:
                log.info("  %s: %d/%d", year, i, len(todo))
            time.sleep(pause)
    return fetched


def backfill_rosters(
    *,
    budget: int = 400,
    max_seconds: float = 240.0,
    first_season: int = FIRST_XG_SEASON,
    last_season: int = 2026,
    pause: float = 0.3,
) -> dict[int, int]:
    """Top the roster archive up by at most ``budget`` matches, newest first.

    The full archive is ~4,600 requests against a free site, which is not a
    reasonable thing to do on every build. Each run instead takes a bounded
    bite; the download is resumable and the raw files are cached between runs,
    so the archive fills in over several days and then stays put. Newest first
    because the current season is what the club pages lead with.

    Returns how many matches were fetched per season, so a caller can report
    whether the archive is complete yet.
    """
    remaining = max(0, budget)
    deadline = time.monotonic() + max_seconds
    fetched: dict[int, int] = {}
    for year in range(last_season, max(first_season, FIRST_XG_SEASON) - 1, -1):
        if remaining <= 0 or time.monotonic() >= deadline:
            break
        missing = len(missing_roster_matches(year))
        if not missing:
            continue
        take = min(missing, remaining)
        got = download_season_rosters(year, pause=pause, limit=take, deadline=deadline)
        if got:
            fetched[year] = got
            remaining -= got
        if got < take:
            break  # out of time, or the source stopped answering
    return fetched


def load_season_rosters(year: int) -> pd.DataFrame:
    """One row per player-match for a season."""
    path = _roster_path(year)
    if not path.exists():
        return pd.DataFrame()
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.extend(json.loads(line)["rows"])
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    numeric = ("time", "goals", "assists", "own_goals", "xG", "xA", "xGChain",
               "positionOrder", "yellow_card", "red_card")
    for col in (c for c in numeric if c in df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["season_start"] = year
    return df


def load_rosters(first_season: int = FIRST_XG_SEASON, last_season: int = 2026) -> pd.DataFrame:
    frames = [
        part
        for year in range(max(first_season, FIRST_XG_SEASON), last_season + 1)
        if not (part := load_season_rosters(year)).empty
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------
# availability features
# --------------------------------------------------------------------------
# How many of a team's previous matches define its "core" squad.
CORE_WINDOW = 10
# A player missing this many consecutive recent matches counts as unavailable.
ABSENCE_STREAKS = (1, 2)

AVAILABILITY_COLS = [
    "absent_minutes_share_1",
    "absent_minutes_share_2",
    "absent_xgchain_share_2",
    "absent_starters_2",
    "squad_rotation",
]


def _team_match_frame(rosters: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Player-match rows labelled with team and date, one row per appearance."""
    # Both frames carry season_start; take it from the index only, or the
    # merge would suffix both copies and the selection below would fail.
    cols = [c for c in rosters.columns if c not in index.columns or c == "understat_id"]
    df = rosters[cols].merge(index, on="understat_id", how="inner")
    df["team"] = df["home_team"].where(df["is_home"], df["away_team"])
    df["started"] = (df["position"] != "Sub").astype(float)
    df["minutes"] = df["time"].fillna(0.0)
    df["xgchain"] = df["xGChain"].fillna(0.0)
    return df[
        ["understat_id", "date", "season_start", "team", "player_id", "player",
         "minutes", "started", "xgchain"]
    ].sort_values(["team", "date"], kind="stable")


def _availability_state(window: list[tuple[int, pd.DataFrame]]) -> dict:
    """Availability summary from a team's previous matches (newest last)."""
    rec: dict = {}
    if len(window) < 3:
        return rec
    past = pd.concat([p for _, p in window], ignore_index=True)
    core = past.groupby("player_id").agg(
        minutes=("minutes", "sum"),
        xgchain=("xgchain", "sum"),
        starts=("started", "sum"),
    )
    total_min = float(core["minutes"].sum())
    total_xgc = float(core["xgchain"].sum())

    for streak in ABSENCE_STREAKS:
        appeared: set = set()
        for _, p in window[-streak:]:
            appeared |= set(p.loc[p["minutes"] > 0, "player_id"])
        miss = core.loc[core.index.difference(list(appeared))]
        if total_min > 0:
            rec[f"absent_minutes_share_{streak}"] = float(miss["minutes"].sum() / total_min)
        if streak == 2:
            rec["absent_xgchain_share_2"] = (
                float(miss["xgchain"].sum() / total_xgc) if total_xgc > 0 else 0.0
            )
            # Regular starters (started over half the window) now absent.
            rec["absent_starters_2"] = float((miss["starts"] >= len(window) / 2).sum())

    if len(window) >= 2:
        a = set(window[-1][1].loc[window[-1][1]["started"] > 0, "player_id"])
        b = set(window[-2][1].loc[window[-2][1]["started"] > 0, "player_id"])
        rec["squad_rotation"] = float(len(a - b))
    return rec


def build_availability(rosters: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Pre-match squad-availability features, one row per team-match.

    For each match, the team's *previous* ``CORE_WINDOW`` matches define a core
    squad weighted by minutes played. A core player who did not appear in the
    last match (or the last two) is treated as unavailable, and the features
    report how much of the squad's minutes and attacking contribution that
    accounts for.

    Everything is computed from matches strictly before the one being
    described, so these are usable days ahead of kick-off. Each team also gets
    a trailing row (``understat_id`` of -1) holding its state after its last
    played match, which is what an upcoming fixture needs.
    """
    df = _team_match_frame(rosters, index)
    if df.empty:
        return pd.DataFrame(columns=["understat_id", "team", *AVAILABILITY_COLS])

    out_rows = []
    for team, grp in df.groupby("team", sort=False):
        matches = list(grp.groupby("understat_id", sort=False))
        matches.sort(key=lambda kv: kv[1]["date"].iloc[0])  # by date, not id
        history: list[tuple[int, pd.DataFrame]] = []

        for match_id, players in matches:
            rec = {"understat_id": match_id, "team": team}
            rec.update(_availability_state(history[-CORE_WINDOW:]))
            out_rows.append(rec)
            history.append((match_id, players))

        # State going into whatever this team plays next.
        trailing = {"understat_id": -1, "team": team}
        trailing.update(_availability_state(history[-CORE_WINDOW:]))
        out_rows.append(trailing)

    out = pd.DataFrame(out_rows)
    for col in AVAILABILITY_COLS:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_availability(
    first_season: int = FIRST_XG_SEASON, last_season: int = 2026
) -> pd.DataFrame:
    """Availability features for every season with roster data on disk."""
    frames = []
    for year in range(max(first_season, FIRST_XG_SEASON), last_season + 1):
        rosters = load_season_rosters(year)
        if rosters.empty:
            continue
        idx = season_match_index(year)
        avail = build_availability(rosters, idx)
        avail = avail.merge(
            idx[["understat_id", "season_start", "home_team", "away_team", "date"]],
            on="understat_id",
            how="left",
        )
        # Trailing rows have no match to join to; label them with the season.
        avail.loc[avail["understat_id"] == -1, "season_start"] = year
        frames.append(avail)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def attach_availability(matches: pd.DataFrame, avail: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join home and away availability onto the match table."""
    if avail is None:
        avail = load_availability()
    out = matches.copy()
    if avail.empty:
        for side in ("home", "away"):
            for col in AVAILABILITY_COLS:
                out[f"{side}_{col}"] = pd.NA
        return out

    key = ["season_start", "home_team", "away_team"]
    played = avail[avail["understat_id"] != -1]
    # One trailing row per team: its state after the most recent match played.
    trailing = (
        avail[avail["understat_id"] == -1]
        .sort_values("season_start")
        .drop_duplicates("team", keep="last")
        .set_index("team")[AVAILABILITY_COLS]
    )

    for side in ("home", "away"):
        part = played[played["team"] == played[f"{side}_team"]]
        part = part[key + AVAILABILITY_COLS].drop_duplicates(key)
        renamed = {c: f"{side}_{c}" for c in AVAILABILITY_COLS}
        out = out.merge(part.rename(columns=renamed), on=key, how="left", validate="many_to_one")

        # Fixtures with no result yet were never in the Understat index, so
        # fill them from each team's trailing state.
        if "result" in out.columns:
            upcoming = out["result"].isna()
            if upcoming.any():
                fill = out.loc[upcoming, f"{side}_team"].map(trailing.to_dict("index"))
                for col in AVAILABILITY_COLS:
                    vals = fill.map(lambda d, c=col: (d or {}).get(c))
                    out.loc[upcoming, renamed[col]] = pd.to_numeric(vals, errors="coerce")
    return out
