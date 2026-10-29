"""Per-club season statistics for the club pages.

Everything here is computed from data the project already holds: results from
football-data, player appearances from Understat. Two things on the club pages
cannot come from either source and are handled separately:

* the prose summary, fetched once from Wikipedia and cached, and
* the honours list, which no structured feed carries and which therefore lives
  in a hand-maintained file (see ``club_data/trophies.json``).
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests

from .config import DATA_RAW, LAST_SEASON, ROOT
from .lineups import load_season_rosters, season_match_index
from .xg import FIRST_XG_SEASON

log = logging.getLogger(__name__)

CLUB_DATA = ROOT / "club_data"
TROPHIES_PATH = CLUB_DATA / "trophies.json"
ALL_TIME_PATH = CLUB_DATA / "all_time_scorers.json"
WIKI_TITLES_PATH = CLUB_DATA / "wikipedia_titles.json"
WIKI_CACHE = DATA_RAW / "wikipedia_summaries.json"

WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_WIKI_HEADERS = {
    "User-Agent": "epl-match-predictor/0.1 (https://github.com/Slzyyy/epl-match-predictor)"
}


def slug(team: str) -> str:
    """'Nott'm Forest' -> 'nottm-forest', for use as a file name."""
    norm = unicodedata.normalize("NFKD", team).encode("ascii", "ignore").decode()
    # Apostrophes are dropped rather than hyphenated, so "Nott'm Forest"
    # becomes nottm-forest and not nott-m-forest.
    norm = norm.lower().replace("'", "").replace("\u2019", "")
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", norm)).strip("-")


# --------------------------------------------------------------------------
# results-derived
# --------------------------------------------------------------------------
def season_matches(matches: pd.DataFrame, team: str, season_start: int) -> pd.DataFrame:
    """Every played match for one club this season, oldest first."""
    sub = matches[
        (matches["season_start"] == season_start)
        & ((matches["home_team"] == team) | (matches["away_team"] == team))
        & matches["result"].notna()
    ].copy()
    if sub.empty:
        return sub.assign(is_home=None, gf=None, ga=None, outcome=None, opponent=None)

    sub["is_home"] = sub["home_team"] == team
    sub["opponent"] = sub["away_team"].where(sub["is_home"], sub["home_team"])
    sub["gf"] = sub["home_goals"].where(sub["is_home"], sub["away_goals"])
    sub["ga"] = sub["away_goals"].where(sub["is_home"], sub["home_goals"])
    sub["outcome"] = pd.Series(
        ["W" if f > a else ("D" if f == a else "L") for f, a in zip(sub["gf"], sub["ga"], strict=True)],
        index=sub.index,
    )
    return sub.sort_values("date")


def season_summary(matches: pd.DataFrame, team: str, season_start: int) -> dict:
    """Played, won, drawn, lost, goals, clean sheets, points."""
    s = season_matches(matches, team, season_start)
    if s.empty:
        return {"played": 0}
    wins = int((s["outcome"] == "W").sum())
    draws = int((s["outcome"] == "D").sum())
    return {
        "played": int(len(s)),
        "won": wins,
        "drawn": draws,
        "lost": int((s["outcome"] == "L").sum()),
        "goals_for": int(s["gf"].sum()),
        "goals_against": int(s["ga"].sum()),
        "goal_difference": int(s["gf"].sum() - s["ga"].sum()),
        "clean_sheets": int((s["ga"] == 0).sum()),
        "failed_to_score": int((s["gf"] == 0).sum()),
        "points": wins * 3 + draws,
    }


def recent_form(matches: pd.DataFrame, team: str, season_start: int, n: int = 5) -> pd.DataFrame:
    """The club's last `n` played matches, most recent first."""
    s = season_matches(matches, team, season_start)
    if s.empty:
        return s
    cols = ["date", "opponent", "is_home", "gf", "ga", "outcome"]
    return s.tail(n)[cols].iloc[::-1].reset_index(drop=True)


def league_table(matches: pd.DataFrame, season_start: int) -> pd.DataFrame:
    """Full table for the season, so each club can show its position."""
    teams = sorted(
        set(matches.loc[matches["season_start"] == season_start, "home_team"].dropna())
    )
    rows = [{"team": t, **season_summary(matches, t, season_start)} for t in teams]
    table = pd.DataFrame([r for r in rows if r.get("played")])
    if table.empty:
        return table
    table = table.sort_values(
        ["points", "goal_difference", "goals_for"], ascending=False
    ).reset_index(drop=True)
    table["position"] = range(1, len(table) + 1)
    return table


# --------------------------------------------------------------------------
# player-derived
# --------------------------------------------------------------------------
def _empty_players(*, seasons: bool = False) -> pd.DataFrame:
    """Typed rather than bare: an object-dtype column reaches ``nlargest`` and
    raises, which is how an empty roster set first surfaced in CI."""
    cols = {
        "team": "object", "player": "object", "apps": "int64",
        "minutes": "float64", "goals": "float64", "xg": "float64",
        "assists": "float64", "xa": "float64",
    }
    if seasons:
        cols["seasons"] = "int64"
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in cols.items()})


def _player_match_frame(season_start: int) -> pd.DataFrame:
    """Player-match rows for one season, labelled with club and date.

    Empty when the season's roster file has not been fetched, which is the
    normal state for a fresh checkout and for CI before the fetch step.
    """
    rosters = load_season_rosters(season_start)
    if rosters.empty:
        return pd.DataFrame()
    idx = season_match_index(season_start)
    # season_start is deliberately left out of the merge: the rosters already
    # carry it, and taking both would collide into season_start_x/_y.
    keep = [c for c in ("understat_id", "home_team", "away_team", "date") if c in idx.columns]
    df = rosters.merge(idx[keep], on="understat_id")
    if df.empty:
        return pd.DataFrame()
    df["team"] = df["home_team"].where(df["is_home"], df["away_team"])
    return df


def _aggregate_players(df: pd.DataFrame, *, seasons: bool) -> pd.DataFrame:
    agg = {
        "apps": ("time", "size"),
        "minutes": ("time", "sum"),
        "goals": ("goals", "sum"),
        "xg": ("xG", "sum"),
    }
    for col, name in (("assists", "assists"), ("xA", "xa")):
        if col in df.columns:
            agg[name] = (col, "sum")
    if seasons:
        agg["seasons"] = ("season_start", "nunique")
    out = df.groupby(["team", "player"], as_index=False).agg(**agg)
    for col in ("assists", "xa"):
        if col not in out.columns:
            out[col] = 0
    return out


def player_stats(season_start: int) -> pd.DataFrame:
    """Per-player totals for a season, labelled with the club they played for."""
    df = _player_match_frame(season_start)
    if df.empty:
        return _empty_players()
    return _aggregate_players(df, seasons=False)


def career_stats(
    first_season: int = FIRST_XG_SEASON, last_season: int = LAST_SEASON
) -> pd.DataFrame:
    """Per-player totals per club across every season of roster data held.

    Understat's player feed starts in 2014-15, so this is a Premier League
    record over that window rather than a true all-time one; the all-time club
    record scorers are hand-maintained instead, in ``all_time_scorers.json``.
    A player who moved clubs gets one row per club, which is what a club page
    wants: it should credit a striker only with the goals he scored there.
    """
    frames = [
        part
        for year in range(max(first_season, FIRST_XG_SEASON), last_season + 1)
        if not (part := _player_match_frame(year)).empty
    ]
    if not frames:
        return _empty_players(seasons=True)
    return _aggregate_players(pd.concat(frames, ignore_index=True), seasons=True)


def career_seasons(
    first_season: int = FIRST_XG_SEASON, last_season: int = LAST_SEASON
) -> list[int]:
    """Which seasons actually have roster data, so a page can say so."""
    return [
        year
        for year in range(max(first_season, FIRST_XG_SEASON), last_season + 1)
        if not load_season_rosters(year).empty
    ]


def _top_n(players: pd.DataFrame, team: str, value: str, expected: str, n: int) -> pd.DataFrame:
    cols = ["player", "apps", "minutes", value, expected]
    if players.empty or value not in players.columns:
        return pd.DataFrame(columns=cols)
    if "seasons" in players.columns:
        cols.append("seasons")
    sub = players[(players["team"] == team) & (players[value].fillna(0) > 0)]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    return sub.nlargest(n, [value, expected])[cols]


def top_scorers(players: pd.DataFrame, team: str, n: int = 10) -> pd.DataFrame:
    return _top_n(players, team, "goals", "xg", n)


def top_assists(players: pd.DataFrame, team: str, n: int = 10) -> pd.DataFrame:
    return _top_n(players, team, "assists", "xa", n)


_KEEPER_COLS = ["player", "starts", "clean_sheets"]


def _team_goals_against(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per club-match: club, date, goals conceded.

    Long rather than wide so a goalkeeper's appearances join on club and date
    without caring which end of the fixture the club was.
    """
    played = matches[matches["result"].notna()]
    home = played[["date", "home_team", "away_goals"]].rename(
        columns={"home_team": "team", "away_goals": "ga"}
    )
    away = played[["date", "away_team", "home_goals"]].rename(
        columns={"away_team": "team", "home_goals": "ga"}
    )
    return pd.concat([home, away], ignore_index=True)


def _keeper_totals(
    df: pd.DataFrame, conceded: pd.DataFrame, *, seasons: bool
) -> pd.DataFrame:
    """Clean sheets credited to whoever kept goal, for clubs that rotate."""
    cols = ["team", *_KEEPER_COLS] + (["seasons"] if seasons else [])
    if df.empty or "position" not in df.columns:
        return pd.DataFrame(columns=cols)
    keepers = df[(df["position"] == "GK") & (df["time"] > 0)]
    if keepers.empty:
        return pd.DataFrame(columns=cols)
    merged = keepers.merge(conceded, on=["team", "date"], how="left")
    agg = {
        "starts": ("player", "size"),
        "clean_sheets": ("ga", lambda g: int((g == 0).sum())),
    }
    if seasons:
        agg["seasons"] = ("season_start", "nunique")
    out = merged.groupby(["team", "player"], as_index=False).agg(**agg)
    return out.sort_values(["clean_sheets", "starts"], ascending=False).reset_index(drop=True)


def clean_sheet_keepers(season_start: int, matches: pd.DataFrame, team: str) -> pd.DataFrame:
    """Clean sheets credited to whoever kept goal, for clubs that rotate."""
    out = _keeper_totals(
        _player_match_frame(season_start), _team_goals_against(matches), seasons=False
    )
    if out.empty:
        return pd.DataFrame(columns=_KEEPER_COLS)
    return out[out["team"] == team].drop(columns="team").reset_index(drop=True)


def career_keepers(
    matches: pd.DataFrame,
    first_season: int = FIRST_XG_SEASON,
    last_season: int = LAST_SEASON,
) -> pd.DataFrame:
    """Clean sheets per goalkeeper per club across every season held.

    Computed once for every club rather than per club: each season's roster
    file is read from disk, and twenty clubs re-reading thirteen files is the
    slowest thing the site build could do.
    """
    frames = [
        part
        for year in range(max(first_season, FIRST_XG_SEASON), last_season + 1)
        if not (part := _player_match_frame(year)).empty
    ]
    if not frames:
        return pd.DataFrame(columns=["team", *_KEEPER_COLS, "seasons"])
    return _keeper_totals(
        pd.concat(frames, ignore_index=True), _team_goals_against(matches), seasons=True
    )


def team_keepers(keepers: pd.DataFrame, team: str, n: int = 10) -> pd.DataFrame:
    """One club's goalkeepers from a career frame, best record first."""
    if keepers.empty or "team" not in keepers.columns:
        return pd.DataFrame(columns=[*_KEEPER_COLS, "seasons"])
    sub = keepers[keepers["team"] == team].drop(columns="team")
    return sub.head(n).reset_index(drop=True)


# --------------------------------------------------------------------------
# hand-maintained and external
# --------------------------------------------------------------------------
def load_trophies() -> dict:
    """Honours, from the hand-maintained file.

    No free structured feed carries club honours — Wikidata models football
    clubs' `award received` as media awards, not league titles — so this is a
    curated file. It is deliberately plain JSON so it can be corrected without
    touching any code.
    """
    if not TROPHIES_PATH.exists():
        return {}
    return json.loads(TROPHIES_PATH.read_text(encoding="utf-8"))


def load_all_time_scorers() -> dict:
    """All-time club goalscoring records, from the hand-maintained file.

    Understat's player feed only reaches 2014-15, so the club record scorers -
    almost all of whom played decades before that - cannot be computed here.
    Researched from each club's records article rather than written from
    memory; the file's own _README gives the source and basis per club.
    """
    if not ALL_TIME_PATH.exists():
        return {}
    return json.loads(ALL_TIME_PATH.read_text(encoding="utf-8"))


def _wiki_title(team: str, titles: dict) -> str:
    return titles.get(team, f"{team} F.C.")


def fetch_summaries(teams: list[str], *, refresh: bool = False) -> dict:
    """Short Wikipedia extracts, cached on disk.

    Only the opening couple of sentences are kept, with the article link, so
    the page quotes briefly and attributes rather than reproducing an article.
    """
    cache = {}
    if WIKI_CACHE.exists() and not refresh:
        cache = json.loads(WIKI_CACHE.read_text(encoding="utf-8"))
    titles = (
        json.loads(WIKI_TITLES_PATH.read_text(encoding="utf-8"))
        if WIKI_TITLES_PATH.exists() else {}
    )

    for team in teams:
        if team in cache and not refresh:
            continue
        title = _wiki_title(team, titles)
        try:
            resp = requests.get(
                WIKI_SUMMARY_URL.format(title=title.replace(" ", "_")),
                headers=_WIKI_HEADERS, timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("no Wikipedia summary for %s (%s): %s", team, title, exc)
            continue
        extract = (data.get("extract") or "").strip()
        sentences = re.split(r"(?<=\.)\s+", extract)
        cache[team] = {
            "extract": " ".join(sentences[:2]).strip(),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "title": data.get("title", title),
        }

    WIKI_CACHE.parent.mkdir(parents=True, exist_ok=True)
    WIKI_CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache


def club_profile(
    team: str,
    matches: pd.DataFrame,
    players: pd.DataFrame,
    season_start: int,
    *,
    table: pd.DataFrame | None = None,
    trophies: dict | None = None,
    summaries: dict | None = None,
    career: pd.DataFrame | None = None,
    career_gk: pd.DataFrame | None = None,
    all_time: dict | None = None,
    career_label: str = "",
) -> dict:
    """Everything one club page needs.

    The career and all-time blocks are passed in rather than computed here:
    both are the same for every club, and recomputing them twenty times would
    re-read every season's roster file twenty times over.
    """
    trophies = trophies if trophies is not None else load_trophies()
    all_time = all_time if all_time is not None else load_all_time_scorers()
    career = career if career is not None else _empty_players(seasons=True)
    career_gk = career_gk if career_gk is not None else pd.DataFrame()
    summaries = summaries or {}
    summary = season_summary(matches, team, season_start)
    position = None
    if table is not None and not table.empty and team in set(table["team"]):
        position = int(table.loc[table["team"] == team, "position"].iloc[0])
    return {
        "team": team,
        "slug": slug(team),
        "season": summary,
        "position": position,
        "form": recent_form(matches, team, season_start),
        "scorers": top_scorers(players, team),
        "assists": top_assists(players, team),
        "keepers": clean_sheet_keepers(season_start, matches, team),
        "career_scorers": top_scorers(career, team),
        "career_assists": top_assists(career, team),
        "career_keepers": team_keepers(career_gk, team),
        "all_time": all_time.get(team, {}),
        "career_label": career_label,
        "trophies": trophies.get(team, {}),
        "wiki": summaries.get(team, {}),
    }
