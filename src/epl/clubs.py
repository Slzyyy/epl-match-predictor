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

from .config import DATA_RAW, ROOT
from .lineups import load_season_rosters, season_match_index

log = logging.getLogger(__name__)

CLUB_DATA = ROOT / "club_data"
TROPHIES_PATH = CLUB_DATA / "trophies.json"
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
def player_stats(season_start: int) -> pd.DataFrame:
    """Per-player totals for a season, labelled with the club they played for."""
    empty = pd.DataFrame(
        {
            "team": pd.Series(dtype="object"), "player": pd.Series(dtype="object"),
            "apps": pd.Series(dtype="int64"), "minutes": pd.Series(dtype="float64"),
            "goals": pd.Series(dtype="float64"), "xg": pd.Series(dtype="float64"),
            "assists": pd.Series(dtype="float64"), "xa": pd.Series(dtype="float64"),
        }
    )
    rosters = load_season_rosters(season_start)
    if rosters.empty:
        # Typed rather than bare: an object-dtype column reaches nlargest and
        # raises, which is how this surfaced in CI where no rosters exist.
        return empty
    idx = season_match_index(season_start)
    df = rosters.merge(idx[["understat_id", "home_team", "away_team"]], on="understat_id")
    if df.empty:
        return empty
    df["team"] = df["home_team"].where(df["is_home"], df["away_team"])

    agg = {
        "apps": ("time", "size"),
        "minutes": ("time", "sum"),
        "goals": ("goals", "sum"),
        "xg": ("xG", "sum"),
    }
    for col, name in (("assists", "assists"), ("xA", "xa")):
        if col in df.columns:
            agg[name] = (col, "sum")
    out = df.groupby(["team", "player"], as_index=False).agg(**agg)
    for col in ("assists", "xa"):
        if col not in out.columns:
            out[col] = 0
    return out


def _top_n(players: pd.DataFrame, team: str, value: str, expected: str, n: int) -> pd.DataFrame:
    cols = ["player", "apps", "minutes", value, expected]
    if players.empty or value not in players.columns:
        return pd.DataFrame(columns=cols)
    sub = players[(players["team"] == team) & (players[value].fillna(0) > 0)]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    return sub.nlargest(n, [value, expected])[cols]


def top_scorers(players: pd.DataFrame, team: str, n: int = 10) -> pd.DataFrame:
    return _top_n(players, team, "goals", "xg", n)


def top_assists(players: pd.DataFrame, team: str, n: int = 10) -> pd.DataFrame:
    return _top_n(players, team, "assists", "xa", n)


def clean_sheet_keepers(season_start: int, matches: pd.DataFrame, team: str) -> pd.DataFrame:
    """Clean sheets credited to whoever kept goal, for clubs that rotate."""
    rosters = load_season_rosters(season_start)
    if rosters.empty:
        return pd.DataFrame(columns=["player", "starts", "clean_sheets"])
    idx = season_match_index(season_start)
    df = rosters.merge(idx[["understat_id", "home_team", "away_team", "date"]], on="understat_id")
    df["team"] = df["home_team"].where(df["is_home"], df["away_team"])
    keepers = df[(df["team"] == team) & (df["position"] == "GK") & (df["time"] > 0)]
    if keepers.empty:
        return pd.DataFrame(columns=["player", "starts", "clean_sheets"])

    played = season_matches(matches, team, season_start)[["date", "ga"]]
    merged = keepers.merge(played, on="date", how="left")
    out = merged.groupby("player", as_index=False).agg(
        starts=("player", "size"), clean_sheets=("ga", lambda g: int((g == 0).sum()))
    )
    return out.sort_values(["clean_sheets", "starts"], ascending=False).reset_index(drop=True)


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
) -> dict:
    """Everything one club page needs."""
    trophies = trophies if trophies is not None else load_trophies()
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
        "trophies": trophies.get(team, {}),
        "wiki": summaries.get(team, {}),
    }
