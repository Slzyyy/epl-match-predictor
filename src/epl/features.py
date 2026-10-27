"""Feature engineering.

Every feature is built from information that existed *before* kick-off. The
two mechanisms used are:

* rolling / cumulative statistics shifted by one match within each team, so a
  match never sees its own outcome, and
* an Elo rating updated strictly in date order.

The entry point is :func:`build_features`, which returns one row per match
with the label plus the feature matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LABEL_TO_INT
from .lineups import AVAILABILITY_COLS

# Rolling windows (in matches) used for form features.
FORM_WINDOWS = (3, 5, 10)
# Windows for venue-specific form (home-only or away-only matches).
VENUE_WINDOWS = (6, 12)
# Goals considered when integrating the Poisson scoreline grid.
MAX_GOALS = 10

# Per-team match statistics that get rolling averages.
STAT_PAIRS = {
    "goals": ("home_goals", "away_goals"),
    "shots": ("home_shots", "away_shots"),
    "sot": ("home_shots_target", "away_shots_target"),
    "corners": ("home_corners", "away_corners"),
    "cards": (None, None),  # derived below
}

ELO_START = 1500.0
ELO_PROMOTED = 1430.0
ELO_K = 20.0
ELO_HFA = 65.0
ELO_SEASON_REGRESSION = 0.80  # fraction of the deviation from 1500 carried over


# --------------------------------------------------------------------------
# long (per team-match) view
# --------------------------------------------------------------------------
def to_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Explode each match into two team-perspective rows."""

    def side(is_home: bool) -> pd.DataFrame:
        p, o = ("home", "away") if is_home else ("away", "home")
        out = pd.DataFrame(
            {
                "match_id": matches["match_id"].to_numpy(),
                "date": matches["date"].to_numpy(),
                "season": matches["season"].to_numpy(),
                "season_start": matches["season_start"].to_numpy(),
                "team": matches[f"{p}_team"].to_numpy(),
                "opponent": matches[f"{o}_team"].to_numpy(),
                "is_home": is_home,
                "goals_for": matches[f"{p}_goals"].to_numpy(),
                "goals_against": matches[f"{o}_goals"].to_numpy(),
                "shots_for": matches[f"{p}_shots"].to_numpy(),
                "shots_against": matches[f"{o}_shots"].to_numpy(),
                "sot_for": matches[f"{p}_shots_target"].to_numpy(),
                "sot_against": matches[f"{o}_shots_target"].to_numpy(),
                "corners_for": matches[f"{p}_corners"].to_numpy(),
                "corners_against": matches[f"{o}_corners"].to_numpy(),
                "cards": (
                    matches[f"{p}_yellows"].fillna(0) + 3 * matches[f"{p}_reds"].fillna(0)
                ).to_numpy(),
            }
        )
        if "home_xg" in matches.columns:
            num = lambda c: pd.to_numeric(matches[c], errors="coerce").to_numpy()
            out["xg_for"] = num(f"{p}_xg")
            out["xg_against"] = num(f"{o}_xg")
            out["npxg_for"] = num(f"{p}_npxg")
            out["npxg_against"] = num(f"{o}_npxg")
            out["ppda"] = num(f"{p}_ppda")
            out["deep_for"] = num(f"{p}_deep")
            out["deep_against"] = num(f"{o}_deep")
            # How much a side out- or under-shot its chances. Finishing runs
            # regress, so a large recent gap is a signal about the *next* match.
            out["xg_overperf"] = out["goals_for"] - out["xg_for"]
            out["xga_overperf"] = out["goals_against"] - out["xg_against"]
        return out

    long = pd.concat([side(True), side(False)], ignore_index=True)
    # Fixtures that have not kicked off yet are carried through the pipeline so
    # they pick up pre-match features, but they must not contribute results to
    # anything: every outcome-derived column stays NaN for them.
    long["played"] = long["goals_for"].notna() & long["goals_against"].notna()
    long["goal_diff"] = long["goals_for"] - long["goals_against"]
    long["points"] = np.select(
        [long["goal_diff"] > 0, long["goal_diff"] == 0], [3.0, 1.0], default=0.0
    )
    long["win"] = (long["goal_diff"] > 0).astype(float)
    long["clean_sheet"] = (long["goals_against"] == 0).astype(float)
    long["failed_to_score"] = (long["goals_for"] == 0).astype(float)
    blanked = ["points", "win", "clean_sheet", "failed_to_score"]
    blanked += [c for c in XG_ROLLING_STATS if c in long.columns]
    long.loc[~long["played"], blanked] = np.nan
    return long.sort_values(["team", "date", "match_id"], kind="stable").reset_index(drop=True)


# xG-derived per-team stats. Present only when Understat data has been
# attached (2014-15 onward); everything before that stays NaN, which the trees
# handle natively.
XG_ROLLING_STATS = [
    "xg_for",
    "xg_against",
    "npxg_for",
    "npxg_against",
    "xg_overperf",
    "xga_overperf",
    "ppda",
    "deep_for",
    "deep_against",
]

ROLLING_STATS = [
    "points",
    "win",
    "goals_for",
    "goals_against",
    "goal_diff",
    "shots_for",
    "shots_against",
    "sot_for",
    "sot_against",
    "corners_for",
    "corners_against",
    "cards",
    "clean_sheet",
    "failed_to_score",
    *XG_ROLLING_STATS,
]


def _pre_match_rolling(
    long: pd.DataFrame,
    stats: list[str],
    window: int,
    min_periods: int,
    keys: list[pd.Series],
) -> pd.DataFrame:
    """Mean of each group's previous ``window`` *played* matches, for every row.

    The rolling mean is computed over played matches only and then carried
    forward, which matters in two places:

    * an unplayed fixture sees the state as of the last match actually played, and
    * several unplayed fixtures in a row all see that same state rather than
      each other's blank results.

    Computing it on the full frame instead would let a NaN row occupy a slot in
    the window, silently shortening it.
    """
    played = long["played"].to_numpy()
    sub = long.loc[played, stats]
    sub_keys = [k.loc[played] for k in keys]
    rolled = sub.groupby(sub_keys, sort=False).rolling(window, min_periods=min_periods).mean()
    rolled.index = rolled.index.get_level_values(-1)

    full = rolled.reindex(long.index)
    carried = full.groupby(keys, sort=False).ffill()
    # Shift by one row inside the group so a match never sees its own result.
    return carried.groupby(keys, sort=False).shift(1)


def _played_matches_in_window(long: pd.DataFrame, days: int) -> np.ndarray:
    """Count of matches each team actually played in the ``days`` before kick-off."""
    out = np.zeros(len(long))
    dates = long["date"].to_numpy()
    played = long["played"].to_numpy()
    span = np.timedelta64(days, "D")
    for idx in long.groupby("team", sort=False).indices.values():
        d = dates[idx]
        hist = np.sort(d[played[idx]])
        lo = np.searchsorted(hist, d - span, side="left")
        hi = np.searchsorted(hist, d, side="left")  # strictly before kick-off
        out[idx] = hi - lo
    return out


def add_form(long: pd.DataFrame) -> pd.DataFrame:
    """Rolling pre-match form, overall and split by venue."""
    team_key = [long["team"]]
    stats = [c for c in ROLLING_STATS if c in long.columns]
    for window in FORM_WINDOWS:
        rolled = _pre_match_rolling(long, stats, window, max(2, window // 2), team_key)
        long[[f"form{window}_{c}" for c in stats]] = rolled

    # Venue-specific form: a team's record at home only, or away only.
    venue_stats = ["points", "goals_for", "goals_against", "shots_for", "shots_against"]
    venue_stats += [c for c in ("xg_for", "xg_against") if c in long.columns]
    venue_key = [long["team"], long["is_home"]]
    for window in VENUE_WINDOWS:
        rolled = _pre_match_rolling(long, venue_stats, window, 3, venue_key)
        long[[f"venue_form{window}_{c}" for c in venue_stats]] = rolled

    # Season to date (resets each August). Sums are taken over a zero-filled
    # copy so an unplayed fixture at the end of the frame does not turn the
    # running totals into NaN; the subtraction then removes the match's own
    # contribution, leaving a strictly pre-match total.
    season_key = [long["team"], long["season"]]
    played_num = long["played"].astype(float)
    season_cols = {
        "season_matches": played_num.groupby(season_key, sort=False).cumsum() - played_num
    }
    for col in ("points", "goals_for", "goals_against", "goal_diff"):
        filled = long[col].fillna(0.0)
        season_cols[f"season_{col}"] = filled.groupby(season_key, sort=False).cumsum() - filled
    played_so_far = season_cols["season_matches"].replace(0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for col in ("points", "goals_for", "goals_against", "goal_diff"):
            season_cols[f"season_{col}_pm"] = season_cols[f"season_{col}"] / played_so_far
    # Added together: one insert per column fragments the frame badly here.
    long = pd.concat([long, pd.DataFrame(season_cols, index=long.index)], axis=1)

    # Scheduling: rest since the last match actually played, and congestion.
    played_date = long["date"].where(long["played"])
    last_played = (
        played_date.groupby(long["team"], sort=False).ffill()
        .groupby(long["team"], sort=False).shift(1)
    )
    rest_days = (long["date"] - last_played).dt.days
    rest_days = rest_days.where(long["season_matches"] != 0)

    # League experience, a proxy for "newly promoted".
    seasons_seen = (
        long.drop_duplicates(["team", "season"])
        .assign(n=lambda d: d.groupby("team", sort=False).cumcount())
        .set_index(["team", "season"])["n"]
    )
    # Added in one go; inserting these one at a time fragments the frame.
    tail = pd.DataFrame(
        {
            "rest_days": rest_days,
            "matches_last_14d": _played_matches_in_window(long, 14),
            "epl_matches_played": (
                played_num.groupby(long["team"], sort=False).cumsum() - played_num
            ),
            "epl_seasons_before": pd.MultiIndex.from_arrays(
                [long["team"], long["season"]]
            ).map(seasons_seen),
        },
        index=long.index,
    )
    return pd.concat([long, tail], axis=1)


# --------------------------------------------------------------------------
# Elo
# --------------------------------------------------------------------------
def _margin_multiplier(goal_diff: int, elo_diff_winner: float) -> float:
    """Dampen rating moves in lopsided games (FiveThirtyEight-style)."""
    return float(np.log(abs(goal_diff) + 1.0) * (2.2 / (0.001 * elo_diff_winner + 2.2)))


def add_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-match Elo ratings, updating in strict date order."""
    ratings: dict[str, float] = {}
    last_season: str | None = None
    first_season = matches["season"].iloc[0] if len(matches) else None
    home_elo = np.empty(len(matches))
    away_elo = np.empty(len(matches))

    cols = matches[["season", "home_team", "away_team", "home_goals", "away_goals"]]
    for i, (season, home, away, hg, ag) in enumerate(cols.itertuples(index=False, name=None)):
        if season != last_season:
            if last_season is not None:
                # Carry ratings over, pulled back toward the mean.
                for team in ratings:
                    ratings[team] = ELO_START + ELO_SEASON_REGRESSION * (ratings[team] - ELO_START)
            last_season = season
        # Teams seen for the first time start average in season one and at a
        # promoted-side discount thereafter.
        default = ELO_START if season == first_season else ELO_PROMOTED
        r_h = ratings.setdefault(home, default)
        r_a = ratings.setdefault(away, default)
        home_elo[i] = r_h
        away_elo[i] = r_a

        if not (np.isfinite(hg) and np.isfinite(ag)):
            continue  # fixture not played yet: rating recorded, no update
        exp_h = 1.0 / (1.0 + 10 ** ((r_a - r_h - ELO_HFA) / 400.0))
        gd = int(hg - ag)
        actual_h = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        if gd == 0:
            mult = 1.0
        else:
            winner_edge = (r_h + ELO_HFA - r_a) if gd > 0 else (r_a - r_h - ELO_HFA)
            mult = _margin_multiplier(gd, max(winner_edge, 0.0))
        delta = ELO_K * mult * (actual_h - exp_h)
        ratings[home] = r_h + delta
        ratings[away] = r_a - delta

    matches = matches.copy()
    matches["home_elo"] = home_elo
    matches["away_elo"] = away_elo
    matches["elo_diff"] = matches["home_elo"] - matches["away_elo"] + ELO_HFA
    matches["elo_exp_home"] = 1.0 / (1.0 + 10 ** (-matches["elo_diff"] / 400.0))
    return matches


# --------------------------------------------------------------------------
# head to head
# --------------------------------------------------------------------------
def add_head_to_head(matches: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Home team's average points in their last `window` meetings with this opponent."""
    key = [
        "|".join(sorted((h, a)))
        for h, a in zip(matches["home_team"], matches["away_team"], strict=True)
    ]
    df = matches[["home_team", "away_team"]].copy()
    df["pair"] = key
    df["home_pts"] = np.select(
        [matches["home_goals"] > matches["away_goals"], matches["home_goals"] == matches["away_goals"]],
        [3.0, 1.0],
        default=0.0,
    )
    df.loc[matches["home_goals"].isna() | matches["away_goals"].isna(), "home_pts"] = np.nan
    # Express points from the perspective of the alphabetically-first team so
    # the rolling mean is comparable across reversed fixtures.
    first_team = [p.split("|")[0] for p in key]
    df["pts_first"] = np.where(
        df["home_team"].to_numpy() == np.asarray(first_team), df["home_pts"], 3.0 - df["home_pts"]
    )
    g = df.groupby("pair", sort=False)["pts_first"]
    rolled = g.shift(1).groupby(df["pair"], sort=False).rolling(window, min_periods=2).mean()
    rolled.index = rolled.index.get_level_values(-1)
    h2h_first = rolled.reindex(df.index)
    matches = matches.copy()
    matches["h2h_home_pts"] = np.where(
        df["home_team"].to_numpy() == np.asarray(first_team), h2h_first, 3.0 - h2h_first
    )
    return matches


# --------------------------------------------------------------------------
# Poisson attack / defence strength
# --------------------------------------------------------------------------
def _poisson_grid_probs(lam_h: np.ndarray, lam_a: np.ndarray) -> np.ndarray:
    """H/D/A probabilities from two independent Poisson goal counts."""
    probs = np.full((len(lam_h), 3), np.nan)
    ok = np.isfinite(lam_h) & np.isfinite(lam_a)
    if not ok.any():
        return probs
    k = np.arange(MAX_GOALS + 1)
    log_fact = np.cumsum(np.concatenate([[0.0], np.log(np.arange(1, MAX_GOALS + 1))]))
    ph = np.exp(-lam_h[ok, None] + k * np.log(lam_h[ok, None]) - log_fact)
    pa = np.exp(-lam_a[ok, None] + k * np.log(lam_a[ok, None]) - log_fact)
    grid = ph[:, :, None] * pa[:, None, :]
    grid /= grid.sum(axis=(1, 2), keepdims=True)
    # grid[:, i, j] = P(home scores i, away scores j).
    home_win = np.tril(np.ones((MAX_GOALS + 1, MAX_GOALS + 1)), -1)  # i > j
    probs[ok, 0] = (grid * home_win).sum(axis=(1, 2))
    probs[ok, 1] = np.trace(grid, axis1=1, axis2=2)
    probs[ok, 2] = (grid * home_win.T).sum(axis=(1, 2))
    return probs


def add_xg_poisson(out: pd.DataFrame) -> pd.DataFrame:
    """The same attack/defence Poisson model, driven by xG instead of goals.

    Expected goals are a less noisy measure of how good the chances were, so
    the strength ratios built from them should settle faster than the
    goal-based ones. Only defined from 2014-15, where Understat data exists.
    """
    hf, af = "home_venue_form12_", "away_venue_form12_"
    if f"{hf}xg_for" not in out.columns:
        return out
    out = out.copy()
    lg_home = out["home_xg"].expanding().mean().shift(1).bfill()
    lg_away = out["away_xg"].expanding().mean().shift(1).bfill()

    out["home_xg_attack"] = out[f"{hf}xg_for"] / lg_home
    out["home_xg_defence"] = out[f"{hf}xg_against"] / lg_away
    out["away_xg_attack"] = out[f"{af}xg_for"] / lg_away
    out["away_xg_defence"] = out[f"{af}xg_against"] / lg_home

    out["exp_xg_home"] = (out["home_xg_attack"] * out["away_xg_defence"] * lg_home).clip(0.05, 6.0)
    out["exp_xg_away"] = (out["away_xg_attack"] * out["home_xg_defence"] * lg_away).clip(0.05, 6.0)
    out["exp_xg_diff"] = out["exp_xg_home"] - out["exp_xg_away"]

    probs = _poisson_grid_probs(
        out["exp_xg_home"].to_numpy(dtype=float), out["exp_xg_away"].to_numpy(dtype=float)
    )
    out["xgpois_p_h"], out["xgpois_p_d"], out["xgpois_p_a"] = probs[:, 0], probs[:, 1], probs[:, 2]
    return out


def add_poisson(out: pd.DataFrame) -> pd.DataFrame:
    """Classic attack/defence strength ratios and the scoreline they imply.

    Each side's venue-specific scoring and conceding rates are divided by the
    league's trailing average to give a strength multiplier, the two
    multipliers give an expected goal count per side, and an independent
    Poisson grid turns those into home/draw/away probabilities. This is the
    standard football baseline and it gives the trees a well-shaped summary of
    the rate features they would otherwise have to recombine themselves.
    """
    out = out.copy()
    # Trailing league averages, shifted so a match never sees its own goals.
    lg_home = out["home_goals"].expanding().mean().shift(1)
    lg_away = out["away_goals"].expanding().mean().shift(1)
    lg_home = lg_home.bfill().fillna(1.5)
    lg_away = lg_away.bfill().fillna(1.1)

    hf, af = "home_venue_form12_", "away_venue_form12_"
    out["home_attack"] = out[f"{hf}goals_for"] / lg_home
    out["home_defence"] = out[f"{hf}goals_against"] / lg_away
    out["away_attack"] = out[f"{af}goals_for"] / lg_away
    out["away_defence"] = out[f"{af}goals_against"] / lg_home

    out["exp_home_goals"] = (out["home_attack"] * out["away_defence"] * lg_home).clip(0.05, 6.0)
    out["exp_away_goals"] = (out["away_attack"] * out["home_defence"] * lg_away).clip(0.05, 6.0)
    out["exp_goal_diff"] = out["exp_home_goals"] - out["exp_away_goals"]
    out["exp_total_goals"] = out["exp_home_goals"] + out["exp_away_goals"]

    probs = _poisson_grid_probs(
        out["exp_home_goals"].to_numpy(dtype=float), out["exp_away_goals"].to_numpy(dtype=float)
    )
    out["pois_p_h"], out["pois_p_d"], out["pois_p_a"] = probs[:, 0], probs[:, 1], probs[:, 2]
    return out


# --------------------------------------------------------------------------
# bookmaker odds and how they moved
# --------------------------------------------------------------------------
def _devig(odds: np.ndarray) -> np.ndarray:
    """Decimal odds -> probabilities, normalised to remove the bookmaker margin."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / odds
    total = inv.sum(axis=1, keepdims=True)
    return inv / total


def add_odds_movement(out: pd.DataFrame) -> pd.DataFrame:
    """Opening and closing market prices, and the drift between them.

    football-data publishes an early price and a closing price for the same
    bookmaker. The difference is where the money went during the week, which
    is the market's own revision of its opinion — a different kind of signal
    from anything else in this project, since it is information about what
    other people know rather than about the teams.
    """
    need = ["open_h", "open_d", "open_a", "close_h", "close_d", "close_a"]
    if not all(c in out.columns for c in need):
        return out
    out = out.copy()
    opens = out[["open_h", "open_d", "open_a"]].to_numpy(dtype=float)
    closes = out[["close_h", "close_d", "close_a"]].to_numpy(dtype=float)

    open_p, close_p = _devig(opens), _devig(closes)
    cols = {}
    for i, k in enumerate(("h", "d", "a")):
        cols[f"mkt_open_{k}"] = open_p[:, i]
        cols[f"mkt_close_{k}"] = close_p[:, i]
        # Log probabilities are the right scale for a linear model to stack on.
        cols[f"mkt_close_log_{k}"] = np.log(np.clip(close_p[:, i], 1e-6, None))
        # Drift: positive means the market moved toward this outcome.
        cols[f"mov_{k}"] = close_p[:, i] - open_p[:, i]
        cols[f"mov_logratio_{k}"] = np.log(
            np.clip(close_p[:, i], 1e-6, None) / np.clip(open_p[:, i], 1e-6, None)
        )

    # How far the market moved overall, and how much margin it took out.
    cols["mov_abs"] = np.abs(close_p - open_p).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cols["mkt_overround_open"] = (1.0 / opens).sum(axis=1)
        cols["mkt_overround_close"] = (1.0 / closes).sum(axis=1)
    cols["mkt_overround_drop"] = cols["mkt_overround_open"] - cols["mkt_overround_close"]

    return pd.concat([out, pd.DataFrame(cols, index=out.index)], axis=1)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def _long_feature_columns(long: pd.DataFrame) -> list[str]:
    prefixes = ("form", "venue_form", "season_")
    cols = [
        c
        for c in long.columns
        if c.startswith(prefixes) and c not in ("season", "season_start")
    ]
    cols += ["rest_days", "matches_last_14d", "epl_matches_played", "epl_seasons_before"]
    return cols


def build_features(matches: pd.DataFrame, *, include_odds: bool = False) -> pd.DataFrame:
    """Build the modelling table: one row per match, label plus features."""
    matches = matches.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
    long = add_form(to_long(matches))
    feat_cols = _long_feature_columns(long)

    home = long[long["is_home"]].set_index("match_id")[feat_cols]
    away = long[~long["is_home"]].set_index("match_id")[feat_cols]
    home.columns = [f"home_{c}" for c in feat_cols]
    away.columns = [f"away_{c}" for c in feat_cols]

    # Joined in one concat: two successive .join() calls leave the frame
    # fragmented across ~300 columns.
    # .copy() consolidates the blocks: `matches` arrives fragmented from the
    # xG and availability joins, and concat would carry that through.
    base = matches.set_index("match_id").copy()
    out = pd.concat([base, home.reindex(base.index), away.reindex(base.index)], axis=1)
    out = out.reset_index()
    out = add_elo(out)
    out = add_head_to_head(out)
    out = add_poisson(out)
    out = add_xg_poisson(out)

    # Differences carry most of the signal, so give the model them directly.
    diffs = pd.DataFrame(
        {
            f"diff_{col}": out[f"home_{col}"].to_numpy(dtype="float64")
            - out[f"away_{col}"].to_numpy(dtype="float64")
            for col in feat_cols
        },
        index=out.index,
    )
    out = pd.concat([out, diffs], axis=1)

    # Added in one go: inserting these one at a time fragments the frame.
    out = pd.concat(
        [
            out,
            pd.DataFrame(
                {
                    "month": out["date"].dt.month,
                    "is_weekend": out["date"].dt.dayofweek.isin([5, 6]).astype(int),
                    "matchweek": (out["home_season_matches"] + out["away_season_matches"]) / 2.0,
                },
                index=out.index,
            ),
        ],
        axis=1,
    )

    if include_odds:
        inv = 1.0 / out[["odds_h", "odds_d", "odds_a"]]
        overround = inv.sum(axis=1)
        out["mkt_prob_h"] = inv["odds_h"] / overround
        out["mkt_prob_d"] = inv["odds_d"] / overround
        out["mkt_prob_a"] = inv["odds_a"] / overround
        out = add_odds_movement(out)

    # Availability features arrive pre-joined in home_/away_ form; give the
    # model their difference too, like every other paired feature.
    avail_pairs = [
        c for c in AVAILABILITY_COLS if f"home_{c}" in out.columns and f"away_{c}" in out.columns
    ]
    if avail_pairs:
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    {
                        f"diff_{c}": pd.to_numeric(out[f"home_{c}"], errors="coerce").to_numpy()
                        - pd.to_numeric(out[f"away_{c}"], errors="coerce").to_numpy()
                        for c in avail_pairs
                    },
                    index=out.index,
                ),
            ],
            axis=1,
        )

    out["y"] = out["result"].map(LABEL_TO_INT)
    return out


META_COLS = {
    "match_id", "date", "time", "season", "season_start", "home_team", "away_team", "played",
    "result", "y", "home_goals", "away_goals", "ht_home_goals", "ht_away_goals",
    "odds_h", "odds_d", "odds_a",
    "open_h", "open_d", "open_a", "close_h", "close_d", "close_a",
}
# Raw statistics from the match itself must never reach the model.
LEAKY_COLS = {
    "home_shots", "away_shots", "home_shots_target", "away_shots_target",
    "home_corners", "away_corners", "home_fouls", "away_fouls",
    "home_yellows", "away_yellows", "home_reds", "away_reds",
    "home_xg", "away_xg", "home_npxg", "away_npxg",
    "home_ppda", "away_ppda", "home_deep", "away_deep",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns, with metadata and post-match stats excluded."""
    cols = []
    for c in df.columns:
        if c in META_COLS or c in LEAKY_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols

# Features kept by the "core" set: the home/away differences plus the ratings
# and Poisson summaries. Backtesting showed this ~78-column set scores exactly
# as well as the full ~214-column one, so it is the default: fewer columns,
# faster fits, less room for the trees to chase noise.
CORE_PREFIXES = ("diff_", "elo", "pois_", "exp_", "xgpois_", "mkt_", "mov_")
CORE_EXTRA = ("h2h_home_pts", "matchweek", "home_elo", "away_elo", "mkt_prob_h",
              "mkt_prob_d", "mkt_prob_a")
CORE_EXTRA += tuple(f"{side}_{c}" for side in ("home", "away") for c in AVAILABILITY_COLS)


def core_feature_columns(df: pd.DataFrame) -> list[str]:
    """The reduced feature set (see :data:`CORE_PREFIXES`)."""
    return [
        c
        for c in feature_columns(df)
        if c.startswith(CORE_PREFIXES) or c in CORE_EXTRA
    ]


# Substrings that identify a feature as xG-derived, for the ablation in the
# backtest (`--no-xg`) and for dropping them when no Understat data is loaded.
XG_MARKERS = ("xg", "npxg", "ppda", "deep")


def is_xg_feature(col: str) -> bool:
    """True when a feature column is derived from Understat data."""
    return any(m in col for m in XG_MARKERS)


def drop_xg(cols: list[str]) -> list[str]:
    return [c for c in cols if not is_xg_feature(c)]


# Substrings identifying squad-availability features, for the `--no-avail`
# ablation in the backtest.
AVAIL_MARKERS = ("absent_", "squad_rotation")


def is_availability_feature(col: str) -> bool:
    return any(m in col for m in AVAIL_MARKERS)


def drop_availability(cols: list[str]) -> list[str]:
    return [c for c in cols if not is_availability_feature(c)]


# Everything derived from bookmaker prices. These are kept out of the model by
# default: the point of the project is a predictor that does not just echo the
# market, and the market is the benchmark it is scored against.
MARKET_PREFIXES = ("mkt_", "mov_")


def is_market_feature(col: str) -> bool:
    return col.startswith(MARKET_PREFIXES)


def drop_market(cols: list[str]) -> list[str]:
    return [c for c in cols if not is_market_feature(c)]
