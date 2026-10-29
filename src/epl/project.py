"""Projected final table: where the model expects the season to end up.

The league table on the front page is the season as played. This is the
season as the model expects it to finish: every remaining fixture simulated
many times over, so a club's finishing position comes out as a distribution
rather than a single guess. That is the honest shape of the answer - "Arsenal
finish second" hides how close second was to fourth.

Two simplifications matter enough that the page states them:

* Each fixture's probabilities are the model's view *today*. Form is carried
  forward rather than rolled on, so a club in form stays in form across the
  whole simulation. Simulating the features too would compound the model's
  own error at every step, which buys precision that is not really there.
* Only the outcome is simulated, not the score, so clubs that finish level on
  points are separated by the goal difference they have now.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT

# Monte Carlo draws. Ten thousand puts the standard error on a 50% finishing
# probability at about half a point, which is well inside how precisely any
# of this can be stated.
DEFAULT_SIMS = 10_000
EURO_PLACES = 5
RELEGATION_PLACES = 3

COLUMNS = [
    "team", "played", "points", "goal_difference", "remaining",
    "proj_points", "proj_position", "pts_low", "pts_high",
    "p_title", "p_europe", "p_relegation",
]


def simulate_season(
    standings: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    n_sims: int = DEFAULT_SIMS,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate the rest of the season from the model's match probabilities.

    ``standings`` is the table as played (team, played, points,
    goal_difference); ``fixtures`` carries p_home / p_draw / p_away for every
    match still to come. The seed is fixed so a rebuild that changes no data
    produces no diff.
    """
    if standings.empty:
        return pd.DataFrame(columns=COLUMNS)

    fixtures = fixtures if fixtures is not None else pd.DataFrame()
    fx_teams: set[str] = set()
    if not fixtures.empty:
        fx_teams = set(fixtures["home_team"]) | set(fixtures["away_team"])
    teams = sorted(set(standings["team"]) | fx_teams)
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    base_points = np.zeros(n)
    base_gd = np.zeros(n)
    base_played = np.zeros(n, dtype=int)
    for row in standings.itertuples():
        i = index[row.team]
        base_points[i] = row.points
        base_gd[i] = row.goal_difference
        base_played[i] = row.played

    totals = np.tile(base_points, (n_sims, 1))
    remaining = np.zeros(n, dtype=int)

    if not fixtures.empty:
        probs = fixtures[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        # Renormalise rather than trust the caller: a row that sums to 0.999
        # would quietly bias every draw towards the away side.
        probs = probs / probs.sum(axis=1, keepdims=True)
        cutoff_home = probs[:, 0]
        cutoff_draw = probs[:, 0] + probs[:, 1]

        rng = np.random.default_rng(seed)
        draw = rng.random((n_sims, len(fixtures)))
        # 0 home win, 1 draw, 2 away win.
        outcome = (draw > cutoff_home).astype(np.int8) + (draw > cutoff_draw).astype(np.int8)
        home_points = np.where(outcome == 0, 3, np.where(outcome == 1, 1, 0))
        away_points = np.where(outcome == 2, 3, np.where(outcome == 1, 1, 0))

        for k, (home, away) in enumerate(
            zip(fixtures["home_team"], fixtures["away_team"], strict=True)
        ):
            h, a = index[home], index[away]
            totals[:, h] += home_points[:, k]
            totals[:, a] += away_points[:, k]
            remaining[h] += 1
            remaining[a] += 1

    # Goal difference breaks ties. Dividing by 1000 keeps it strictly smaller
    # than a single point, so it orders level clubs without ever outranking a
    # club that finished ahead on points.
    ranking_key = totals + base_gd[None, :] / 1000.0
    order = np.argsort(-ranking_key, axis=1, kind="stable")
    positions = np.empty_like(order)
    np.put_along_axis(
        positions, order, np.broadcast_to(np.arange(1, n + 1), (n_sims, n)), axis=1
    )

    out = pd.DataFrame({
        "team": teams,
        "played": base_played,
        "points": base_points.astype(int),
        "goal_difference": base_gd.astype(int),
        "remaining": remaining,
        "proj_points": totals.mean(axis=0),
        "proj_position": np.median(positions, axis=0),
        "pts_low": np.percentile(totals, 10, axis=0),
        "pts_high": np.percentile(totals, 90, axis=0),
        "p_title": (positions == 1).mean(axis=0),
        "p_europe": (positions <= EURO_PLACES).mean(axis=0),
        "p_relegation": (positions > n - RELEGATION_PLACES).mean(axis=0),
    })
    return (
        out.sort_values(["proj_points", "goal_difference"], ascending=False)
        .reset_index(drop=True)[COLUMNS]
    )


# ---------------------------------------------------------------------------
# the record of what the projection said, week by week
# ---------------------------------------------------------------------------
# A projection is only interesting next to the one before it. Kept in the repo
# beside the prediction log for the same reason: the record has to outlive the
# runner that produced it, or there is nothing to compare against.
PROJECTION_LOG = ROOT / "predictions" / "projection_log.csv"

LOG_KEY = ["recorded_at", "season_start", "team"]
LOG_COLUMNS = [
    *LOG_KEY, "position", "played", "points", "proj_points",
    "p_title", "p_europe", "p_relegation",
]


def empty_projection_log() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in LOG_COLUMNS})


def load_projection_log(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else PROJECTION_LOG
    if not path.exists():
        return empty_projection_log()
    df = pd.read_csv(path)
    for c in LOG_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], errors="coerce")
    df["season_start"] = pd.to_numeric(df["season_start"], errors="coerce").astype("Int64")
    for c in ("position", "played", "points"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("proj_points", "p_title", "p_europe", "p_relegation"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[LOG_COLUMNS].dropna(subset=["recorded_at", "team"])


def save_projection_log(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = Path(path) if path else PROJECTION_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[LOG_COLUMNS].sort_values(["recorded_at", "position"], kind="stable").copy()
    out["recorded_at"] = pd.to_datetime(out["recorded_at"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    return path


def append_projection(
    projection: pd.DataFrame,
    season_start: int,
    *,
    recorded_at: pd.Timestamp | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    """Record today's projection, one row per club.

    Snapshots are keyed by calendar day, so rebuilding twice in one day
    revises that day rather than stacking two points on the chart. Earlier
    days are never touched: the point of the log is that it is a record.
    """
    if projection is None or projection.empty:
        return load_projection_log(path)

    day = pd.Timestamp(recorded_at or pd.Timestamp.now()).normalize()
    fresh = pd.DataFrame({
        "recorded_at": day,
        "season_start": int(season_start),
        "team": projection["team"].to_numpy(),
        "position": np.arange(1, len(projection) + 1),
        "played": projection["played"].to_numpy(),
        "points": projection["points"].to_numpy(),
        "proj_points": projection["proj_points"].round(2).to_numpy(),
        "p_title": projection["p_title"].round(4).to_numpy(),
        "p_europe": projection["p_europe"].round(4).to_numpy(),
        "p_relegation": projection["p_relegation"].round(4).to_numpy(),
    })

    combined = pd.concat([load_projection_log(path), fresh], ignore_index=True)
    combined["recorded_at"] = pd.to_datetime(combined["recorded_at"])
    # keep="last" so a rebuild revises today rather than being ignored.
    combined = combined.drop_duplicates(LOG_KEY, keep="last")
    save_projection_log(combined, path)
    return combined


def projection_history(log: pd.DataFrame, season_start: int | None = None) -> pd.DataFrame:
    """One row per snapshot per club, oldest first, for one season."""
    if log.empty:
        return log
    out = log
    if season_start is not None:
        out = out[out["season_start"] == season_start]
    return out.sort_values(["recorded_at", "position"], kind="stable").reset_index(drop=True)
