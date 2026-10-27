"""A persistent record of predictions made, and how they turned out.

The backtest answers "how would this model have done". This file answers the
harder and more honest question: "what did it actually say, before the match,
and was it right". Predictions are appended to a CSV the moment they are
published and are never rewritten afterwards — results are filled in beside
them once the match is played, so the record cannot be quietly improved by a
later model.

Two kinds of row live in the log, kept apart by the ``source`` column:

``published``  logged from the live site before kick-off. The real track record.
``backtest``   backfilled for matches that were played before the log existed,
               from a model trained only on earlier seasons. Genuinely
               out-of-sample, but not something anyone saw at the time.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import INT_TO_LABEL, ROOT

log = logging.getLogger(__name__)

LOG_PATH = ROOT / "predictions" / "log.csv"

# (season_start, home_team, away_team) identifies a fixture: each ordered pair
# meets once per season.
KEY = ["season_start", "home_team", "away_team"]

COLUMNS = [
    *KEY, "date", "p_h", "p_d", "p_a", "predicted_at", "source",
    "home_goals", "away_goals", "actual", "predicted", "correct",
]


def empty_log() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})


def load_log(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else LOG_PATH
    if not path.exists():
        return empty_log()
    df = pd.read_csv(path)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("p_h", "p_d", "p_a", "home_goals", "away_goals", "correct"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["season_start"] = pd.to_numeric(df["season_start"], errors="coerce").astype("Int64")
    return df[COLUMNS]


def save_log(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = Path(path) if path else LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[COLUMNS].sort_values(["date", "home_team"], kind="stable")
    out.to_csv(path, index=False)
    return path


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture, with a live prediction always beating a backfill."""
    rank = df["source"].map({"published": 0, "backtest": 1}).fillna(2)
    return (
        df.assign(_rank=rank)
        .sort_values(["_rank", "predicted_at"], na_position="last")
        .drop_duplicates(KEY, keep="first")
        .drop(columns="_rank")
    )


def append_predictions(
    logged: pd.DataFrame, fixtures: pd.DataFrame, *, source: str = "published"
) -> pd.DataFrame:
    """Add predictions for fixtures not already recorded.

    An existing row is never overwritten: whatever was said first is what the
    record keeps.
    """
    if fixtures.empty:
        return logged
    need = {"home_team", "away_team", "p_home", "p_draw", "p_away", "date"}
    missing = need - set(fixtures.columns)
    if missing:
        raise ValueError(f"fixtures are missing {sorted(missing)}")

    season_start = fixtures.get("season_start")
    if season_start is None:
        # Fixtures from August onward belong to the season that starts that year.
        d = pd.to_datetime(fixtures["date"])
        season_start = np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1)

    fresh = pd.DataFrame(
        {
            "season_start": pd.array(season_start, dtype="Int64"),
            "home_team": fixtures["home_team"].astype(str).to_numpy(),
            "away_team": fixtures["away_team"].astype(str).to_numpy(),
            "date": pd.to_datetime(fixtures["date"]).to_numpy(),
            "p_h": fixtures["p_home"].to_numpy(dtype=float),
            "p_d": fixtures["p_draw"].to_numpy(dtype=float),
            "p_a": fixtures["p_away"].to_numpy(dtype=float),
            "predicted_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "source": source,
        }
    )
    for c in ("home_goals", "away_goals", "actual", "predicted", "correct"):
        fresh[c] = pd.NA

    if logged.empty:
        combined = fresh[COLUMNS]
    else:
        seen = set(map(tuple, logged[KEY].astype(str).to_numpy()))
        keep = [tuple(r) not in seen for r in fresh[KEY].astype(str).to_numpy()]
        added = int(sum(keep))
        if added:
            log.info("logging %d new prediction(s)", added)
        combined = pd.concat([logged, fresh.loc[keep, COLUMNS]], ignore_index=True)
    return _dedupe(combined).reset_index(drop=True)


def score_log(logged: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Fill in results for logged predictions whose match has now been played."""
    if logged.empty:
        return logged
    played = matches.dropna(subset=["result"])[
        [*KEY, "home_goals", "away_goals", "result"]
    ].copy()
    played["home_team"] = played["home_team"].astype(str)
    played["away_team"] = played["away_team"].astype(str)
    played["season_start"] = pd.to_numeric(played["season_start"]).astype("Int64")
    played = played.drop_duplicates(KEY)

    out = logged.copy()
    out["home_team"] = out["home_team"].astype(str)
    out["away_team"] = out["away_team"].astype(str)
    merged = out.drop(columns=["home_goals", "away_goals"]).merge(
        played, on=KEY, how="left", validate="one_to_one"
    )

    probs = merged[["p_h", "p_d", "p_a"]].to_numpy(dtype=float)
    merged["predicted"] = [INT_TO_LABEL[i] for i in probs.argmax(axis=1)]
    merged["actual"] = merged["result"]
    done = merged["actual"].notna()
    merged.loc[done, "correct"] = (
        merged.loc[done, "predicted"] == merged.loc[done, "actual"]
    ).astype(int)
    merged.loc[~done, "correct"] = pd.NA
    return merged.drop(columns=["result"])[COLUMNS]


def backfill_from_backtest(
    logged: pd.DataFrame, preds: pd.DataFrame, *, model: str = "xgb", season: str | None = None
) -> pd.DataFrame:
    """Seed the log with out-of-sample predictions for already-played matches.

    Without this the record would start empty and stay nearly empty for weeks.
    These rows are marked ``backtest`` so they are never passed off as
    something the site published at the time.
    """
    sub = preds[preds["model"] == model]
    if season is not None:
        sub = sub[sub["season"] == season]
    if sub.empty:
        return logged
    as_fixtures = pd.DataFrame(
        {
            "season_start": sub["season_start"].to_numpy(),
            "date": sub["date"].to_numpy(),
            "home_team": sub["home_team"].to_numpy(),
            "away_team": sub["away_team"].to_numpy(),
            "p_home": sub["p_h"].to_numpy(),
            "p_draw": sub["p_d"].to_numpy(),
            "p_away": sub["p_a"].to_numpy(),
        }
    )
    return append_predictions(logged, as_fixtures, source="backtest")


def summarise_log(logged: pd.DataFrame) -> dict:
    """Headline accuracy over the scored part of the record."""
    scored = logged[logged["correct"].notna()]
    if scored.empty:
        return {"n": 0}
    probs = scored[["p_h", "p_d", "p_a"]].to_numpy(dtype=float)
    idx = {"H": 0, "D": 1, "A": 2}
    hit = np.array([probs[i, idx[a]] for i, a in enumerate(scored["actual"])])
    by_source = (
        scored.groupby("source")["correct"].agg(["size", "mean"]).to_dict("index")
    )
    return {
        "n": int(len(scored)),
        "correct": int(scored["correct"].sum()),
        "accuracy": float(scored["correct"].mean()),
        # Mean probability assigned to what actually happened: a finer measure
        # than accuracy, since it rewards being confident and right.
        "mean_prob_actual": float(hit.mean()),
        "pending": int(logged["correct"].isna().sum()),
        "by_source": by_source,
    }
