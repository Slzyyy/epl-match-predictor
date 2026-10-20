"""Fit on all completed matches and predict upcoming fixtures.

Upcoming fixtures have no statistics of their own, so they are appended to the
history as rows with blank results and run through the same feature pipeline.
That way an unplayed match picks up exactly the pre-match features a played
one would have had: current Elo, current form, rest days, and so on.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import RESULTS
from .data import load_fixtures, load_matches
from .features import build_features, core_feature_columns, feature_columns
from .model import fit_model
from .lineups import attach_availability, load_availability
from .xg import attach_xg, load_xg

log = logging.getLogger(__name__)


def _append_fixtures(matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Stack fixtures onto history with NaN results, keeping one id space."""
    fx = fixtures.copy()
    fx["season"] = matches["season"].iloc[-1]
    fx["season_start"] = matches["season_start"].iloc[-1]
    fx["result"] = pd.NA
    for col in matches.columns:
        if col not in fx.columns:
            fx[col] = np.nan
    fx = fx[matches.columns]
    fx["match_id"] = range(len(matches), len(matches) + len(fx))
    return pd.concat([matches, fx], ignore_index=True)


def predict_fixtures(
    *,
    kind: str = "xgb",
    refresh: bool = True,
    include_odds: bool = False,
    limit_days: int | None = 14,
    feature_set: str = "core",
    fixtures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one row per upcoming fixture with H/D/A probabilities."""
    matches = load_matches(refresh=refresh)
    matches = attach_xg(matches, load_xg(refresh=refresh))
    fx = load_fixtures(refresh=refresh) if fixtures is None else fixtures.copy()
    if fx.empty:
        return pd.DataFrame(columns=["date", "home_team", "away_team", *RESULTS])

    # Drop anything already played. The fixtures feed keeps recent rounds for a
    # while, and a cached copy can be several days stale if the source is down,
    # so without this the page can show matches that have already finished.
    today = pd.Timestamp.now().normalize()
    fx = fx[fx["date"] >= today]
    if fx.empty:
        log.warning("no fixtures dated today or later; the feed may be stale")
        return pd.DataFrame(columns=["date", "home_team", "away_team", *RESULTS])

    if limit_days is not None:
        horizon = fx["date"].min() + pd.Timedelta(days=limit_days)
        fx = fx[fx["date"] <= horizon]

    known = set(matches["home_team"]) | set(matches["away_team"])
    unknown = sorted((set(fx["home_team"]) | set(fx["away_team"])) - known)
    if unknown:
        log.warning("fixtures contain teams with no match history: %s", unknown)

    combined = _append_fixtures(matches, fx)
    # Availability is joined after the fixtures are appended so upcoming rows
    # pick up each team's trailing state.
    combined = attach_availability(combined, load_availability())
    feats = build_features(combined, include_odds=include_odds)
    cols = core_feature_columns(feats) if feature_set == "core" else feature_columns(feats)
    if not include_odds:
        cols = [c for c in cols if not c.startswith("mkt_")]

    train = feats[feats["result"].notna()]
    future = feats[feats["result"].isna()].copy()
    if future.empty:
        return pd.DataFrame(columns=["date", "home_team", "away_team", *RESULTS])

    model = fit_model(train, cols, kind)
    proba = model.predict_proba(future)

    out = future[["date", "home_team", "away_team", "home_elo", "away_elo"]].copy()
    out["p_home"], out["p_draw"], out["p_away"] = proba[:, 0], proba[:, 1], proba[:, 2]
    out["pick"] = np.array(RESULTS)[proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1)
    # Fair odds are the inverse of the model probability: what price the model
    # thinks each outcome is worth.
    for name, col in zip(RESULTS, ("p_home", "p_draw", "p_away"), strict=True):
        out[f"fair_odds_{name}"] = (1.0 / out[col]).round(2)
    if {"odds_h", "odds_d", "odds_a"} <= set(future.columns) and future["odds_h"].notna().any():
        out["odds_h"] = future["odds_h"].to_numpy()
        out["odds_d"] = future["odds_d"].to_numpy()
        out["odds_a"] = future["odds_a"].to_numpy()
    out.attrs["model_meta"] = {"kind": kind, "n_train": len(train), **model.meta}
    return out.sort_values("date").reset_index(drop=True)
