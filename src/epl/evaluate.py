"""Time-aware backtesting.

A match is only ever predicted by a model trained on matches that finished
before it. The default protocol is season-by-season: to score season S, fit on
everything up to the day before S starts, then predict all 380 matches. That
mirrors how the model would be used in practice and makes the per-season
numbers comparable.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import INT_TO_LABEL
from .metrics import evaluate
from .model import MODEL_KINDS, fit_model

log = logging.getLogger(__name__)


def walk_forward(
    df: pd.DataFrame,
    features: list[str],
    kinds: tuple[str, ...] = ("xgb", "logreg", "elo", "market", "base"),
    *,
    first_test_season: int = 2010,
    min_train_matches: int = 2000,
    refit_every_days: int | None = None,
    half_life: float | None = None,
) -> pd.DataFrame:
    """Score every season from ``first_test_season`` onward out-of-sample.

    ``refit_every_days`` refits mid-season at that cadence (e.g. 30 for
    roughly monthly); ``None`` fits once per season.
    """
    for kind in kinds:
        if kind not in MODEL_KINDS:
            raise ValueError(f"unknown model kind {kind!r}")

    df = df.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
    seasons = sorted(s for s in df["season_start"].unique() if s >= first_test_season)
    frames: list[pd.DataFrame] = []

    for season in seasons:
        test_all = df[df["season_start"] == season]
        if test_all.empty:
            continue
        # Mid-season refits split the season into consecutive date blocks.
        if refit_every_days:
            start = test_all["date"].min()
            block = ((test_all["date"] - start).dt.days // refit_every_days).to_numpy()
        else:
            block = np.zeros(len(test_all), dtype=int)

        for b in np.unique(block):
            test = test_all[block == b]
            cutoff = test["date"].min()
            train = df[df["date"] < cutoff]
            if len(train) < min_train_matches:
                log.warning("skipping %s block %s: only %d training matches", season, b, len(train))
                continue

            for kind in kinds:
                model = fit_model(train, features, kind, half_life=half_life)
                proba = model.predict_proba(test)
                frames.append(
                    pd.DataFrame(
                        {
                            "match_id": test["match_id"].to_numpy(),
                            "date": test["date"].to_numpy(),
                            "season": test["season"].to_numpy(),
                            "season_start": test["season_start"].to_numpy(),
                            "home_team": test["home_team"].to_numpy(),
                            "away_team": test["away_team"].to_numpy(),
                            "matchweek": test["matchweek"].to_numpy(),
                            "y": test["y"].to_numpy(dtype=int),
                            "model": kind,
                            "p_h": proba[:, 0],
                            "p_d": proba[:, 1],
                            "p_a": proba[:, 2],
                            "n_train": len(train),
                        }
                    )
                )
            log.info(
                "season %s block %s: trained on %d matches, scored %d",
                season, b, len(train), len(test),
            )

    if not frames:
        raise RuntimeError("no seasons were scored; check first_test_season / min_train_matches")
    preds = pd.concat(frames, ignore_index=True)
    preds["pred"] = preds[["p_h", "p_d", "p_a"]].to_numpy().argmax(axis=1)
    preds["pred_label"] = preds["pred"].map(INT_TO_LABEL)
    preds["actual_label"] = preds["y"].map(INT_TO_LABEL)
    preds["correct"] = (preds["pred"] == preds["y"]).astype(int)
    return preds


def _proba(sub: pd.DataFrame) -> np.ndarray:
    return sub[["p_h", "p_d", "p_a"]].to_numpy(dtype=float)


def summarise(preds: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Metrics per model, optionally grouped (e.g. ``by=['season']``)."""
    keys = ["model"] + (by or [])
    rows = []
    for key, sub in preds.groupby(keys, sort=True):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        rec.update(evaluate(sub["y"].to_numpy(), _proba(sub)))
        rows.append(rec)
    out = pd.DataFrame(rows)
    sort_cols = (by or []) + ["rps"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def compare_on_common_rows(preds: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Compare models on the subset of matches all of them scored.

    The market baseline is missing for the two oldest seasons, so a naive
    comparison would score it on an easier sample.
    """
    sub = preds[preds["model"].isin(models)]
    counts = sub.groupby("match_id")["model"].nunique()
    common = counts[counts == len(models)].index
    return summarise(sub[sub["match_id"].isin(common)])


def season_tracking(preds: pd.DataFrame, season: str, model: str = "xgb") -> pd.DataFrame:
    """Cumulative accuracy and log loss through a single season, by matchweek.

    This answers the "does it get better as the season goes on?" question: as
    more matches are played the form features fill in, so the within-season
    trend is informative even though the model itself is not being refitted.
    """
    sub = preds[(preds["season"] == season) & (preds["model"] == model)].sort_values("date")
    if sub.empty:
        raise ValueError(f"no predictions for season {season!r} / model {model!r}")
    sub = sub.copy()
    sub["week"] = np.ceil(sub["matchweek"]).astype(int).clip(lower=0)
    rows = []
    for week, grp in sub.groupby("week"):
        upto = sub[sub["week"] <= week]
        rows.append(
            {
                "week": int(week),
                "matches": int(len(grp)),
                "week_accuracy": float(grp["correct"].mean()),
                "cum_matches": int(len(upto)),
                "cum_accuracy": float(upto["correct"].mean()),
                "cum_log_loss": evaluate(upto["y"].to_numpy(), _proba(upto))["log_loss"],
                "cum_rps": evaluate(upto["y"].to_numpy(), _proba(upto))["rps"],
            }
        )
    return pd.DataFrame(rows)


def confusion(preds: pd.DataFrame, model: str = "xgb") -> pd.DataFrame:
    sub = preds[preds["model"] == model]
    return pd.crosstab(sub["actual_label"], sub["pred_label"], dropna=False).reindex(
        index=["H", "D", "A"], columns=["H", "D", "A"], fill_value=0
    )
