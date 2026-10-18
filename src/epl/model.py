"""Model definitions and fitting.

Four estimators share one interface so the backtest can score them side by
side:

``xgb``      gradient-boosted trees on the full feature set (the main model)
``logreg``   multinomial logistic regression, the linear baseline
``elo``      logistic regression on the Elo rating difference alone
``market``   closing bookmaker odds, de-vigged; not fitted, used as a yardstick
``market_open`` the same bookmaker's *opening* price, to show how much the
             week's money improves the forecast
``base``     the historical home/draw/away base rate; the floor any model must clear
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_KINDS = ("xgb", "logreg", "elo", "market", "market_open", "base")

XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 3,
    "min_child_weight": 12,
    "subsample": 0.85,
    "colsample_bytree": 0.5,
    "reg_lambda": 4.0,
    "reg_alpha": 0.5,
    "gamma": 0.2,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 7,
}

ELO_FEATURES = ["elo_diff"]

# Optional exponential down-weighting of old matches, by age in seasons.
# Down-weighting was tried to correct the model's mild home-win bias and did
# not earn its keep: a half-life of 8 seasons moved RPS from 0.19924 to
# 0.19905 over 6,140 matches and left the bias at +1.4pp, so the default is
# equal weighting and this stays an opt-in knob.
DEFAULT_HALF_LIFE_SEASONS = None


def recency_weights(train: pd.DataFrame, half_life: float | None) -> np.ndarray | None:
    """Exponential decay in a match's age, measured in seasons."""
    if not half_life:
        return None
    age = train["season_start"].max() - train["season_start"]
    return np.power(0.5, age.to_numpy(dtype=float) / half_life)


@dataclass
class FittedModel:
    """A fitted estimator plus the columns it expects."""

    kind: str
    features: list[str]
    estimator: object | None = None
    base_rate: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "base":
            assert self.base_rate is not None
            return np.tile(self.base_rate, (len(df), 1))
        if self.kind == "market":
            return market_proba(df, fallback=self.base_rate)
        if self.kind == "market_open":
            return market_proba(
                df, fallback=self.base_rate, cols=("open_h", "open_d", "open_a")
            )
        assert self.estimator is not None
        proba = self.estimator.predict_proba(df[self.features])
        return np.asarray(proba, dtype=float)


def market_proba(
    df: pd.DataFrame,
    fallback: np.ndarray | None = None,
    cols: tuple[str, str, str] = ("odds_h", "odds_d", "odds_a"),
) -> np.ndarray:
    """De-vig decimal odds into probabilities (proportional normalisation)."""
    odds = df[list(cols)].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / odds
    total = np.nansum(inv, axis=1, keepdims=True)
    proba = inv / total
    bad = ~np.isfinite(proba).all(axis=1)
    if bad.any():
        fill = fallback if fallback is not None else np.array([0.45, 0.25, 0.30])
        proba[bad] = fill
    return proba


def _logreg_pipeline(c: float = 0.05) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c, max_iter=3000, solver="lbfgs", random_state=7
                ),
            ),
        ]
    )


def fit_model(
    train: pd.DataFrame,
    features: list[str],
    kind: str = "xgb",
    *,
    valid_fraction: float = 0.12,
    xgb_params: dict | None = None,
    half_life: float | None = DEFAULT_HALF_LIFE_SEASONS,
) -> FittedModel:
    """Fit one model on a date-ordered training slice.

    For ``xgb`` the most recent ``valid_fraction`` of the training window is
    held back as an early-stopping set, so the number of trees is chosen on
    data that comes after what the trees were grown on.
    """
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown model kind {kind!r}")

    y = train["y"].to_numpy(dtype=int)
    w = recency_weights(train, half_life)
    # The base rate is weighted too, so the "always predict the base rate"
    # floor reflects the same recent-seasons view as the trained models.
    counts = np.bincount(y, weights=w, minlength=3)
    base_rate = counts / counts.sum()

    if kind in ("base", "market", "market_open"):
        return FittedModel(kind=kind, features=[], base_rate=base_rate)

    if kind == "elo":
        model = _logreg_pipeline(c=1.0)
        model.fit(train[ELO_FEATURES], y)
        return FittedModel(kind=kind, features=list(ELO_FEATURES), estimator=model, base_rate=base_rate)

    if kind == "logreg":
        model = _logreg_pipeline()
        model.fit(train[features], y, clf__sample_weight=recency_weights(train, half_life))
        return FittedModel(kind=kind, features=list(features), estimator=model, base_rate=base_rate)

    # xgb
    params = {**XGB_PARAMS, **(xgb_params or {})}
    weights = recency_weights(train, half_life)
    n_valid = int(len(train) * valid_fraction)
    if n_valid >= 150:
        tr, va = train.iloc[:-n_valid], train.iloc[-n_valid:]
        w_tr = weights[:-n_valid] if weights is not None else None
        w_va = weights[-n_valid:] if weights is not None else None
        model = XGBClassifier(**params, early_stopping_rounds=120)
        model.fit(
            tr[features],
            tr["y"].to_numpy(dtype=int),
            sample_weight=w_tr,
            eval_set=[(va[features], va["y"].to_numpy(dtype=int))],
            sample_weight_eval_set=[w_va] if w_va is not None else None,
            verbose=False,
        )
        meta = {"best_iteration": int(model.best_iteration), "n_valid": n_valid}
    else:
        params = {**params, "n_estimators": 400}
        model = XGBClassifier(**params)
        model.fit(train[features], y, sample_weight=weights, verbose=False)
        meta = {"best_iteration": params["n_estimators"], "n_valid": 0}

    return FittedModel(
        kind=kind, features=list(features), estimator=model, base_rate=base_rate, meta=meta
    )


def feature_importance(model: FittedModel, top: int = 25) -> pd.DataFrame:
    """Gain-based importance for xgb, absolute coefficients for linear models."""
    if model.kind == "xgb":
        booster = model.estimator.get_booster()  # type: ignore[union-attr]
        gain = booster.get_score(importance_type="gain")
        names = {f"f{i}": c for i, c in enumerate(model.features)}
        rows = [{"feature": names.get(k, k), "importance": v} for k, v in gain.items()]
    elif model.kind in ("logreg", "elo"):
        clf = model.estimator.named_steps["clf"]  # type: ignore[union-attr]
        imp = np.abs(clf.coef_).mean(axis=0)
        rows = [{"feature": f, "importance": float(v)} for f, v in zip(model.features, imp, strict=True)]
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    return df.head(top).reset_index(drop=True)
