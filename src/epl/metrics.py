"""Scoring rules for three-way match predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RESULTS


def _onehot(y: np.ndarray) -> np.ndarray:
    out = np.zeros((len(y), 3))
    out[np.arange(len(y)), y.astype(int)] = 1.0
    return out


def log_loss(y: np.ndarray, proba: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(proba, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.log(p[np.arange(len(y)), y.astype(int)]).mean())


def brier(y: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass Brier score (mean squared error over the 3 outcomes)."""
    return float(((proba - _onehot(y)) ** 2).sum(axis=1).mean())


def rps(y: np.ndarray, proba: np.ndarray) -> float:
    """Ranked probability score, the standard football forecasting metric.

    Outcomes are treated as ordered (home win > draw > away win), so predicting
    a draw when the home side wins is penalised less than predicting an away
    win. Lower is better; 0 is perfect.
    """
    cum_p = np.cumsum(proba, axis=1)[:, :-1]
    cum_y = np.cumsum(_onehot(y), axis=1)[:, :-1]
    return float(((cum_p - cum_y) ** 2).sum(axis=1).mean() / (proba.shape[1] - 1))


def accuracy(y: np.ndarray, proba: np.ndarray) -> float:
    return float((proba.argmax(axis=1) == y.astype(int)).mean())


def evaluate(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "accuracy": accuracy(y, proba),
        "log_loss": log_loss(y, proba),
        "rps": rps(y, proba),
        "brier": brier(y, proba),
    }


def calibration_table(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability table: predicted vs realised frequency, pooled over outcomes."""
    rows = []
    onehot = _onehot(y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    for k, name in enumerate(RESULTS):
        p = proba[:, k]
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        for b in range(bins):
            mask = idx == b
            if not mask.any():
                continue
            rows.append(
                {
                    "outcome": name,
                    "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                    "n": int(mask.sum()),
                    "predicted": float(p[mask].mean()),
                    "realised": float(onehot[mask, k].mean()),
                }
            )
    return pd.DataFrame(rows)
