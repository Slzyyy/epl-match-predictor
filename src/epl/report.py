"""Charts for the backtest: season tracking, calibration, model comparison."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import REPORTS
from .evaluate import season_tracking, summarise
from .metrics import calibration_table

# Brand-neutral, colour-blind-safe series colours.
PALETTE = {
    "xgb": "#2f6fdb",
    "logreg": "#e07b39",
    "elo": "#49a078",
    "market": "#8a6fbf",
    "base": "#9aa0a6",
}
GRID = {"color": "#d8dade", "linewidth": 0.6}


def _style(ax, *, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_season_tracking(preds: pd.DataFrame, season: str, models: list[str]) -> str:
    """Cumulative accuracy and RPS through one season, model by model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for model in models:
        try:
            t = season_tracking(preds, season, model)
        except ValueError:
            continue
        colour = PALETTE.get(model, "#444444")
        axes[0].plot(t["week"], t["cum_accuracy"], label=model, color=colour, linewidth=1.8)
        axes[1].plot(t["week"], t["cum_rps"], label=model, color=colour, linewidth=1.8)

    _style(axes[0], title=f"Cumulative accuracy, {season}", xlabel="matchweek", ylabel="accuracy")
    _style(axes[1], title=f"Cumulative RPS (lower is better), {season}", xlabel="matchweek", ylabel="RPS")
    axes[0].legend(frameon=False, fontsize=8)
    dest = REPORTS / f"tracking_{season}.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return str(dest)


def plot_calibration(preds: pd.DataFrame, model: str = "xgb") -> str:
    """Predicted vs realised frequency; the diagonal is perfect calibration."""
    sub = preds[preds["model"] == model]
    cal = calibration_table(sub["y"].to_numpy(), sub[["p_h", "p_d", "p_a"]].to_numpy())
    cal = cal[cal["n"] >= 25]
    fig, ax = plt.subplots(figsize=(5.2, 5), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="#9aa0a6", linestyle="--", linewidth=1)
    for outcome, colour in zip(("H", "D", "A"), ("#2f6fdb", "#e07b39", "#49a078"), strict=True):
        part = cal[cal["outcome"] == outcome]
        ax.plot(part["predicted"], part["realised"], "o-", label=outcome, color=colour, linewidth=1.6, markersize=5)
    _style(ax, title=f"Calibration, {model}", xlabel="predicted probability", ylabel="realised frequency")
    ax.grid(True, axis="both", **GRID)
    ax.legend(frameon=False, fontsize=8, title="outcome", title_fontsize=8)
    dest = REPORTS / f"calibration_{model}.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return str(dest)


def plot_season_comparison(preds: pd.DataFrame, models: list[str]) -> str:
    """Per-season RPS for each model, so single-season noise is visible."""
    per = summarise(preds, by=["season"])
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    seasons = sorted(per["season"].unique())
    x = np.arange(len(seasons))
    for model in models:
        part = per[per["model"] == model].set_index("season").reindex(seasons)
        ax.plot(x, part["rps"], "o-", label=model, color=PALETTE.get(model, "#444"), linewidth=1.6, markersize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons, rotation=45, ha="right")
    _style(ax, title="Out-of-sample RPS by season (lower is better)", xlabel="", ylabel="RPS")
    ax.legend(frameon=False, fontsize=8, ncols=len(models))
    dest = REPORTS / "rps_by_season.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return str(dest)


def write_all(preds: pd.DataFrame, models: list[str], season: str | None = None) -> list[str]:
    season = season or str(preds["season"].max())
    paths = [
        plot_season_comparison(preds, models),
        plot_calibration(preds, models[0]),
        plot_season_tracking(preds, season, models),
    ]
    return paths
