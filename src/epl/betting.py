"""Value-betting backtest.

A bet is placed on an outcome when the model thinks it is more likely than the
price implies. The test that matters is whether that produces a positive
return after the bookmaker's margin, which is a much harder bar than scoring
well on RPS.

Two details decide whether this kind of backtest is honest:

* **Compare against the vigged implied probability**, 1/price, not the
  de-vigged one. The de-vigged probabilities sum to 1 and ignore the margin,
  so comparing against them invents an edge on roughly every match. To profit
  you need ``p_model * price > 1``, and that is the test used here.
* **Say which price you could actually have got.** The same model is a loser
  at a sharp bookmaker's closing price and a winner at the best price across
  every bookmaker, because the latter has an overround below 1 and is partly
  unobtainable. Every scenario here is reported with a control that shows what
  a bettor with no skill would have made at the same prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RESULTS

# Price scenarios, from most realistic to most optimistic.
PRICE_SCENARIOS = {
    "close": ("close_h", "close_d", "close_a"),
    "open": ("open_h", "open_d", "open_a"),
    "best_close": ("best_close_h", "best_close_d", "best_close_a"),
    "best_open": ("best_open_h", "best_open_d", "best_open_a"),
}

STAKE_RULES = ("flat", "kelly")


def _long_bets(preds: pd.DataFrame, matches: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """One row per (match, outcome) with the model's probability and the price."""
    if scenario not in PRICE_SCENARIOS:
        raise ValueError(f"unknown price scenario {scenario!r}")
    cols = PRICE_SCENARIOS[scenario]
    missing = [c for c in cols if c not in matches.columns]
    if missing:
        raise ValueError(f"matches frame has no {missing}")

    prices = matches.set_index("match_id")[list(cols)]
    sub = preds.join(prices, on="match_id")
    rows = []
    for i, name in enumerate(RESULTS):
        rows.append(
            pd.DataFrame(
                {
                    "match_id": sub["match_id"].to_numpy(),
                    "season": sub["season"].to_numpy(),
                    "date": sub["date"].to_numpy(),
                    "model": sub["model"].to_numpy() if "model" in sub else "model",
                    "outcome": name,
                    "price": sub[cols[i]].to_numpy(dtype=float),
                    "p_model": sub[["p_h", "p_d", "p_a"]].to_numpy(dtype=float)[:, i],
                    "won": (sub["y"].to_numpy() == i).astype(float),
                }
            )
        )
    long = pd.concat(rows, ignore_index=True)
    long = long[long["price"].notna() & (long["price"] > 1.0) & long["p_model"].notna()]
    # Expected profit per unit staked. Positive means the model sees value.
    long["edge"] = long["p_model"] * long["price"] - 1.0
    return long


def find_bets(
    preds: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    scenario: str = "close",
    threshold: float = 0.0,
    stake_rule: str = "flat",
    kelly_fraction: float = 0.25,
    max_stake: float = 5.0,
    bet_everything: bool = False,
) -> pd.DataFrame:
    """Select bets and settle them.

    ``bet_everything`` ignores the model and stakes every outcome of every
    match. That is the no-skill control: its return is whatever the price
    scenario gives away by construction.
    """
    if stake_rule not in STAKE_RULES:
        raise ValueError(f"unknown stake rule {stake_rule!r}")

    long = _long_bets(preds, matches, scenario)
    bets = long if bet_everything else long[long["edge"] > threshold].copy()
    bets = bets.copy()

    if stake_rule == "flat" or bet_everything:
        bets["stake"] = 1.0
    else:
        # Kelly: edge divided by net odds, scaled down and capped. Full Kelly
        # on estimated probabilities over-bets badly, hence the fraction.
        net = bets["price"] - 1.0
        bets["stake"] = (kelly_fraction * bets["edge"] / net).clip(0.0, max_stake)

    bets["profit"] = np.where(
        bets["won"] > 0, bets["stake"] * (bets["price"] - 1.0), -bets["stake"]
    )
    bets["ret"] = bets["profit"] / bets["stake"].replace(0, np.nan)
    return bets.reset_index(drop=True)


def _bootstrap_roi(bets: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile CI for ROI, resampling whole matches.

    Bets on the same match are mutually exclusive and so negatively
    correlated; resampling individual bets would understate the variance.
    """
    if bets.empty:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    groups = bets.groupby("match_id")[["stake", "profit"]].sum()
    stake = groups["stake"].to_numpy()
    profit = groups["profit"].to_numpy()
    n = len(stake)
    idx = rng.integers(0, n, size=(n_boot, n))
    roi = profit[idx].sum(axis=1) / np.maximum(stake[idx].sum(axis=1), 1e-9)
    return (float(np.percentile(roi, 2.5)), float(np.percentile(roi, 97.5)))


def summarise_bets(bets: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Headline betting statistics with a match-clustered confidence interval."""
    if bets.empty:
        return {"n_bets": 0, "staked": 0.0, "profit": 0.0, "roi": np.nan,
                "hit_rate": np.nan, "t_stat": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    staked = float(bets["stake"].sum())
    profit = float(bets["profit"].sum())
    roi = profit / staked

    # Cluster by match for the standard error, same reasoning as the bootstrap.
    per_match = bets.groupby("match_id")[["stake", "profit"]].sum()
    n = len(per_match)
    centred = per_match["profit"] - roi * per_match["stake"]
    se = float(centred.std(ddof=1) * np.sqrt(n) / staked) if n > 1 else np.nan
    lo, hi = _bootstrap_roi(bets, n_boot, seed)
    return {
        "n_bets": int(len(bets)),
        "n_matches": n,
        "staked": staked,
        "profit": profit,
        "roi": roi,
        "hit_rate": float((bets["won"] > 0).mean()),
        "avg_price": float(bets["price"].mean()),
        "avg_edge": float(bets["edge"].mean()),
        "t_stat": float(roi / se) if se and np.isfinite(se) and se > 0 else np.nan,
        "ci_low": lo,
        "ci_high": hi,
    }


def sweep(
    preds: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    models: list[str] | None = None,
    scenarios: list[str] | None = None,
    thresholds: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10),
    stake_rule: str = "flat",
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Every model x price scenario x edge threshold, plus the no-skill control."""
    models = models or sorted(preds["model"].unique())
    scenarios = scenarios or ["close"]
    rows = []
    for scenario in scenarios:
        # No-skill control: stake everything at these prices.
        one = preds[preds["model"] == models[0]]
        ctrl = find_bets(one, matches, scenario=scenario, bet_everything=True)
        rows.append({"model": "(stake everything)", "scenario": scenario,
                     "threshold": np.nan, **summarise_bets(ctrl, n_boot=n_boot)})
        for model in models:
            sub = preds[preds["model"] == model]
            for thr in thresholds:
                bets = find_bets(
                    sub, matches, scenario=scenario, threshold=thr, stake_rule=stake_rule
                )
                rows.append({"model": model, "scenario": scenario, "threshold": thr,
                             **summarise_bets(bets, n_boot=n_boot)})
    return pd.DataFrame(rows)


def by_season(bets: pd.DataFrame) -> pd.DataFrame:
    """Season-by-season profit, to show how unstable the ROI is."""
    g = bets.groupby("season")[["stake", "profit"]].sum()
    g["roi"] = g["profit"] / g["stake"]
    g["n_bets"] = bets.groupby("season").size()
    return g.reset_index()
