"""Command line interface.

    python -m epl.cli fetch                 # download / refresh match data
    python -m epl.cli backtest              # walk-forward evaluation
    python -m epl.cli track --season 2025-26
    python -m epl.cli predict               # probabilities for upcoming fixtures
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

from .config import DATA_PROCESSED, FIRST_SEASON, LAST_SEASON, REPORTS
from .betting import PRICE_SCENARIOS, by_season, find_bets, summarise_bets, sweep
from .data import load_matches
from .evaluate import (
    compare_on_common_rows,
    confusion,
    season_tracking,
    summarise,
    walk_forward,
)
from .features import (build_features, core_feature_columns, drop_availability,
                       drop_market, drop_xg, feature_columns)
from .metrics import calibration_table
from .model import MODEL_KINDS, feature_importance, fit_model
from .predict import predict_fixtures
from .lineups import attach_availability, download_season_rosters, load_availability
from .xg import attach_xg, load_xg
from .report import write_all
from .site import SITE_DIR, build as build_site
from .track import (LOG_PATH, append_predictions, backfill_from_backtest,
                    load_log, save_log, score_log, summarise_log)

FEATURES_PATH = DATA_PROCESSED / "features.pkl"
PREDS_PATH = DATA_PROCESSED / "backtest_preds.pkl"


def _wide(df: pd.DataFrame) -> str:
    with pd.option_context("display.width", 200, "display.max_columns", 60):
        return df.to_string(index=False)


def build_feature_table(
    *, refresh: bool, include_odds: bool, first: int, last: int,
    with_xg: bool = True, cache: bool = True,
) -> pd.DataFrame:
    matches = load_matches(first, last, refresh=refresh)
    if with_xg:
        matches = attach_xg(matches, load_xg(refresh=refresh))
        matches = attach_availability(matches, load_availability())
    feats = build_features(matches, include_odds=include_odds)
    if cache:
        feats.to_pickle(FEATURES_PATH)
    return feats


def _select_features(feats: pd.DataFrame, args) -> list[str]:
    cols = core_feature_columns(feats) if args.feature_set == "core" else feature_columns(feats)
    if not args.use_odds:
        cols = drop_market(cols)
    if getattr(args, "no_xg", False):
        cols = drop_xg(cols)
    if getattr(args, "no_avail", False):
        cols = drop_availability(cols)
    return cols


def _load_features(args) -> pd.DataFrame:
    if FEATURES_PATH.exists() and not args.refresh:
        return pd.read_pickle(FEATURES_PATH)
    return build_feature_table(
        refresh=args.refresh, include_odds=True, first=args.first_season, last=args.last_season
    )


def cmd_fetch(args) -> int:
    if getattr(args, "with_squads", False):
        # One request per match against a free site: slow, and opt-in because
        # the features it produces did not measurably improve the model.
        from .xg import FIRST_XG_SEASON

        for year in range(max(args.first_season, FIRST_XG_SEASON), args.last_season + 1):
            print(f"fetching rosters for {year} ...", flush=True)
            download_season_rosters(year)

    feats = build_feature_table(
        refresh=True, include_odds=True, first=args.first_season, last=args.last_season
    )
    n_xg = int(feats["home_xg"].notna().sum()) if "home_xg" in feats.columns else 0
    n_av = (int(feats["home_absent_minutes_share_2"].notna().sum())
            if "home_absent_minutes_share_2" in feats.columns else 0)
    print(f"{len(feats)} matches, {feats['season'].nunique()} seasons, "
          f"{len(feature_columns(feats))} features, {n_xg} with xG, {n_av} with squad data")
    print(f"seasons: {feats['season'].min()} .. {feats['season'].max()}")
    print(f"cached -> {FEATURES_PATH}")
    return 0


def cmd_backtest(args) -> int:
    feats = _load_features(args)
    cols = _select_features(feats, args)
    print(f"{len(feats)} matches, {len(cols)} features ({args.feature_set}), models={args.models}")

    preds = walk_forward(
        feats,
        cols,
        tuple(args.models),
        first_test_season=args.first_test_season,
        refit_every_days=args.refit_days,
    )
    preds.to_pickle(PREDS_PATH)

    print("\n=== overall, out-of-sample ===")
    print(_wide(summarise(preds)))
    if len(args.models) > 1:
        print("\n=== same matches for every model ===")
        print(_wide(compare_on_common_rows(preds, list(args.models))))
    print(f"\n=== per season ({args.main_model}) ===")
    per = summarise(preds, by=["season"])
    print(_wide(per[per["model"] == args.main_model].drop(columns=["model"])))
    print(f"\n=== confusion, {args.main_model} (rows actual, cols predicted) ===")
    print(confusion(preds, args.main_model).to_string())
    print(f"\n=== calibration, {args.main_model} ===")
    sub = preds[preds["model"] == args.main_model]
    cal = calibration_table(sub["y"].to_numpy(), sub[["p_h", "p_d", "p_a"]].to_numpy())
    print(_wide(cal[cal["n"] >= 25]))

    if not args.no_charts:
        for path in write_all(preds, list(args.models)):
            print(f"chart -> {path}")

    csv = REPORTS / "backtest_summary.csv"
    summarise(preds, by=["season"]).to_csv(csv, index=False)
    preds.to_csv(REPORTS / "backtest_predictions.csv", index=False)
    print(f"\nwrote {csv} and {REPORTS / 'backtest_predictions.csv'}")
    print(f"predictions cached -> {PREDS_PATH}")
    return 0


def cmd_log(args) -> int:
    """Record the current predictions and score any that have now been played."""
    logged = load_log()
    before = len(logged)

    if not args.no_predict:
        fixtures = predict_fixtures(
            kind=args.main_model, refresh=not args.no_refresh,
            include_odds=args.use_odds, limit_days=args.days,
            feature_set=args.feature_set,
        )
        logged = append_predictions(logged, fixtures, source="published")

    if args.backfill and PREDS_PATH.exists():
        preds = pd.read_pickle(PREDS_PATH)
        season = args.season or str(preds["season"].max())
        logged = backfill_from_backtest(logged, preds, model=args.main_model, season=season)

    feats = _load_features(args)
    logged = score_log(logged, feats)
    path = save_log(logged)

    s = summarise_log(logged)
    print(f"{len(logged)} predictions logged ({len(logged) - before:+d} new), "
          f"{s.get('pending', 0)} awaiting results")
    if s["n"]:
        print(f"scored: {s['correct']}/{s['n']} correct = {s['accuracy']:.1%}; "
              f"mean probability given to the actual result {s['mean_prob_actual']:.1%}")
        for src, row in s["by_source"].items():
            print(f"  {src:<10} {int(row['size']):>4} matches  {row['mean']:.1%}")
    print(f"wrote {path}")
    return 0


def cmd_site(args) -> int:
    out = Path(args.out) if args.out else SITE_DIR
    preds = pd.read_pickle(PREDS_PATH) if PREDS_PATH.exists() else None
    logged = load_log()
    if preds is None:
        print("note: no cached backtest, so the performance section will be "
              "omitted; run `backtest` first", file=sys.stderr)
    fixtures = predict_fixtures(
        kind=args.main_model,
        refresh=not args.no_refresh,
        include_odds=args.use_odds,
        limit_days=args.days,
        feature_set=args.feature_set,
    )
    dest = build_site(fixtures, preds, logged=logged, out_dir=out, repo_url=args.repo_url)
    print(f"{len(fixtures)} fixtures; wrote {dest}")
    return 0


def cmd_bet(args) -> int:
    if not PREDS_PATH.exists():
        print("no cached backtest; run `backtest` first", file=sys.stderr)
        return 1
    preds = pd.read_pickle(PREDS_PATH)
    feats = _load_features(args)

    scenarios = args.scenarios or ["close"]
    table = sweep(
        preds,
        feats,
        models=args.models,
        scenarios=scenarios,
        thresholds=tuple(args.thresholds),
        stake_rule=args.stake,
        n_boot=args.boot,
    )
    show = table[[
        "model", "scenario", "threshold", "n_bets", "staked", "profit",
        "roi", "hit_rate", "avg_price", "t_stat", "ci_low", "ci_high",
    ]].copy()
    for c in ("staked", "profit"):
        show[c] = show[c].round(1)
    for c in ("roi", "hit_rate", "ci_low", "ci_high"):
        show[c] = (show[c] * 100).round(2)
    show = show.rename(columns={"roi": "roi%", "hit_rate": "hit%",
                                "ci_low": "ci_low%", "ci_high": "ci_high%"})
    print(f"staking: {args.stake}; ROI and CI in percent; "
          f"'(stake everything)' is the no-skill control\n")
    print(_wide(show))

    dest = REPORTS / "betting_summary.csv"
    table.to_csv(dest, index=False)

    bets = find_bets(
        preds[preds["model"] == args.main_model], feats,
        scenario=scenarios[0], threshold=args.thresholds[0], stake_rule=args.stake,
    )
    if not bets.empty:
        print(f"\n=== {args.main_model} at {scenarios[0]}, "
              f"threshold {args.thresholds[0]:.0%}, by season ===")
        season = by_season(bets)
        season["roi"] = (season["roi"] * 100).round(2)
        print(_wide(season.round({"stake": 1, "profit": 1})))
        summary = summarise_bets(bets, n_boot=args.boot)
        print(f"\noverall ROI {summary['roi']:+.2%} over {summary['n_bets']} bets "
              f"(95% CI {summary['ci_low']:+.2%} .. {summary['ci_high']:+.2%})")
        bets.to_csv(REPORTS / "bets.csv", index=False)
    print(f"\nwrote {dest}")
    return 0


def cmd_track(args) -> int:
    if not PREDS_PATH.exists():
        print("no cached backtest; run `backtest` first", file=sys.stderr)
        return 1
    preds = pd.read_pickle(PREDS_PATH)
    season = args.season or preds["season"].max()
    table = season_tracking(preds, season, args.main_model)
    print(f"=== {season}, {args.main_model}: accuracy through the season ===")
    print(_wide(table))
    out = REPORTS / f"tracking_{season.replace('/', '-')}.csv"
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


def cmd_importance(args) -> int:
    feats = _load_features(args)
    cols = _select_features(feats, args)
    train = feats[feats["result"].notna()]
    model = fit_model(train, cols, args.main_model)
    print(f"trained {args.main_model} on {len(train)} matches {model.meta}")
    print(_wide(feature_importance(model, args.top)))
    return 0


def cmd_predict(args) -> int:
    out = predict_fixtures(
        kind=args.main_model,
        refresh=not args.no_refresh,
        include_odds=args.use_odds,
        limit_days=args.days,
        feature_set=args.feature_set,
    )
    if out.empty:
        print("no upcoming Premier League fixtures published right now")
        return 0
    meta = out.attrs.get("model_meta", {})
    print(f"model={meta.get('kind')} trained on {meta.get('n_train')} matches\n")
    show = out[["date", "home_team", "away_team", "p_home", "p_draw", "p_away", "pick", "confidence"]].copy()
    show["date"] = show["date"].dt.strftime("%a %d %b")
    for c in ("p_home", "p_draw", "p_away", "confidence"):
        show[c] = (show[c] * 100).round(1)
    print(_wide(show))
    dest = REPORTS / "upcoming_predictions.csv"
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Shared flags live on a parent parser so they work *after* the
    # subcommand, which is the order anyone actually types.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--verbose", "-v", action="store_true")
    common.add_argument("--first-season", type=int, default=FIRST_SEASON)
    common.add_argument("--last-season", type=int, default=LAST_SEASON)
    common.add_argument("--refresh", action="store_true", help="re-download data")
    common.add_argument("--use-odds", action="store_true",
                        help="let the model see bookmaker prices and how they moved")
    common.add_argument("--main-model", default="xgb", choices=MODEL_KINDS)
    common.add_argument("--feature-set", default="core", choices=("core", "full"),
                        help="core (~130 cols, the default) or full (~354)")

    parser = argparse.ArgumentParser(prog="epl", description=__doc__, parents=[common])
    sub = parser.add_subparsers(dest="cmd", required=True)

    fe = sub.add_parser("fetch", parents=[common],
                        help="download data and build the feature table")
    fe.add_argument("--with-squads", action="store_true",
                    help="also scrape per-match line-ups from Understat "
                         "(~4,600 requests, ~25 min; see the README on why this is optional)")
    fe.set_defaults(func=cmd_fetch)

    b = sub.add_parser("backtest", parents=[common], help="walk-forward evaluation")
    b.add_argument("--models", nargs="+", default=["xgb", "logreg", "elo", "market", "base"],
                   choices=MODEL_KINDS)
    b.add_argument("--first-test-season", type=int, default=2010)
    b.add_argument("--refit-days", type=int, default=None,
                   help="refit mid-season at this cadence in days (default: once per season)")
    b.add_argument("--no-charts", action="store_true")
    b.add_argument("--no-xg", action="store_true",
                   help="ablate the Understat xG features")
    b.add_argument("--no-avail", action="store_true",
                   help="ablate the squad-availability features")
    b.set_defaults(func=cmd_backtest)

    lg = sub.add_parser("log", parents=[common],
                        help="record current predictions and score past ones")
    lg.add_argument("--days", type=int, default=10, help="fixture horizon")
    lg.add_argument("--no-refresh", action="store_true")
    lg.add_argument("--no-predict", action="store_true",
                    help="only score existing entries, do not add new ones")
    lg.add_argument("--backfill", action="store_true",
                    help="seed from out-of-sample backtest predictions")
    lg.add_argument("--season", default=None, help="season to backfill (default: latest)")
    lg.set_defaults(func=cmd_log)

    st = sub.add_parser("site", parents=[common], help="build the static website")
    st.add_argument("--out", default=None, help="output directory (default: site/)")
    st.add_argument("--days", type=int, default=10, help="fixture horizon")
    st.add_argument("--no-refresh", action="store_true")
    st.add_argument("--repo-url", default="https://github.com/Slzyyy/epl-match-predictor")
    st.set_defaults(func=cmd_site)

    bt = sub.add_parser("bet", parents=[common], help="value-betting backtest")
    bt.add_argument("--models", nargs="+", default=None)
    bt.add_argument("--scenarios", nargs="+", default=None,
                    choices=sorted(PRICE_SCENARIOS), help="which price to bet into")
    bt.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.10],
                    help="minimum edge required to place a bet")
    bt.add_argument("--stake", default="flat", choices=("flat", "kelly"))
    bt.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    bt.set_defaults(func=cmd_bet)

    t = sub.add_parser("track", parents=[common], help="accuracy through a single season")
    t.add_argument("--season", default=None, help="e.g. 2025-26 (default: latest scored)")
    t.set_defaults(func=cmd_track)

    i = sub.add_parser("importance", parents=[common], help="feature importance")
    i.add_argument("--top", type=int, default=30)
    i.set_defaults(func=cmd_importance)

    p = sub.add_parser("predict", parents=[common], help="probabilities for upcoming fixtures")
    p.add_argument("--days", type=int, default=14, help="horizon from the next fixture")
    p.add_argument("--no-refresh", action="store_true")
    p.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    # pandas' fragmentation hint is advisory, about block layout rather than
    # correctness, and the wide feature frame triggers it unavoidably. It is
    # noise in CI logs, so it is silenced at the CLI boundary only.
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
