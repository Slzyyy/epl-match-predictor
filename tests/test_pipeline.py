"""Correctness tests, with the emphasis on leakage.

A match-outcome model is easy to make look good by accident: one post-match
statistic left in the feature matrix and accuracy jumps into the 90s. The
tests below pin down the things that would hide that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epl import features as F
from epl import metrics as M
from epl.data import _parse_dates, _read_csv_robust
from epl.evaluate import walk_forward
from epl.features import build_features, feature_columns


# --------------------------------------------------------------------------
# synthetic fixture: a tiny deterministic league
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def toy_matches() -> pd.DataFrame:
    """Six teams, four seasons, a proper double round-robin.

    Every ordered pair meets exactly once a season, matching the real
    competition — which is what makes (season, home, away) a unique key for
    the xG join. Team A always beats everyone, F always loses, so the ordering
    of any strength feature is known in advance.
    """
    import itertools

    teams = ["A", "B", "C", "D", "E", "F"]
    strength = {t: float(len(teams) - 1 - i) for i, t in enumerate(teams)}
    rows = []
    day = pd.Timestamp("2018-08-10")
    for year in (2018, 2019, 2020, 2021):
        for home, away in itertools.permutations(teams, 2):
            hg = int(strength[home]) + 1
            ag = int(strength[away])
            rows.append(
                {
                    "date": day,
                    "season": f"{year}-{(year + 1) % 100:02d}",
                    "season_start": year,
                    "home_team": home,
                    "away_team": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "result": "H" if hg > ag else ("D" if hg == ag else "A"),
                    "home_shots": 10 + hg,
                    "away_shots": 10 + ag,
                    "home_shots_target": 5 + hg,
                    "away_shots_target": 5 + ag,
                    "home_corners": 5,
                    "away_corners": 5,
                    "home_fouls": 10,
                    "away_fouls": 10,
                    "home_yellows": 1,
                    "away_yellows": 1,
                    "home_reds": 0,
                    "away_reds": 0,
                    "ht_home_goals": 0,
                    "ht_away_goals": 0,
                    "time": "15:00",
                    "odds_h": 2.0,
                    "odds_d": 3.5,
                    "odds_a": 4.0,
                }
            )
            day += pd.Timedelta(days=4)
        day += pd.Timedelta(days=60)  # close season
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["match_id"] = df.index
    return df


# --------------------------------------------------------------------------
# leakage
# --------------------------------------------------------------------------
def test_feature_columns_exclude_match_outcome(toy_matches):
    feats = build_features(toy_matches, include_odds=True)
    cols = set(feature_columns(feats))
    forbidden = {
        "y", "result", "home_goals", "away_goals", "ht_home_goals", "ht_away_goals",
        "home_shots", "away_shots", "home_shots_target", "away_shots_target",
        "home_corners", "away_corners", "home_yellows", "away_yellows",
        "home_reds", "away_reds", "home_fouls", "away_fouls",
        "odds_h", "odds_d", "odds_a",
    }
    assert cols & forbidden == set()


def test_first_match_of_history_has_no_form(toy_matches):
    """A team's very first match cannot know anything about its own form."""
    feats = build_features(toy_matches)
    first = feats.iloc[0]
    for col in feats.columns:
        if col.startswith(("home_form", "away_form", "home_venue_form", "away_venue_form")):
            assert pd.isna(first[col]), f"{col} populated on the first ever match"


def test_form_feature_matches_hand_computation(toy_matches):
    """home_form3_points must equal the mean of that team's previous 3 results."""
    feats = build_features(toy_matches)
    team = "B"
    hist = []
    for row in feats.itertuples():
        if row.home_team == team:
            if len(hist) >= 3:
                expected = float(np.mean(hist[-3:]))
                got = getattr(row, "home_form3_points")
                assert got == pytest.approx(expected), f"match {row.match_id}"
        pts_home = 3.0 if row.home_goals > row.away_goals else (1.0 if row.home_goals == row.away_goals else 0.0)
        if row.home_team == team:
            hist.append(pts_home)
        elif row.away_team == team:
            hist.append(3.0 - pts_home if pts_home != 1.0 else 1.0)


def test_shuffling_future_results_does_not_change_past_features(toy_matches):
    """Rewriting later results must leave earlier rows' features untouched."""
    cut = 40
    tampered = toy_matches.copy()
    tampered.loc[cut:, "home_goals"] = 9
    tampered.loc[cut:, "away_goals"] = 0
    tampered.loc[cut:, "result"] = "H"

    base = build_features(toy_matches)
    alt = build_features(tampered)
    cols = [c for c in feature_columns(base) if c in set(feature_columns(alt))]
    pd.testing.assert_frame_equal(
        base.loc[: cut - 1, cols], alt.loc[: cut - 1, cols], check_dtype=False
    )


def test_walk_forward_never_trains_on_the_future(toy_matches):
    """The recorded training size must match only matches strictly before the test block."""
    feats = build_features(toy_matches)
    cols = feature_columns(feats)
    preds = walk_forward(
        feats, cols, ("elo",), first_test_season=2020, min_train_matches=10
    )
    for season, sub in preds.groupby("season"):
        cutoff = sub["date"].min()
        expected = int((feats["date"] < cutoff).sum())
        assert sub["n_train"].unique().tolist() == [expected]
        assert expected > 0


# --------------------------------------------------------------------------
# Elo
# --------------------------------------------------------------------------
def test_elo_rewards_the_winner(toy_matches):
    feats = build_features(toy_matches)
    last = feats.iloc[-1]
    final = {}
    for row in feats.itertuples():
        final[row.home_team] = row.home_elo
        final[row.away_team] = row.away_elo
    assert final["A"] > final["B"] > final["C"] > final["D"] > final["E"] > final["F"]
    assert last["elo_exp_home"] == pytest.approx(
        1.0 / (1.0 + 10 ** (-last["elo_diff"] / 400.0))
    )


def test_elo_is_zero_sum_within_a_match():
    matches = pd.DataFrame(
        {
            "season": ["2020-21"] * 2,
            "home_team": ["X", "Y"],
            "away_team": ["Y", "X"],
            "home_goals": [2, 0],
            "away_goals": [0, 1],
            "date": pd.to_datetime(["2020-08-01", "2020-08-08"]),
        }
    )
    out = F.add_elo(matches)
    # Both teams start level, so after one match the gain and loss must cancel.
    assert out.loc[0, "home_elo"] == out.loc[0, "away_elo"] == F.ELO_START
    second = out.loc[1, ["home_elo", "away_elo"]].sum()
    assert second == pytest.approx(2 * F.ELO_START)


def test_elo_home_advantage_favours_the_host():
    matches = pd.DataFrame(
        {
            "season": ["2020-21"],
            "home_team": ["X"],
            "away_team": ["Y"],
            "home_goals": [1],
            "away_goals": [1],
            "date": pd.to_datetime(["2020-08-01"]),
        }
    )
    out = F.add_elo(matches)
    assert out.loc[0, "elo_exp_home"] > 0.5


# --------------------------------------------------------------------------
# Poisson
# --------------------------------------------------------------------------
def test_poisson_probabilities_are_a_distribution(toy_matches):
    feats = build_features(toy_matches)
    pois = feats[["pois_p_h", "pois_p_d", "pois_p_a"]].dropna()
    assert len(pois) > 0
    assert np.allclose(pois.sum(axis=1), 1.0, atol=1e-6)
    assert (pois >= 0).all().all()


def test_poisson_favours_the_side_expected_to_score_more(toy_matches):
    feats = build_features(toy_matches).dropna(subset=["pois_p_h", "pois_p_a"])
    clear = feats[(feats["exp_home_goals"] - feats["exp_away_goals"]).abs() > 0.5]
    assert len(clear) > 0
    favours_home = clear["exp_home_goals"] > clear["exp_away_goals"]
    assert ((clear["pois_p_h"] > clear["pois_p_a"]) == favours_home).all()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def test_metrics_on_perfect_and_worst_forecasts():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    assert M.rps(y, perfect) == pytest.approx(0.0)
    assert M.brier(y, perfect) == pytest.approx(0.0)
    assert M.accuracy(y, perfect) == pytest.approx(1.0)
    assert M.log_loss(y, perfect) == pytest.approx(0.0, abs=1e-9)

    uniform = np.full((3, 3), 1 / 3)
    assert M.log_loss(y, uniform) == pytest.approx(np.log(3))


def test_rps_penalises_distance_on_the_ordered_scale():
    """Calling an away win when it was a home win is worse than calling a draw."""
    y = np.array([0])  # home win
    draw_call = np.array([[0.0, 1.0, 0.0]])
    away_call = np.array([[0.0, 0.0, 1.0]])
    assert M.rps(y, draw_call) < M.rps(y, away_call)


def test_rps_beats_a_worse_forecast_of_the_same_winner():
    y = np.array([0])
    confident = np.array([[0.8, 0.15, 0.05]])
    timid = np.array([[0.4, 0.35, 0.25]])
    assert M.rps(y, confident) < M.rps(y, timid)


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def test_parse_dates_handles_both_archive_formats():
    s = pd.Series(["16/08/03", "08/05/2004", "bad", None])
    out = _parse_dates(s)
    assert out.iloc[0] == pd.Timestamp("2003-08-16")
    assert out.iloc[1] == pd.Timestamp("2004-05-08")
    assert pd.isna(out.iloc[2]) and pd.isna(out.iloc[3])


def test_robust_reader_keeps_rows_with_extra_commas(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("Div,Date,HomeTeam\nE0,01/01/20,A\nE0,02/01/20,B,,,\nE0,03/01/20\n")
    df = _read_csv_robust(path)
    assert len(df) == 3
    assert df["HomeTeam"].tolist()[:2] == ["A", "B"]
    assert pd.isna(df["HomeTeam"].iloc[2])


def test_robust_reader_falls_back_to_latin1(tmp_path):
    path = tmp_path / "y.csv"
    path.write_bytes(b"Div,Referee\nE0,M\xa0Dean\n")
    df = _read_csv_robust(path)
    assert len(df) == 1


# --------------------------------------------------------------------------
# unplayed fixtures
# --------------------------------------------------------------------------
def test_unplayed_fixture_gets_features_without_polluting_history(toy_matches):
    """An appended fixture must inherit pre-match features and change nothing."""
    fixture = toy_matches.iloc[[-1]].copy()
    fixture["date"] = toy_matches["date"].max() + pd.Timedelta(days=7)
    fixture["home_team"], fixture["away_team"] = "B", "C"
    for col in ("home_goals", "away_goals", "home_shots", "away_shots",
                "home_shots_target", "away_shots_target"):
        fixture[col] = np.nan
    fixture["result"] = pd.NA
    fixture["match_id"] = toy_matches["match_id"].max() + 1
    combined = pd.concat([toy_matches, fixture], ignore_index=True)

    base = build_features(toy_matches)
    withfx = build_features(combined)
    cols = feature_columns(base)

    # History is untouched.
    pd.testing.assert_frame_equal(
        base[cols], withfx.iloc[: len(base)][cols].reset_index(drop=True), check_dtype=False
    )

    row = withfx.iloc[-1]
    assert pd.isna(row["result"])
    # The fixture still gets ratings, form and a Poisson forecast.
    assert np.isfinite(row["home_elo"]) and np.isfinite(row["away_elo"])
    assert np.isfinite(row["home_form5_points"])
    assert np.isfinite(row["pois_p_h"])
    # Season-to-date counts only matches actually played.
    played_b = (
        ((toy_matches["home_team"] == "B") | (toy_matches["away_team"] == "B"))
        & (toy_matches["season_start"] == fixture["season_start"].iloc[0])
    ).sum()
    assert row["home_season_matches"] == played_b


def test_two_unplayed_fixtures_do_not_feed_each_other(toy_matches):
    """The second of two upcoming fixtures must not see a fake result from the first."""
    rows = []
    for i, (h, a) in enumerate((("B", "E"), ("B", "F"))):
        fx = toy_matches.iloc[[-1]].copy()
        fx["date"] = toy_matches["date"].max() + pd.Timedelta(days=7 * (i + 1))
        fx["home_team"], fx["away_team"] = h, a
        for col in ("home_goals", "away_goals", "home_shots", "away_shots",
                    "home_shots_target", "away_shots_target"):
            fx[col] = np.nan
        fx["result"] = pd.NA
        fx["match_id"] = toy_matches["match_id"].max() + 1 + i
        rows.append(fx)
    combined = pd.concat([toy_matches, *rows], ignore_index=True)
    feats = build_features(combined)
    first, second = feats.iloc[-2], feats.iloc[-1]
    # Same team, no match played in between -> identical pre-match state.
    assert first["home_team"] == second["home_team"] == "B"
    assert first["home_elo"] == pytest.approx(second["home_elo"])
    assert first["home_form5_points"] == pytest.approx(second["home_form5_points"])
    assert first["home_season_matches"] == second["home_season_matches"]


# --------------------------------------------------------------------------
# xG
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def toy_xg(toy_matches) -> pd.DataFrame:
    """Synthetic Understat-shaped xG: proportional to goals, plus a constant."""
    rng = np.random.default_rng(0)
    xg = toy_matches[["season_start", "home_team", "away_team"]].copy()
    xg["home_xg"] = toy_matches["home_goals"] * 0.8 + 0.3
    xg["away_xg"] = toy_matches["away_goals"] * 0.8 + 0.3
    xg["home_npxg"] = xg["home_xg"] * 0.95
    xg["away_npxg"] = xg["away_xg"] * 0.95
    xg["home_ppda"] = rng.uniform(8, 16, len(xg))
    xg["away_ppda"] = rng.uniform(8, 16, len(xg))
    xg["home_deep"] = rng.integers(2, 14, len(xg)).astype(float)
    xg["away_deep"] = rng.integers(2, 14, len(xg)).astype(float)
    return xg


def test_attach_xg_joins_every_match(toy_matches, toy_xg):
    from epl.xg import attach_xg

    out = attach_xg(toy_matches, toy_xg)
    assert len(out) == len(toy_matches)
    assert out["home_xg"].notna().all()
    # The join must not reorder or duplicate rows.
    pd.testing.assert_series_equal(out["match_id"], toy_matches["match_id"])


def test_attach_xg_without_data_leaves_columns_empty(toy_matches):
    from epl.xg import XG_MATCH_COLS, attach_xg

    out = attach_xg(toy_matches, pd.DataFrame())
    for col in XG_MATCH_COLS:
        assert col in out.columns
        assert out[col].isna().all()


def test_raw_match_xg_never_becomes_a_feature(toy_matches, toy_xg):
    from epl.xg import XG_MATCH_COLS, attach_xg

    feats = build_features(attach_xg(toy_matches, toy_xg))
    assert set(feature_columns(feats)) & set(XG_MATCH_COLS) == set()


def test_xg_features_respect_the_leakage_guarantee(toy_matches, toy_xg):
    """Rewriting future xG must not disturb earlier rows' xG features."""
    from epl.xg import attach_xg

    cut = 40
    tampered = toy_xg.copy()
    tampered.loc[cut:, ["home_xg", "home_npxg"]] = 9.0
    tampered.loc[cut:, ["away_xg", "away_npxg"]] = 0.0

    base = build_features(attach_xg(toy_matches, toy_xg))
    alt = build_features(attach_xg(toy_matches, tampered))
    xg_cols = [c for c in feature_columns(base) if F.is_xg_feature(c)]
    assert xg_cols, "no xG features were built"
    pd.testing.assert_frame_equal(
        base.loc[: cut - 1, xg_cols], alt.loc[: cut - 1, xg_cols], check_dtype=False
    )


def test_xg_poisson_is_a_distribution(toy_matches, toy_xg):
    from epl.xg import attach_xg

    feats = build_features(attach_xg(toy_matches, toy_xg))
    pois = feats[["xgpois_p_h", "xgpois_p_d", "xgpois_p_a"]].dropna()
    assert len(pois) > 0
    assert np.allclose(pois.sum(axis=1), 1.0, atol=1e-6)


def test_drop_xg_removes_exactly_the_xg_features(toy_matches, toy_xg):
    from epl.xg import attach_xg

    feats = build_features(attach_xg(toy_matches, toy_xg))
    cols = feature_columns(feats)
    kept = F.drop_xg(cols)
    assert len(kept) < len(cols)
    assert not any(F.is_xg_feature(c) for c in kept)
    # Nothing non-xG should have been dropped.
    assert set(cols) - set(kept) == {c for c in cols if F.is_xg_feature(c)}


def test_features_build_without_any_xg_columns(toy_matches):
    """The pipeline must still work when Understat data was never attached."""
    feats = build_features(toy_matches)
    cols = feature_columns(feats)
    assert not any(F.is_xg_feature(c) for c in cols)
    assert "pois_p_h" in cols  # the goals-based Poisson is unaffected


# --------------------------------------------------------------------------
# squad availability
# --------------------------------------------------------------------------
def _toy_rosters(matches: pd.DataFrame, drop_after: tuple[str, int] | None = None):
    """Understat-shaped rosters: 11 named players per team, all playing 90.

    ``drop_after`` makes one player stop appearing from a given match index on,
    which is how the absence features are exercised.
    """
    index = pd.DataFrame(
        {
            "understat_id": matches["match_id"].to_numpy(),
            "date": matches["date"].to_numpy(),
            "home_team": matches["home_team"].to_numpy(),
            "away_team": matches["away_team"].to_numpy(),
            "season_start": matches["season_start"].to_numpy(),
        }
    )
    rows = []
    for pos, row in enumerate(matches.itertuples()):
        for side, team in (("h", row.home_team), ("a", row.away_team)):
            for i in range(11):
                pid = f"{team}{i}"
                if drop_after and pid == drop_after[0] and pos >= drop_after[1]:
                    continue
                rows.append(
                    {
                        "understat_id": row.match_id,
                        "player_id": pid,
                        "player": pid,
                        "position": "GK" if i == 0 else "MC",
                        "positionOrder": i + 1,
                        "time": 90.0,
                        "goals": 0,
                        "xG": 0.1,
                        "xA": 0.1,
                        "xGChain": 0.2,
                        "red_card": 0,
                        "is_home": side == "h",
                        "season_start": row.season_start,
                    }
                )
    return pd.DataFrame(rows), index


def test_availability_is_zero_when_nobody_is_missing(toy_matches):
    from epl.lineups import build_availability

    rosters, index = _toy_rosters(toy_matches)
    avail = build_availability(rosters, index)
    played = avail[avail["understat_id"] != -1].dropna(subset=["absent_minutes_share_2"])
    assert len(played) > 0
    assert (played["absent_minutes_share_2"] == 0.0).all()
    assert (played["absent_starters_2"] == 0.0).all()


def test_a_player_who_stops_appearing_registers_as_absent(toy_matches):
    """Absence is a *recent* signal, so it fires then fades as the core rolls on."""
    from epl.lineups import build_availability

    # B3 vanishes a third of the way through, while still in the recent core.
    cut = 40
    rosters, index = _toy_rosters(toy_matches, drop_after=("B3", cut))
    avail = build_availability(rosters, index)
    b = (
        avail[(avail["team"] == "B") & (avail["understat_id"] != -1)]
        .merge(index, on="understat_id")
        .sort_values("date")
        .dropna(subset=["absent_minutes_share_2"])
    )
    before = b[b["understat_id"] < cut]["absent_minutes_share_2"]
    just_after = b[(b["understat_id"] > cut) & (b["understat_id"] < cut + 30)]
    long_after = b[b["understat_id"] > cut + 60]["absent_minutes_share_2"]

    assert (before == 0.0).all(), "nobody was missing before the drop"
    assert (just_after["absent_minutes_share_2"] > 0).any(), "absence never registered"
    assert just_after["absent_starters_2"].max() >= 1
    # Once he falls out of the 10-match core he is no longer 'missing' — the
    # team's form features carry a long-term absence from there on.
    assert (long_after == 0.0).all()


def test_availability_never_looks_at_the_current_match(toy_matches):
    """Changing who played in later matches must not alter earlier rows."""
    from epl.lineups import build_availability

    base_r, index = _toy_rosters(toy_matches)
    alt_r, _ = _toy_rosters(toy_matches, drop_after=("B3", 60))
    base = build_availability(base_r, index).set_index(["understat_id", "team"]).sort_index()
    alt = build_availability(alt_r, index).set_index(["understat_id", "team"]).sort_index()
    early = [i for i in base.index if i[0] != -1 and i[0] < 60]
    pd.testing.assert_frame_equal(
        base.loc[early].sort_index(), alt.loc[early].sort_index(), check_dtype=False
    )


def test_every_team_gets_a_trailing_state_row(toy_matches):
    from epl.lineups import AVAILABILITY_COLS, build_availability

    rosters, index = _toy_rosters(toy_matches)
    avail = build_availability(rosters, index)
    trailing = avail[avail["understat_id"] == -1]
    assert set(trailing["team"]) == set(toy_matches["home_team"])
    assert trailing[AVAILABILITY_COLS].notna().any(axis=1).all()


def test_upcoming_fixture_inherits_the_trailing_state(toy_matches):
    from epl.lineups import attach_availability, build_availability

    # Dropped near the end, so he is still inside B's last-10 core.
    rosters, index = _toy_rosters(toy_matches, drop_after=("B3", len(toy_matches) - 12))
    avail = build_availability(rosters, index).merge(
        index[["understat_id", "season_start", "home_team", "away_team"]],
        on="understat_id", how="left",
    )
    avail.loc[avail["understat_id"] == -1, "season_start"] = toy_matches["season_start"].max()

    # An upcoming fixture belongs to the next season, as a real one would.
    fixture = toy_matches.iloc[[-1]].copy()
    fixture["date"] = toy_matches["date"].max() + pd.Timedelta(days=60)
    fixture["season_start"] = toy_matches["season_start"].max() + 1
    fixture["season"] = "2022-23"
    fixture["home_team"], fixture["away_team"] = "B", "E"
    for col in ("home_goals", "away_goals"):
        fixture[col] = np.nan
    fixture["result"] = pd.NA
    fixture["match_id"] = toy_matches["match_id"].max() + 1
    combined = pd.concat([toy_matches, fixture], ignore_index=True)

    out = attach_availability(combined, avail)
    row = out.iloc[-1]
    assert pd.isna(row["result"])
    # B is missing a player, so the upcoming fixture must know about it.
    assert row["home_absent_minutes_share_2"] > 0.0


def test_drop_availability_removes_exactly_those_features(toy_matches):
    from epl.lineups import attach_availability, build_availability

    rosters, index = _toy_rosters(toy_matches)
    avail = build_availability(rosters, index).merge(
        index[["understat_id", "season_start", "home_team", "away_team"]],
        on="understat_id", how="left",
    )
    feats = build_features(attach_availability(toy_matches, avail))
    cols = feature_columns(feats)
    assert any(F.is_availability_feature(c) for c in cols), "no availability features built"
    kept = F.drop_availability(cols)
    assert set(cols) - set(kept) == {c for c in cols if F.is_availability_feature(c)}


# --------------------------------------------------------------------------
# odds movement
# --------------------------------------------------------------------------
def _with_open_close(matches: pd.DataFrame) -> pd.DataFrame:
    """Add an opening price that drifts toward the home side by close."""
    df = matches.copy()
    df["open_h"], df["open_d"], df["open_a"] = 2.5, 3.5, 3.0
    df["close_h"], df["close_d"], df["close_a"] = 2.0, 3.5, 4.0
    return df


def test_movement_features_are_probability_drifts(toy_matches):
    feats = build_features(_with_open_close(toy_matches), include_odds=True)
    for prefix in ("mkt_open", "mkt_close"):
        probs = feats[[f"{prefix}_h", f"{prefix}_d", f"{prefix}_a"]]
        assert np.allclose(probs.sum(axis=1), 1.0)
    # Money moved to the home side, so its probability must have risen.
    assert (feats["mov_h"] > 0).all()
    assert (feats["mov_a"] < 0).all()
    # Drifts are a redistribution, so they sum to zero.
    assert np.allclose(feats[["mov_h", "mov_d", "mov_a"]].sum(axis=1), 0.0, atol=1e-9)
    assert np.allclose(feats["mov_abs"], feats[["mov_h", "mov_d", "mov_a"]].abs().sum(axis=1))


def test_overround_is_measured_on_both_prices(toy_matches):
    feats = build_features(_with_open_close(toy_matches), include_odds=True)
    # 1/2.5 + 1/3.5 + 1/3.0 = 1.0190...
    assert feats["mkt_overround_open"].iloc[0] == pytest.approx(
        1 / 2.5 + 1 / 3.5 + 1 / 3.0
    )
    assert feats["mkt_overround_close"].iloc[0] == pytest.approx(
        1 / 2.0 + 1 / 3.5 + 1 / 4.0
    )
    assert feats["mkt_overround_drop"].iloc[0] == pytest.approx(
        feats["mkt_overround_open"].iloc[0] - feats["mkt_overround_close"].iloc[0]
    )


def test_raw_prices_never_become_features(toy_matches):
    feats = build_features(_with_open_close(toy_matches), include_odds=True)
    cols = set(feature_columns(feats))
    assert cols & {"open_h", "open_d", "open_a", "close_h", "close_d", "close_a"} == set()


def test_market_features_are_excluded_by_default(toy_matches):
    feats = build_features(_with_open_close(toy_matches), include_odds=True)
    cols = feature_columns(feats)
    assert any(F.is_market_feature(c) for c in cols)
    kept = F.drop_market(cols)
    assert not any(c.startswith(("mkt_", "mov_")) for c in kept)
    assert set(cols) - set(kept) == {c for c in cols if F.is_market_feature(c)}


def test_movement_absent_when_only_one_price_is_published(toy_matches):
    """Seasons with no closing price must not get fabricated drift."""
    df = toy_matches.copy()
    df["open_h"], df["open_d"], df["open_a"] = 2.5, 3.5, 3.0
    feats = build_features(df, include_odds=True)
    assert "mov_h" not in feats.columns


def test_open_close_pair_requires_both_sides_from_one_bookmaker():
    """_pick_open_close must not mix one firm's opening price with another's close."""
    from epl.data import _pick_open_close

    raw = pd.DataFrame(
        {
            # Pinnacle has an opening price but no close on this row.
            "PSH": [2.0], "PSD": [3.0], "PSA": [4.0],
            "PSCH": [np.nan], "PSCD": [np.nan], "PSCA": [np.nan],
            # Bet365 has both, so that pair should be the one taken.
            "B365H": [2.2], "B365D": [3.2], "B365A": [3.8],
            "B365CH": [2.1], "B365CD": [3.3], "B365CA": [3.9],
        }
    )
    out = _pick_open_close(raw)
    assert out.loc[0, "open_h"] == 2.2
    assert out.loc[0, "close_h"] == 2.1


def test_market_open_model_uses_the_opening_price(toy_matches):
    from epl.model import fit_model

    feats = build_features(_with_open_close(toy_matches), include_odds=True)
    model = fit_model(feats, [], "market_open")
    proba = model.predict_proba(feats)
    expected = np.array([1 / 2.5, 1 / 3.5, 1 / 3.0])
    assert np.allclose(proba[0], expected / expected.sum())


# --------------------------------------------------------------------------
# value betting
# --------------------------------------------------------------------------
# Three equally likely outcomes priced with a 5% margin: 3 / 1.05.
FAIR_PRICE = 3.0 / 1.05
NET = FAIR_PRICE - 1.0


@pytest.fixture
def toy_bets():
    """Three equally likely outcomes carrying a 5% bookmaker margin."""
    rng = np.random.default_rng(1)
    n = 300
    y = rng.integers(0, 3, n)
    matches = pd.DataFrame(
        {
            "match_id": np.arange(n),
            "close_h": FAIR_PRICE, "close_d": FAIR_PRICE, "close_a": FAIR_PRICE,
        }
    )
    preds = pd.DataFrame(
        {
            "match_id": np.arange(n),
            "season": "2020-21",
            "date": pd.Timestamp("2021-01-01"),
            "model": "toy",
            "y": y,
            "p_h": 1 / 3, "p_d": 1 / 3, "p_a": 1 / 3,
        }
    )
    return preds, matches


def test_profit_accounting(toy_bets):
    from epl.betting import find_bets

    preds, matches = toy_bets
    bets = find_bets(preds, matches, scenario="close", bet_everything=True)
    assert len(bets) == 3 * len(matches)
    won = bets[bets["won"] > 0]
    lost = bets[bets["won"] == 0]
    # Stake 1: a win returns the net odds as profit, a loss costs the stake.
    assert np.allclose(won["profit"], NET)
    assert np.allclose(lost["profit"], -1.0)
    # Exactly one outcome per match wins.
    assert len(won) == len(matches)


def test_a_fair_model_finds_no_value_against_the_margin(toy_bets):
    """1/3 against a 5%-margin price is a negative edge, so nothing is staked."""
    from epl.betting import find_bets

    preds, matches = toy_bets
    bets = find_bets(preds, matches, scenario="close", threshold=0.0)
    assert len(bets) == 0


def test_the_bookmakers_own_probabilities_find_no_value():
    """De-vigged market probabilities must never beat the price they came from.

    This is the check that catches the classic error of comparing a model
    against de-vigged probabilities instead of 1/price.
    """
    from epl.betting import find_bets
    from epl.model import market_proba

    rng = np.random.default_rng(3)
    n = 400
    raw = rng.uniform(0.15, 0.7, size=(n, 3))
    probs = raw / raw.sum(axis=1, keepdims=True)
    overround = 1.06
    prices = 1.0 / (probs * overround)
    matches = pd.DataFrame(
        {
            "match_id": np.arange(n),
            "close_h": prices[:, 0], "close_d": prices[:, 1], "close_a": prices[:, 2],
            "odds_h": prices[:, 0], "odds_d": prices[:, 1], "odds_a": prices[:, 2],
        }
    )
    devig = market_proba(matches)
    preds = pd.DataFrame(
        {
            "match_id": np.arange(n), "season": "2020-21",
            "date": pd.Timestamp("2021-01-01"), "model": "market",
            "y": rng.integers(0, 3, n),
            "p_h": devig[:, 0], "p_d": devig[:, 1], "p_a": devig[:, 2],
        }
    )
    assert len(find_bets(preds, matches, scenario="close", threshold=0.0)) == 0


def test_a_perfect_model_makes_the_price_minus_one(toy_bets):
    from epl.betting import find_bets, summarise_bets

    preds, matches = toy_bets
    preds = preds.copy()
    for i, col in enumerate(("p_h", "p_d", "p_a")):
        preds[col] = (preds["y"] == i).astype(float)
    bets = find_bets(preds, matches, scenario="close", threshold=0.0)
    assert len(bets) == len(matches)  # exactly the winning outcome each time
    assert summarise_bets(bets, n_boot=50)["roi"] == pytest.approx(NET)


def test_higher_thresholds_select_fewer_bets(toy_bets):
    from epl.betting import find_bets

    preds, matches = toy_bets
    preds = preds.copy()
    preds["p_h"] = 0.55  # a genuine edge at this price
    preds["p_d"] = 0.30
    preds["p_a"] = 0.15
    counts = [
        len(find_bets(preds, matches, scenario="close", threshold=t))
        for t in (0.0, 0.2, 0.6, 1.0)
    ]
    assert counts[0] == len(matches)  # the home bet clears a zero threshold
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_kelly_stakes_scale_with_the_edge(toy_bets):
    from epl.betting import find_bets

    preds, matches = toy_bets
    preds = preds.copy()
    preds["p_h"], preds["p_d"], preds["p_a"] = 0.60, 0.25, 0.15
    bets = find_bets(preds, matches, scenario="close", stake_rule="kelly",
                     kelly_fraction=0.25)
    # Only the home bet has an edge; stake = 0.25 * edge / (price - 1).
    edge = 0.60 * FAIR_PRICE - 1.0
    assert np.allclose(bets["stake"], 0.25 * edge / NET)
    assert (bets["outcome"] == "H").all()


def test_summary_is_internally_consistent(toy_bets):
    from epl.betting import find_bets, summarise_bets

    preds, matches = toy_bets
    bets = find_bets(preds, matches, scenario="close", bet_everything=True)
    s = summarise_bets(bets, n_boot=200)
    assert s["staked"] == pytest.approx(bets["stake"].sum())
    assert s["profit"] == pytest.approx(bets["profit"].sum())
    assert s["roi"] == pytest.approx(s["profit"] / s["staked"])
    # Staking every outcome returns exactly 1/overround - 1, here -4.76%.
    assert s["roi"] == pytest.approx(1 / 1.05 - 1, abs=1e-9)
    # Every match returns the same amount, so the bootstrap has no spread.
    assert s["ci_low"] == pytest.approx(s["roi"])
    assert s["ci_high"] == pytest.approx(s["roi"])


def test_unknown_scenario_and_stake_rule_are_rejected(toy_bets):
    from epl.betting import find_bets

    preds, matches = toy_bets
    with pytest.raises(ValueError):
        find_bets(preds, matches, scenario="nonsense")
    with pytest.raises(ValueError):
        find_bets(preds, matches, scenario="close", stake_rule="martingale")


# --------------------------------------------------------------------------
# site
# --------------------------------------------------------------------------
def test_already_played_fixtures_are_dropped(monkeypatch):
    """A stale fixtures feed must not put finished matches on the site."""
    import epl.predict as P

    today = pd.Timestamp.now().normalize()
    feed = pd.DataFrame(
        {
            "date": [today - pd.Timedelta(days=3), today, today + pd.Timedelta(days=2)],
            "home_team": pd.array(["A", "B", "C"], dtype="string"),
            "away_team": pd.array(["D", "E", "F"], dtype="string"),
            "odds_h": 2.0, "odds_d": 3.5, "odds_a": 4.0,
        }
    )
    captured = {}

    def fake_build(matches, fx, *a, **k):
        captured["dates"] = list(fx["date"])
        raise RuntimeError("stop here")

    history = pd.DataFrame(
        {"home_team": pd.array(["A", "C"], dtype="string"),
         "away_team": pd.array(["D", "F"], dtype="string")}
    )
    monkeypatch.setattr(P, "load_matches", lambda **k: history)
    monkeypatch.setattr(P, "attach_xg", lambda m, x: m)
    monkeypatch.setattr(P, "load_xg", lambda **k: pd.DataFrame())
    monkeypatch.setattr(P, "_append_fixtures", fake_build)
    with pytest.raises(RuntimeError):
        P.predict_fixtures(fixtures=feed, refresh=False, limit_days=30)
    assert all(d >= today for d in captured["dates"]), "a past fixture survived"
    assert len(captured["dates"]) == 2


def test_site_renders_without_fixtures_or_backtest(tmp_path):
    """The page must still build when the feed is empty."""
    from epl.site import build

    dest = build(pd.DataFrame(), None, out_dir=tmp_path)
    page = dest.read_text()
    assert "No Premier League fixtures" in page
    assert "does not beat the bookmakers" in page
    assert "<html" in page and "</html>" in page


def test_site_renders_fixture_probabilities(tmp_path):
    from epl.site import build

    fx = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-11-01")],
            "home_team": ["Arsenal"], "away_team": ["Chelsea"],
            "p_home": [0.5], "p_draw": [0.3], "p_away": [0.2],
        }
    )
    page = build(fx, None, out_dir=tmp_path).read_text()
    assert "Arsenal v Chelsea" in page
    assert "50%" in page and "30%" in page
    # Widths must be the probabilities, so the bar is not misleading.
    assert "width:50.00%" in page


def test_site_escapes_team_names(tmp_path):
    """Team names come from a third-party feed, so they are escaped."""
    from epl.site import build

    fx = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-11-01")],
            "home_team": ["<script>alert(1)</script>"], "away_team": ["Chelsea"],
            "p_home": [0.5], "p_draw": [0.3], "p_away": [0.2],
        }
    )
    page = build(fx, None, out_dir=tmp_path).read_text()
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


# --------------------------------------------------------------------------
# fixture sourcing
# --------------------------------------------------------------------------
def _understat_payload():
    """Minimal Understat league payload: one played match, two to come."""
    return {
        "dates": [
            {"isResult": True, "datetime": "2026-08-15 15:00",
             "h": {"title": "Arsenal"}, "a": {"title": "Chelsea"},
             "goals": {"h": "1", "a": "0"}, "xG": {"h": "1.5", "a": "0.8"}},
            {"isResult": False, "datetime": "2026-08-22 17:30",
             "h": {"title": "Nottingham Forest"}, "a": {"title": "Wolverhampton Wanderers"},
             "goals": {"h": None, "a": None}, "xG": {"h": None, "a": None}},
            {"isResult": False, "datetime": "2026-08-23 14:00",
             "h": {"title": "Queens Park Rangers"}, "a": {"title": "Manchester United"},
             "goals": {"h": None, "a": None}, "xG": {"h": None, "a": None}},
        ],
        "teams": {},
    }


def test_unplayed_fixtures_are_extracted_with_mapped_names(monkeypatch):
    import epl.xg as X

    monkeypatch.setattr(X, "download_season_xg", lambda y, **k: _understat_payload())
    fx = X.load_season_fixtures(2026)
    assert len(fx) == 2, "only the unplayed matches belong in the fixture list"
    # Understat's long names must map to the names the model knows.
    assert fx["home_team"].tolist() == ["Nott'm Forest", "QPR"]
    assert fx["away_team"].tolist() == ["Wolves", "Man United"]
    assert fx["time"].tolist() == ["17:30", "14:00"]
    assert fx["date"].tolist() == [pd.Timestamp("2026-08-22"), pd.Timestamp("2026-08-23")]


def test_played_matches_never_appear_as_fixtures(monkeypatch):
    import epl.xg as X

    monkeypatch.setattr(X, "download_season_xg", lambda y, **k: _understat_payload())
    fx = X.load_season_fixtures(2026)
    assert not ((fx["home_team"] == "Arsenal") & (fx["away_team"] == "Chelsea")).any()


def test_fixtures_survive_football_data_being_down(monkeypatch):
    """The calendar must still arrive when the odds feed throws."""
    import epl.predict as P

    base = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-22")], "time": ["17:30"],
            "home_team": pd.array(["Arsenal"], dtype="string"),
            "away_team": pd.array(["Chelsea"], dtype="string"),
        }
    )
    monkeypatch.setattr(P, "load_upcoming_fixtures", lambda **k: base)
    monkeypatch.setattr(
        P, "load_fixtures", lambda **k: (_ for _ in ()).throw(RuntimeError("site down"))
    )
    out = P.upcoming_fixtures()
    assert len(out) == 1
    assert out["home_team"].iloc[0] == "Arsenal"
    assert "odds_h" not in out.columns  # no prices, but the fixture is there


def test_odds_are_merged_onto_the_calendar_when_available(monkeypatch):
    import epl.predict as P

    base = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-22"), pd.Timestamp("2027-01-10")],
            "time": ["17:30", "14:00"],
            "home_team": pd.array(["Arsenal", "Leeds"], dtype="string"),
            "away_team": pd.array(["Chelsea", "Hull"], dtype="string"),
        }
    )
    # football-data lists only the imminent round.
    extra = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-22")],
            "home_team": pd.array(["Arsenal"], dtype="string"),
            "away_team": pd.array(["Chelsea"], dtype="string"),
            "odds_h": [2.0], "odds_d": [3.5], "odds_a": [4.0],
        }
    )
    monkeypatch.setattr(P, "load_upcoming_fixtures", lambda **k: base)
    monkeypatch.setattr(P, "load_fixtures", lambda **k: extra)
    out = P.upcoming_fixtures()
    assert len(out) == 2, "the far fixture must survive the merge"
    assert out.loc[out["home_team"] == "Arsenal", "odds_h"].iloc[0] == 2.0
    assert pd.isna(out.loc[out["home_team"] == "Leeds", "odds_h"].iloc[0])
