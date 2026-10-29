"""Render a static page: upcoming predictions, how the model is doing, and why.

The page is deliberately plain HTML with inline CSS and no JavaScript. Nothing
on it changes between visitors or between page loads, so there is nothing for
a client-side framework to do: predictions move only when fixtures and results
move, which is once or twice a week.
"""

from __future__ import annotations

import html
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .config import REPORTS, ROOT
from .evaluate import summarise

SITE_DIR = ROOT / "site"

# Charts copied next to the page. Missing ones are skipped rather than fatal.
CHARTS = [
    ("rps_by_season.png", "Out-of-sample RPS by season",
     "Lower is better. The bookmakers' closing price (purple) sits below every "
     "model in every season tested."),
    ("calibration_xgb.png", "Calibration",
     "When the model says 70%, it happens about 70% of the time. Points on the "
     "dashed line are perfectly calibrated."),
]

CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --bg:#ffffff; --panel:#f7f8fa; --line:#e3e6ea; --ink:#15181d; --muted:#5d646e;
  --home:#2f6fdb; --draw:#e07b39; --away:#49a078; --warn-bg:#fff8e8; --warn-line:#e8c87a;
  --radius:10px;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#13161a; --panel:#1b1f25; --line:#2b313a; --ink:#e8ebef; --muted:#9aa3ae;
  --home:#5b91e8; --draw:#f0945a; --away:#63bb92; --warn-bg:#2a2317; --warn-line:#5c4a22;
}}
:root[data-theme="dark"]{
  --bg:#13161a; --panel:#1b1f25; --line:#2b313a; --ink:#e8ebef; --muted:#9aa3ae;
  --home:#5b91e8; --draw:#f0945a; --away:#63bb92; --warn-bg:#2a2317; --warn-line:#5c4a22;
}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
.wrap{max-width:860px; margin:0 auto; padding:48px 16px 72px}
header{border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:36px}
h1{font-size:1.75rem; margin:0 0 6px; letter-spacing:-.02em}
h2{font-size:1.15rem; margin:44px 0 14px; letter-spacing:-.01em}
h3{font-size:.95rem; margin:28px 0 8px}
p{margin:0 0 14px}
.sub{color:var(--muted); font-size:.9rem; margin:0}
.muted{color:var(--muted)}
small{font-size:.82rem}
a{color:var(--home)}
.note{
  background:var(--warn-bg); border:1px solid var(--warn-line);
  border-radius:var(--radius); padding:14px 16px; margin:0 0 8px; font-size:.9rem;
}
.note strong{display:block; margin-bottom:4px}
table{border-collapse:collapse; width:100%; font-size:.9rem}
th,td{text-align:right; padding:8px 10px; border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted); font-weight:600; font-size:.8rem; text-transform:uppercase; letter-spacing:.04em}
tr.best td{font-weight:600}
.scroll{overflow-x:auto; -webkit-overflow-scrolling:touch}
.fixture{
  border:1px solid var(--line); border-radius:var(--radius);
  padding:14px 16px; margin-bottom:10px; background:var(--panel);
}
.fx-top{display:flex; justify-content:space-between; align-items:baseline; gap:12px; margin-bottom:10px}
.teams{font-weight:600}
.kick{color:var(--muted); font-size:.82rem; white-space:nowrap}
.bar{display:flex; height:26px; border-radius:6px; overflow:hidden; background:var(--line)}
.seg{display:flex; align-items:center; justify-content:center;
     font-size:.75rem; font-weight:600; color:#fff; min-width:0; white-space:nowrap}
.seg.h{background:var(--home)} .seg.d{background:var(--draw)} .seg.a{background:var(--away)}
.key{display:flex; gap:16px; flex-wrap:wrap; margin:0 0 14px; font-size:.8rem; color:var(--muted)}
.key i{display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px}
figure{margin:0 0 28px}
figure img{width:100%; height:auto; border:1px solid var(--line); border-radius:var(--radius); display:block}
figcaption{color:var(--muted); font-size:.84rem; margin-top:8px}
.res{
  border:1px solid var(--line); border-radius:var(--radius);
  padding:12px 14px; margin-bottom:8px; background:var(--panel);
}
.res-top{display:flex; justify-content:space-between; align-items:baseline; gap:10px; margin-bottom:8px}
.res-top .teams{font-weight:600}
.score{font-variant-numeric:tabular-nums; font-weight:700; margin-left:8px}
.verdict{font-size:.8rem; font-weight:700; white-space:nowrap}
.hit{color:var(--away)} .miss{color:var(--muted)}
.bar.small{height:20px}
.seg.dim{opacity:.32}
.seg.mark{box-shadow:inset 0 0 0 2px rgba(255,255,255,.85)}
.stats{display:flex; flex-wrap:wrap; gap:10px; margin:0 0 18px}
.stat{
  flex:1 1 150px; border:1px solid var(--line); border-radius:var(--radius);
  padding:12px 14px; background:var(--panel);
}
.stat b{display:block; font-size:1.5rem; line-height:1.2; letter-spacing:-.02em}
.stat span{color:var(--muted); font-size:.78rem}
.pill{
  display:inline-flex; align-items:center; justify-content:center;
  width:26px; height:26px; border-radius:6px; margin-right:4px;
  font-size:.78rem; font-weight:700; color:#fff;
}
.pill.W{background:var(--away)} .pill.D{background:var(--draw)} .pill.L{background:#c2504a}
.formrow{display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:14px}
.back{display:inline-block; margin-bottom:18px; font-size:.88rem}
.ltab{width:100%; border-collapse:collapse; font-size:.9rem;
      font-variant-numeric:tabular-nums}
.ltab th,.ltab td{padding:9px 6px; border-bottom:1px solid var(--line); text-align:right}
.ltab th{color:var(--muted); font-weight:600; font-size:.75rem;
         text-transform:uppercase; letter-spacing:.04em}
.ltab th:nth-child(2),.ltab td:nth-child(2){text-align:left; width:100%}
.ltab th:first-child,.ltab td:first-child{text-align:center; color:var(--muted);
         padding-left:10px; white-space:nowrap}
.ltab tbody tr:hover{background:var(--panel)}
.ltab a{color:inherit; text-decoration:none; font-weight:600}
.ltab a:hover{color:var(--home)}
.ltab .pts{font-weight:700}
.ltab td.zone{border-left:3px solid transparent}
.ltab td.z-eu{border-left-color:var(--away)}
.ltab td.z-rel{border-left-color:#c2504a}
.ltab .form{white-space:nowrap; padding-left:14px}
.ltab .pill{width:20px; height:20px; font-size:.68rem; border-radius:4px; margin-right:3px}
.k-eu{background:var(--away)} .k-rel{background:#c2504a}
@media (max-width:620px){
  .ltab .opt{display:none}
  .ltab th,.ltab td{padding:9px 4px}
}
.club-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:10px}
.club-grid a{
  display:block; padding:10px 12px; border:1px solid var(--line);
  border-radius:var(--radius); background:var(--panel);
  text-decoration:none; color:var(--ink); font-weight:600; font-size:.92rem;
}
.club-grid a:hover{border-color:var(--home)}
.club-grid span{display:block; font-weight:400; color:var(--muted); font-size:.78rem}
.teams a{color:inherit; text-decoration:none; border-bottom:1px dotted var(--muted)}
.teams a:hover{border-bottom-color:var(--home); color:var(--home)}
.honours{columns:2; column-gap:24px; font-size:.88rem}
.honours div{break-inside:avoid; margin-bottom:10px}
.honours b{display:block}
.honours span{color:var(--muted); font-size:.82rem}
@media (max-width:560px){.honours{columns:1}}
footer{margin-top:56px; padding-top:20px; border-top:1px solid var(--line);
       color:var(--muted); font-size:.84rem}
@media (max-width:560px){
  .wrap{padding:28px 16px 56px}
  h1{font-size:1.4rem}
  .fx-top{flex-direction:column; gap:2px}
  .seg{font-size:.7rem}
}
"""


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _team_link(team: str, linked: set[str] | None) -> str:
    """Club name, linked to its page when one exists."""
    safe = html.escape(str(team))
    if not linked or team not in linked:
        return safe
    from .clubs import slug
    return f'<a href="clubs/{slug(team)}.html">{safe}</a>'


def _fixture_card(row: pd.Series, linked: set[str] | None = None) -> str:
    probs = [(row["p_home"], "h"), (row["p_draw"], "d"), (row["p_away"], "a")]
    # Labels are dropped from slivers too narrow to hold them.
    segs = "".join(
        f'<div class="seg {cls}" style="width:{p * 100:.2f}%">'
        f'{_pct(p) if p >= 0.12 else ""}</div>'
        for p, cls in probs
    )
    kick = pd.Timestamp(row["date"]).strftime("%a %d %b")
    if pd.notna(row.get("time")):
        kick += f" &middot; {html.escape(str(row['time']))}"
    return f"""      <div class="fixture">
        <div class="fx-top">
          <span class="teams">{_team_link(row['home_team'], linked)} v {_team_link(row['away_team'], linked)}</span>
          <span class="kick">{kick}</span>
        </div>
        <div class="bar">{segs}</div>
      </div>"""


def _table(df: pd.DataFrame, cols: dict[str, str], best: str | None = None) -> str:
    head = "".join(f"<th>{html.escape(v)}</th>" for v in cols.values())
    rows = []
    for _, r in df.iterrows():
        cls = ' class="best"' if best and r.iloc[0] == best else ""
        cells = "".join(f"<td>{r[c]}</td>" for c in cols)
        rows.append(f"<tr{cls}>{cells}</tr>")
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _performance(preds: pd.DataFrame) -> tuple[str, str]:
    """Overall backtest table, and the current season's running accuracy."""
    names = {"xgb": "This model", "market": "Bookmakers (closing odds)",
             "elo": "Elo rating only", "base": "Always pick the home side"}
    overall = summarise(preds)
    overall = overall[overall["model"].isin(names)].copy()
    overall["Model"] = overall["model"].map(names)
    overall["Accuracy"] = overall["accuracy"].map(_pct)
    overall["RPS"] = overall["rps"].map(lambda v: f"{v:.4f}")
    overall["Matches"] = overall["n"].map(lambda v: f"{v:,}")
    overall = overall.sort_values("rps")
    table = _table(
        overall[["Model", "Matches", "Accuracy", "RPS"]],
        {"Model": "Model", "Matches": "Matches", "Accuracy": "Accuracy", "RPS": "RPS"},
        best="Bookmakers (closing odds)",
    )

    season = preds["season"].max()
    cur = preds[preds["season"] == season]
    rows = []
    for key, label in names.items():
        sub = cur[cur["model"] == key]
        if not sub.empty:
            rows.append({"Model": label, "Correct": f"{int(sub['correct'].sum())} / {len(sub)}",
                         "Accuracy": _pct(sub["correct"].mean()), "_s": sub["correct"].mean()})
    if not rows:
        return table, ""
    cur_df = pd.DataFrame(rows).sort_values("_s", ascending=False)
    cur_table = _table(cur_df[["Model", "Correct", "Accuracy"]],
                       {"Model": "Model", "Correct": "Correct", "Accuracy": "Accuracy"})
    return table, f"<h3>{html.escape(season)} so far</h3>{cur_table}"


_OUTCOME_IDX = {"H": 0, "D": 1, "A": 2}


def _result_card(row: pd.Series, linked: set[str] | None = None) -> str:
    """One settled prediction: what was said, what happened, hit or miss."""
    probs = [(float(row["p_h"]), "h"), (float(row["p_d"]), "d"), (float(row["p_a"]), "a")]
    actual = str(row["actual"])
    shown = _OUTCOME_IDX.get(actual)
    segs = []
    for i, (prob, cls) in enumerate(probs):
        # The bar is dimmed except for the outcome that actually happened, so
        # the eye goes straight to the probability that mattered.
        extra = " mark" if i == shown else " dim"
        label = _pct(prob) if prob >= 0.14 else ""
        segs.append(f'<div class="seg {cls}{extra}" style="width:{prob * 100:.2f}%">{label}</div>')

    hit = bool(row["correct"])
    verdict = ('<span class="verdict hit">&#10003; hit</span>' if hit
               else '<span class="verdict miss">&#10007; miss</span>')
    gave = probs[shown][0] if shown is not None else float("nan")
    score = ""
    if pd.notna(row.get("home_goals")) and pd.notna(row.get("away_goals")):
        score = f'<span class="score">{int(row["home_goals"])}&ndash;{int(row["away_goals"])}</span>'
    when = pd.Timestamp(row["date"]).strftime("%d %b")
    gave_txt = f"gave {_pct(gave)} to this result" if shown is not None else ""
    return f"""      <div class="res">
        <div class="res-top">
          <span class="teams">{_team_link(row['home_team'], linked)} v {_team_link(row['away_team'], linked)}{score}</span>
          {verdict}
        </div>
        <div class="bar small">{"".join(segs)}</div>
        <p style="margin:6px 0 0"><small class="muted">{when} &middot; {gave_txt}</small></p>
      </div>"""


def _results_section(logged: pd.DataFrame, limit: int = 20, linked: set[str] | None = None) -> str:
    """Scored predictions, newest first, with the running totals above them."""
    from .track import summarise_log

    if logged is None or logged.empty:
        return ""
    scored = logged[logged["correct"].notna()].sort_values("date", ascending=False)
    if scored.empty:
        return ("<section><h2>Track record</h2><p class=\"muted\">No predictions have "
                "been settled yet. This fills in as results come in.</p></section>")

    s = summarise_log(logged)
    published = s["by_source"].get("published", {})
    pub_line = ""
    if published:
        pub_line = (f'<div class="stat"><b>{published["mean"] * 100:.0f}%</b>'
                    f'<span>of {int(published["size"])} published live</span></div>')

    cards = "\n".join(_result_card(r, linked) for _, r in scored.head(limit).iterrows())
    more = ("" if len(scored) <= limit else
            f'<p><small class="muted">Showing the {limit} most recent of '
            f'{len(scored)} settled predictions.</small></p>')

    return f"""  <section>
    <h2>Track record</h2>
    <p>Every prediction is written down before the match and scored once the
    result is in. Nothing here is rewritten afterwards.</p>
    <div class="stats">
      <div class="stat"><b>{s["accuracy"] * 100:.0f}%</b><span>{s["correct"]} of {s["n"]} correct</span></div>
      <div class="stat"><b>{s["mean_prob_actual"] * 100:.0f}%</b><span>average probability given to what happened</span></div>
      {pub_line}
    </div>
{cards}
{more}
    <p><small class="muted">A &quot;hit&quot; means the most likely outcome was the
    one that occurred. Because draws are rarely any single match&#39;s most likely
    result, the average probability given to the actual outcome is the fairer
    measure of the two.</small></p>
  </section>

"""


# --------------------------------------------------------------------------
# club pages
# --------------------------------------------------------------------------
def _form_pills(form: pd.DataFrame) -> str:
    if form.empty:
        return '<p class="muted">No matches played yet this season.</p>'
    pills = "".join(
        f'<span class="pill {r.outcome}" title="{html.escape(str(r.opponent))} '
        f'{int(r.gf)}-{int(r.ga)}">{r.outcome}</span>'
        for r in form.itertuples()
    )
    rows = []
    for r in form.itertuples():
        where = "H" if r.is_home else "A"
        rows.append({
            "Date": pd.Timestamp(r.date).strftime("%d %b"),
            "": where,
            "Opponent": html.escape(str(r.opponent)),
            "Score": f"{int(r.gf)}&ndash;{int(r.ga)}",
            "Result": r.outcome,
        })
    table = _table(pd.DataFrame(rows), {c: c for c in rows[0]})
    return f'<div class="formrow">{pills}<small class="muted">most recent first</small></div>{table}'


def _player_table(
    df: pd.DataFrame, value: str, label: str, expected: str, exp_label: str,
    *, empty: str | None = None,
) -> str:
    if df.empty:
        msg = empty or f"No {label.lower()} recorded yet this season."
        return f'<p class="muted">{msg}</p>'
    cols = {
        "Player": [html.escape(str(p)) for p in df["player"]],
        "Apps": df["apps"].astype(int).to_numpy(),
        "Mins": df["minutes"].astype(int).to_numpy(),
        label: df[value].astype(int).to_numpy(),
        exp_label: df[expected].astype(float).round(1).to_numpy(),
    }
    # Present only on a multi-season frame, so one renderer serves both.
    if "seasons" in df.columns:
        cols["Seasons"] = df["seasons"].astype(int).to_numpy()
    show = pd.DataFrame(cols)
    return _table(show, {c: c for c in show.columns})


def _keeper_table(keepers: pd.DataFrame, empty: str) -> str:
    if keepers.empty:
        return f'<p class="muted">{empty}</p>'
    cols = {
        "Goalkeeper": [html.escape(str(p)) for p in keepers["player"]],
        "Starts": keepers["starts"].astype(int).to_numpy(),
        "Clean sheets": keepers["clean_sheets"].astype(int).to_numpy(),
    }
    if "seasons" in keepers.columns:
        cols["Seasons"] = keepers["seasons"].astype(int).to_numpy()
    kt = pd.DataFrame(cols)
    return _table(kt, {c: c for c in kt.columns})


# How each club's record table counts, spelled out rather than left as jargon.
_BASIS_NOTE = {
    "all competitions": "all competitions",
    "competitive matches": ("competitive matches only, so friendlies and most "
                            "wartime fixtures are excluded"),
    "league goals only": "league goals only, so cup scorers are understated",
}


def _all_time(entry: dict, wiki_url: str) -> str:
    """The curated club record-scorer table."""
    players = entry.get("players") or []
    if not players:
        link = f' <a href="{html.escape(wiki_url)}">Wikipedia</a>' if wiki_url else ""
        return ('<p class="muted">No all-time scoring record recorded. See'
                f'{link or " the club&#39;s page"}.</p>')
    rows = []
    for i, pl in enumerate(players, start=1):
        years = html.escape(str(pl.get("years") or "")) or "&ndash;"
        rows.append(
            f'<tr><td>{i}</td><td>{html.escape(str(pl["player"]))}</td>'
            f'<td>{int(pl["goals"])}</td><td>{years}</td></tr>'
        )
    bits = []
    basis = entry.get("_basis", "")
    if basis:
        bits.append(f"Counts {_BASIS_NOTE.get(basis, basis)}.")
    if entry.get("_note"):
        bits.append(html.escape(entry["_note"]))
    tail = f'<p><small class="muted">{" ".join(bits)}</small></p>' if bits else ""
    body = "".join(rows)
    return ('<div class="scroll"><table><thead><tr><th>#</th><th>Player</th>'
            f'<th>Goals</th><th>Years</th></tr></thead><tbody>{body}</tbody>'
            f'</table></div>{tail}')


def _honours(trophies: dict, wiki_url: str) -> str:
    comps = {k: v for k, v in trophies.items() if not k.startswith("_") and v}
    note = trophies.get("_note", "")
    note_html = f'<p><small class="muted">{html.escape(note)}</small></p>' if note else ""
    if not comps:
        link = (f' <a href="{html.escape(wiki_url)}">Wikipedia</a>' if wiki_url else "")
        return note_html or ('<p class="muted">No major senior honours recorded. See'
                             f'{link or " the club&#39;s page"} for the full list.</p>')
    blocks = []
    for comp, years in sorted(comps.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        yrs = ", ".join(str(y) for y in sorted(years))
        blocks.append(
            f'<div><b>{html.escape(comp)} &times;{len(years)}</b><span>{yrs}</span></div>'
        )
    return f'<div class="honours">{"".join(blocks)}</div>{note_html}'


def _club_page(profile: dict, season_label: str, repo_url: str | None) -> str:
    team = profile["team"]
    s = profile["season"]
    wiki = profile.get("wiki") or {}
    pos = profile["position"]

    if s.get("played"):
        tiles = [
            (f'{pos}' if pos else "&ndash;", "in the table"),
            (f'{s["points"]}', f'points from {s["played"]} matches'),
            (f'{s["won"]}&ndash;{s["drawn"]}&ndash;{s["lost"]}', "won / drawn / lost"),
            (f'{s["clean_sheets"]}', "clean sheets"),
            (f'{s["goals_for"]}:{s["goals_against"]}', "goals for / against"),
        ]
        stats = '<div class="stats">' + "".join(
            f'<div class="stat"><b>{v}</b><span>{l}</span></div>' for v, l in tiles
        ) + "</div>"
    else:
        stats = '<p class="muted">The season has not started for this club yet.</p>'

    blurb = ""
    if wiki.get("extract"):
        link = (f' <a href="{html.escape(wiki["url"])}">Read more on Wikipedia</a>.'
                if wiki.get("url") else "")
        blurb = (f'<p>{html.escape(wiki["extract"])}</p>'
                 f'<p><small class="muted">Summary from Wikipedia, CC BY-SA.{link}</small></p>')

    empty_frame = pd.DataFrame()
    scorers_html = _player_table(profile["scorers"], "goals", "Goals", "xg", "xG")
    assists_html = _player_table(profile["assists"], "assists", "Assists", "xa", "xA")
    keeper_html = _keeper_table(
        profile["keepers"], "No goalkeeper appearances recorded yet."
    )
    career_scorers_html = _player_table(
        profile.get("career_scorers", empty_frame), "goals", "Goals", "xg", "xG",
        empty="No goals on record for this club over the period covered.",
    )
    career_assists_html = _player_table(
        profile.get("career_assists", empty_frame), "assists", "Assists", "xa", "xA",
        empty="No assists on record for this club over the period covered.",
    )
    career_keeper_html = _keeper_table(
        profile.get("career_keepers", empty_frame),
        "No goalkeeper appearances on record over the period covered.",
    )
    career_label = profile.get("career_label") or "recent seasons"
    all_time_html = _all_time(profile.get("all_time", {}), wiki.get("url", ""))

    repo_line = (f'<a href="{html.escape(repo_url)}">Source and methodology</a> &middot; '
                 if repo_url else "")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(team)} &mdash; {html.escape(season_label)}</title>
<meta name="description" content="{html.escape(team)} season statistics: form, top scorers, assists, clean sheets and honours.">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <a class="back" href="../index.html">&larr; All fixtures and predictions</a>
  <h1>{html.escape(team)}</h1>
  <p class="sub">{html.escape(season_label)} season</p>
</header>

<main>
  <section>
{stats}
{blurb}
  </section>

  <section>
    <h2>Recent form</h2>
{_form_pills(profile["form"])}
  </section>

  <section>
    <h2>This season</h2>
    <h3>Top scorers</h3>
{scorers_html}
    <h3>Most assists</h3>
{assists_html}
    <h3>Clean sheets</h3>
{keeper_html}
    <p><small class="muted">xG and xA are expected goals and assists &mdash; the
    quality of the chances created, which says more about repeatable performance
    than the raw count.</small></p>
  </section>

  <section>
    <h2>Premier League record, {career_label}</h2>
    <h3>Top scorers</h3>
{career_scorers_html}
    <h3>Most assists</h3>
{career_assists_html}
    <h3>Clean sheets</h3>
{career_keeper_html}
    <p><small class="muted">For this club only: a player who moved is credited
    here with what he scored in these colours, not his career total. Computed
    from Understat appearance data, which begins in 2014-15 &mdash; so it is a
    record over that window and not an all-time one.</small></p>
  </section>

  <section>
    <h2>All-time club record scorers</h2>
{all_time_html}
    <p><small class="muted">A club&#39;s all-time scorers reach back long before
    any free appearance feed, so this list is curated rather than computed: taken
    from the club&#39;s records article, and from the club&#39;s own site where it
    publishes a list. Ranked on goals, which is how the sources rank them.</small></p>
  </section>

  <section>
    <h2>Honours</h2>
{_honours(profile["trophies"], wiki.get("url", ""))}
    <p><small class="muted">Major senior honours, hand-compiled: unlike every
    other figure on this site these are not computed from data, because no free
    structured feed carries club honours. Treat them as indicative and check
    the club&#39;s Wikipedia page for the authoritative list.</small></p>
  </section>
</main>

<footer>
  {repo_line}Match data from football-data.co.uk, player data from Understat.
  For interest only &mdash; not betting advice.
</footer>
</div>
</body>
</html>
"""


def build_club_pages(
    profiles: list[dict], season_label: str, out_dir: Path, repo_url: str | None = None
) -> list[Path]:
    """Write one page per club into ``<out_dir>/clubs``."""
    club_dir = Path(out_dir) / "clubs"
    club_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale pages first: clubs get relegated and slugs change, and an
    # orphan page would otherwise sit there being wrong for years.
    expected = {f"{p['slug']}.html" for p in profiles}
    for old_page in club_dir.glob("*.html"):
        if old_page.name not in expected:
            old_page.unlink()
    written = []
    for profile in profiles:
        dest = club_dir / f"{profile['slug']}.html"
        dest.write_text(_club_page(profile, season_label, repo_url), encoding="utf-8")
        written.append(dest)
    return written


def _club_index(profiles: list[dict]) -> str:
    """Grid of club links, used before any match has been played."""
    if not profiles:
        return ""
    ordered = sorted(profiles, key=lambda p: (p["position"] or 99, p["team"]))
    cards = []
    for p in ordered:
        s = p["season"]
        line = (f'{p["position"]}. {s["points"]} pts from {s["played"]}'
                if p["position"] and s.get("played") else "season not started")
        cards.append(
            f'<a href="clubs/{p["slug"]}.html">{html.escape(p["team"])}<span>{line}</span></a>'
        )
    return f"""  <section>
    <h2>Clubs</h2>
    <p>Form, scorers, assists, clean sheets and honours for every club this season.</p>
    <div class="club-grid">
{"".join(cards)}
    </div>
  </section>

"""


# Places marked at each end of the table. Which European competition a place
# leads to depends on the cup winners and on UEFA's coefficient spots, so the
# top group is marked as "European places" rather than named.
EURO_PLACES = 5
RELEGATION_PLACES = 3


def _table_form(form: pd.DataFrame) -> str:
    """Last five results as pills, oldest first - how a form guide reads."""
    if form.empty:
        return ""
    return "".join(
        f'<span class="pill {r.outcome}" title="{html.escape(str(r.opponent))} '
        f'{int(r.gf)}-{int(r.ga)}">{r.outcome}</span>'
        for r in form.iloc[::-1].itertuples()
    )


def _league_table(profiles: list[dict]) -> str:
    """The league table: standings and the way in to every club page."""
    ranked = [p for p in profiles if p.get("position") and p["season"].get("played")]
    if not ranked:
        # Nothing has been played yet, so there is no table to show.
        return _club_index(profiles)
    ordered = sorted(ranked, key=lambda p: p["position"])
    last = len(ordered)

    rows = []
    for p in ordered:
        s, pos = p["season"], p["position"]
        if pos <= EURO_PLACES:
            zone = " z-eu"
        elif pos > last - RELEGATION_PLACES:
            zone = " z-rel"
        else:
            zone = ""
        rows.append(
            f'<tr><td class="zone{zone}">{pos}</td>'
            f'<td><a href="clubs/{p["slug"]}.html">{html.escape(p["team"])}</a></td>'
            f'<td>{s["played"]}</td>'
            f'<td class="opt">{s["won"]}</td>'
            f'<td class="opt">{s["drawn"]}</td>'
            f'<td class="opt">{s["lost"]}</td>'
            f'<td class="opt">{s["goals_for"]}</td>'
            f'<td class="opt">{s["goals_against"]}</td>'
            f'<td>{s["goal_difference"]:+d}</td>'
            f'<td class="pts">{s["points"]}</td>'
            f'<td class="form opt">{_table_form(p["form"])}</td></tr>'
        )

    head = ('<tr><th>#</th><th>Club</th><th>P</th><th class="opt">W</th>'
            '<th class="opt">D</th><th class="opt">L</th><th class="opt">GF</th>'
            '<th class="opt">GA</th><th>GD</th><th>Pts</th>'
            '<th class="opt">Form</th></tr>')
    body = "".join(rows)
    return f"""  <section>
    <h2>Premier League table</h2>
    <p>Click any club for its form, scorers, assists, clean sheets and honours.</p>
    <p class="key"><span><i class="k-eu"></i>European places</span>
      <span><i class="k-rel"></i>Relegation</span></p>
    <div class="scroll"><table class="ltab"><thead>{head}</thead>
      <tbody>{body}</tbody></table></div>
    <p><small class="muted">Computed from results, not taken from a feed, so it
    reflects exactly the matches this site knows about. Which European competition
    each of the top five places leads to depends on the cup winners, so they are
    marked as one group. Form runs oldest to newest; hover a result for the
    score.</small></p>
  </section>

"""


def build(
    fixtures: pd.DataFrame,
    preds: pd.DataFrame | None = None,
    *,
    logged: pd.DataFrame | None = None,
    profiles: list[dict] | None = None,
    season_label: str = "",
    out_dir: Path | None = None,
    repo_url: str | None = None,
) -> Path:
    """Write the site to ``out_dir`` and return the path to index.html."""
    out_dir = Path(out_dir) if out_dir else SITE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    linked: set[str] = set()
    if profiles:
        build_club_pages(profiles, season_label, out_dir, repo_url)
        linked = {p["team"] for p in profiles}

    if fixtures.empty:
        cards = ('<p class="muted">No Premier League fixtures are published right now. '
                 "This page refreshes when the next round is listed.</p>")
    else:
        cards = "\n".join(_fixture_card(r, linked) for _, r in fixtures.iterrows())

    perf_table, cur_table = _performance(preds) if preds is not None else ("", "")
    results = _results_section(logged, linked=linked) if logged is not None else ""
    clubs = _league_table(profiles) if profiles else ""

    figures = []
    for name, title, caption in CHARTS:
        src = REPORTS / name
        if not src.exists():
            continue
        shutil.copyfile(src, out_dir / name)
        figures.append(
            f"""      <figure>
        <h3>{html.escape(title)}</h3>
        <img src="{name}" alt="{html.escape(title)}" loading="lazy">
        <figcaption>{html.escape(caption)}</figcaption>
      </figure>"""
        )

    built = datetime.now(UTC).strftime("%d %B %Y, %H:%M UTC")
    repo_line = (f'<a href="{html.escape(repo_url)}">Source and full methodology</a> &middot; '
                 if repo_url else "")

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Premier League Match Predictor</title>
<meta name="description" content="Home, draw and away probabilities for upcoming Premier League fixtures, with an honest record of how the model performs against bookmakers.">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Premier League Match Predictor</h1>
  <p class="sub">Gradient-boosted model trained on 27 seasons. Updated {built}.</p>
</header>

<main>
  <section>
    <h2>Upcoming fixtures</h2>
    <div class="key">
      <span><i style="background:var(--home)"></i>Home win</span>
      <span><i style="background:var(--draw)"></i>Draw</span>
      <span><i style="background:var(--away)"></i>Away win</span>
    </div>
{cards}
    <p><small class="muted">Probabilities are produced before kick-off from team
    form, Elo ratings and expected goals. They never see bookmaker prices.</small></p>
  </section>

{clubs}{results}  <section>
    <h2>How it actually performs</h2>
    <div class="note">
      <strong>This model does not beat the bookmakers.</strong>
      Across 6,140 matches it has never finished a season ahead of the closing
      odds &mdash; not once in 17 seasons. Betting its predictions returns
      &minus;3.87%, which is worse than staking at random. It is a modelling
      exercise, not a betting system.
    </div>
    <p>Every number below is out-of-sample: each season was predicted by a model
    trained only on matches played before that season started.</p>
{perf_table}
{cur_table}
    <p><small class="muted">RPS is the ranked probability score, the standard
    measure for football forecasts &mdash; it rewards getting the whole
    probability spread right, not just the single most likely result. Lower is
    better.</small></p>
  </section>

  <section>
    <h2>The record in detail</h2>
{chr(10).join(figures)}
  </section>
</main>

<footer>
  {repo_line}Built from football-data.co.uk and Understat.
  For interest only &mdash; not betting advice.
</footer>
</div>
</body>
</html>
"""
    dest = out_dir / "index.html"
    dest.write_text(page, encoding="utf-8")
    return dest
