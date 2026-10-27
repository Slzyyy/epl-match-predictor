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


def _fixture_card(row: pd.Series) -> str:
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
          <span class="teams">{html.escape(str(row['home_team']))} v {html.escape(str(row['away_team']))}</span>
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


def _result_card(row: pd.Series) -> str:
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
          <span class="teams">{html.escape(str(row['home_team']))} v {html.escape(str(row['away_team']))}{score}</span>
          {verdict}
        </div>
        <div class="bar small">{"".join(segs)}</div>
        <p style="margin:6px 0 0"><small class="muted">{when} &middot; {gave_txt}</small></p>
      </div>"""


def _results_section(logged: pd.DataFrame, limit: int = 20) -> str:
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

    cards = "\n".join(_result_card(r) for _, r in scored.head(limit).iterrows())
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


def build(
    fixtures: pd.DataFrame,
    preds: pd.DataFrame | None = None,
    *,
    logged: pd.DataFrame | None = None,
    out_dir: Path | None = None,
    repo_url: str | None = None,
) -> Path:
    """Write the site to ``out_dir`` and return the path to index.html."""
    out_dir = Path(out_dir) if out_dir else SITE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if fixtures.empty:
        cards = ('<p class="muted">No Premier League fixtures are published right now. '
                 "This page refreshes when the next round is listed.</p>")
    else:
        cards = "\n".join(_fixture_card(r) for _, r in fixtures.iterrows())

    perf_table, cur_table = _performance(preds) if preds is not None else ("", "")
    results = _results_section(logged) if logged is not None else ""

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

{results}  <section>
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
