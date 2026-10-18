"""Project-wide paths and constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

for _d in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# football-data.co.uk season codes, oldest first. Shot/corner/card detail
# starts in 2000-01, which is why that is the default history window.
FIRST_SEASON = 2000
LAST_SEASON = 2026  # season starting year; 2026 == the 2026/27 campaign

RESULTS = ("H", "D", "A")
LABEL_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}
