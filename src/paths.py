from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
REPORT_DIR = ROOT / "report"
PRESENTATION_DIR = ROOT / "presentation"

RAW_CSV = DATA_DIR / "srilanka_dengue_climate_weekly_2006_2025.csv"
KAGGLE_SLUG = "shinthurie/sri-lanka-weekly-dengue-climate-2006-2025"

for _d in (DATA_DIR, ARTIFACTS, FIGURES, REPORT_DIR, PRESENTATION_DIR):
    _d.mkdir(parents=True, exist_ok=True)
