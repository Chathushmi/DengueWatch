"""Cleaning, feature engineering and chronological split.

Convention: a row describes a *forecast week* (the week we predict). A lag of k
means "k weeks before the forecast week", so lag 1 is the latest week whose data
is already known when the forecast is made.

Target: target_delta = log1p(cases in forecast week) - log1p(cases 1 week before),
i.e. the log growth of cases. Forecast cases = expm1(log1p(cases_lag1) + prediction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.paths import RAW_CSV

DISTRICTS = ["Colombo", "Gampaha", "Kandy"]
RAIN_LAGS = list(range(1, 13))
CLIMATE_LAGS = [1, 4, 8, 12]
CASE_LAGS = [1, 2, 4]

# Chronological split on whole weeks: first 65% train, next 15% validation, last 20% test.
TRAIN_FRAC = 0.65
VAL_FRAC = 0.15

CASE_FEATURES = [f"cases_lag{k}" for k in CASE_LAGS] + ["growth_1w", "growth_3w"]
CLIMATE_FEATURES = (
    [f"rain_lag{k}" for k in RAIN_LAGS]
    + ["rain_sum_4w", "rain_sum_8w"]
    + [f"humidity_lag{k}" for k in CLIMATE_LAGS]
    + [f"temp_lag{k}" for k in CLIMATE_LAGS]
)
CALENDAR_FEATURES = ["week_of_year", "season", "district"]
CATEGORICAL = ["season", "district"]

ALL_FEATURES = CASE_FEATURES + CLIMATE_FEATURES + CALENDAR_FEATURES
CLIMATE_ONLY_FEATURES = CLIMATE_FEATURES + CALENDAR_FEATURES

FEATURE_LABELS = {
    "cases_lag1": "Cases 1 week before",
    "cases_lag2": "Cases 2 weeks before",
    "cases_lag4": "Cases 4 weeks before",
    "growth_1w": "Case trend over last week (log change)",
    "growth_3w": "Case trend over last 3 weeks (log change)",
    "rain_sum_4w": "Rainfall total, last 4 weeks",
    "rain_sum_8w": "Rainfall total, last 8 weeks",
    "week_of_year": "Week of year",
    "season": "Monsoon season",
    "district": "District",
    **{f"rain_lag{k}": f"Rainfall {k} week{'s' if k > 1 else ''} before" for k in RAIN_LAGS},
    **{f"humidity_lag{k}": f"Humidity {k} week{'s' if k > 1 else ''} before" for k in CLIMATE_LAGS},
    **{f"temp_lag{k}": f"Mean temp {k} week{'s' if k > 1 else ''} before" for k in CLIMATE_LAGS},
}


def season_of_month(month: int) -> str:
    """Sri Lanka's four climatic seasons (Department of Meteorology)."""
    if month in (12, 1, 2):
        return "NE monsoon"
    if month in (3, 4):
        return "First inter-monsoon"
    if month in (5, 6, 7, 8, 9):
        return "SW monsoon"
    return "Second inter-monsoon"


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["week_start_date", "week_end_date"])
    df = df.drop_duplicates(["district", "week_start_date"])
    df = df.sort_values(["district", "week_start_date"]).reset_index(drop=True)
    df["cases"] = df["cases"].clip(lower=0).astype(int)
    return df


def recompute_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Derived columns that depend on other features (used again by the app's what-if)."""
    frame = frame.copy()
    frame["rain_sum_4w"] = frame[[f"rain_lag{k}" for k in range(1, 5)]].sum(axis=1)
    frame["rain_sum_8w"] = frame[[f"rain_lag{k}" for k in range(1, 9)]].sum(axis=1)
    frame["growth_1w"] = np.log1p(frame["cases_lag1"]) - np.log1p(frame["cases_lag2"])
    frame["growth_3w"] = np.log1p(frame["cases_lag1"]) - np.log1p(frame["cases_lag4"])
    frame["base_log"] = np.log1p(frame["cases_lag1"])
    return frame


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for district, g in raw.groupby("district", sort=False):
        g = g.sort_values("week_start_date").reset_index(drop=True)
        out = pd.DataFrame(
            {
                "district": district,
                "week_start_date": g["week_start_date"],
                "year": g["year"],
                "week_of_year": g["week"],
                "target_cases": g["cases"],
            }
        )
        for k in CASE_LAGS:
            out[f"cases_lag{k}"] = g["cases"].shift(k)
        for k in RAIN_LAGS:
            out[f"rain_lag{k}"] = g["rainfall_mm"].shift(k)
        for k in CLIMATE_LAGS:
            out[f"humidity_lag{k}"] = g["humidity_pct"].shift(k)
            out[f"temp_lag{k}"] = g["temp_avg_c"].shift(k)
        parts.append(out)

    feats = pd.concat(parts, ignore_index=True)
    feats = feats.dropna().reset_index(drop=True)
    for k in CASE_LAGS:
        feats[f"cases_lag{k}"] = feats[f"cases_lag{k}"].astype(int)
    feats = recompute_derived(feats)
    feats["season"] = feats["week_start_date"].dt.month.map(season_of_month)
    feats["target_log"] = np.log1p(feats["target_cases"])
    # the model learns the log change from the latest known week
    feats["target_delta"] = feats["target_log"] - feats["base_log"]
    feats = feats.sort_values(["week_start_date", "district"]).reset_index(drop=True)
    # cut on whole weeks so all districts of a week fall in the same split
    weeks = np.sort(feats["week_start_date"].unique())
    train_end = weeks[int(len(weeks) * TRAIN_FRAC) - 1]
    val_end = weeks[int(len(weeks) * (TRAIN_FRAC + VAL_FRAC)) - 1]
    feats["split"] = np.select(
        [feats["week_start_date"] <= train_end, feats["week_start_date"] <= val_end],
        ["train", "val"],
        default="test",
    )
    return feats


def load_features() -> pd.DataFrame:
    return build_features(load_raw())


def split(feats: pd.DataFrame):
    return (
        feats[feats.split == "train"],
        feats[feats.split == "val"],
        feats[feats.split == "test"],
    )


if __name__ == "__main__":
    f = load_features()
    tr, va, te = split(f)
    print(f.shape, len(ALL_FEATURES), "features")
    for name, part in (("train", tr), ("val", va), ("test", te)):
        print(name, len(part), part.week_start_date.min().date(), part.week_start_date.max().date())
