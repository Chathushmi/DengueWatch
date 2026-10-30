"""Tune and train the Explainable Boosting Machine, plus baselines and ablation."""
from __future__ import annotations

import itertools
import json
import time

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.features import (
    ALL_FEATURES,
    CATEGORICAL,
    CLIMATE_ONLY_FEATURES,
    DISTRICTS,
    load_features,
    split,
)
from src.paths import ARTIFACTS

SEED = 42
GRID = {
    "learning_rate": [0.005, 0.01, 0.02],
    "interactions": [0, 10],
    "max_leaves": [2, 3],
    "max_bins": [32, 64],
}
FIXED = {"outer_bags": 8, "min_samples_leaf": 4, "random_state": SEED, "n_jobs": -1}


def make_ebm(features: list[str], **params) -> ExplainableBoostingRegressor:
    feature_types = ["nominal" if f in CATEGORICAL else "continuous" for f in features]
    return ExplainableBoostingRegressor(
        feature_names=features, feature_types=feature_types, **FIXED, **params
    )


def predict_cases(model, frame: pd.DataFrame) -> np.ndarray:
    """Convert predicted log growth back to a case count."""
    features = list(model.feature_names_in_)
    return np.clip(np.expm1(frame["base_log"].to_numpy() + model.predict(frame[features])), 0, None)


def tune(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> tuple[dict, pd.DataFrame]:
    rows = []
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        params = dict(zip(keys, values))
        t0 = time.time()
        model = make_ebm(features, **params).fit(train[features], train["target_delta"])
        pred = predict_cases(model, val)
        rows.append(
            {
                **params,
                "val_mae": mean_absolute_error(val["target_cases"], pred),
                "val_mae_log": mean_absolute_error(val["target_log"], np.log1p(pred)),
                "fit_seconds": round(time.time() - t0, 1),
            }
        )
        print(f"  {params} -> val MAE {rows[-1]['val_mae']:.2f}")
    results = pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)
    best = {k: (float(v) if k == "learning_rate" else int(v)) for k, v in results.iloc[0][keys].items()}
    return best, results


def main() -> None:
    feats = load_features()
    train, val, test = split(feats)
    trainval = pd.concat([train, val])

    print("Tuning EBM on the validation slice ...")
    best, results = tune(train, val, ALL_FEATURES)
    results.to_csv(ARTIFACTS / "tuning_results.csv", index=False)
    print("Best params:", best)

    val_model = make_ebm(ALL_FEATURES, **best).fit(train[ALL_FEATURES], train["target_delta"])

    print("Refitting best EBM on train + validation ...")
    ebm = make_ebm(ALL_FEATURES, **best).fit(trainval[ALL_FEATURES], trainval["target_delta"])
    ebm_climate = make_ebm(CLIMATE_ONLY_FEATURES, **best).fit(
        trainval[CLIMATE_ONLY_FEATURES], trainval["target_log"]
    )

    # alert thresholds come from training-period data only
    thresholds = train.groupby("district")["target_cases"].quantile(0.75).round(1).to_dict()
    seasonal = (
        trainval.groupby(["district", "week_of_year"])["target_cases"].mean().rename("seasonal_avg").reset_index()
    )

    preds = feats[["week_start_date", "year", "week_of_year", "district", "split", "target_cases"]].copy()
    preds["ebm"] = predict_cases(ebm, feats)
    preds.loc[preds.split != "test", "ebm"] = np.nan
    val_mask = preds.split == "val"
    preds.loc[val_mask, "ebm_val_model"] = predict_cases(val_model, feats[val_mask])
    preds["ebm_climate_only"] = np.clip(np.expm1(ebm_climate.predict(feats[CLIMATE_ONLY_FEATURES])), 0, None)
    preds["persistence"] = feats["cases_lag1"].astype(float)
    preds = preds.merge(seasonal, on=["district", "week_of_year"], how="left")
    preds["seasonal_avg"] = preds["seasonal_avg"].fillna(preds["persistence"])
    preds["threshold"] = preds["district"].map(thresholds)
    preds.to_csv(ARTIFACTS / "predictions.csv", index=False)

    joblib.dump(ebm, ARTIFACTS / "ebm_model.pkl")
    joblib.dump(ebm_climate, ARTIFACTS / "ebm_climate_only.pkl")
    feats.to_csv(ARTIFACTS / "features.csv", index=False)

    info = {
        "algorithm": "ExplainableBoostingRegressor (interpret)",
        "target": "log1p(cases next week) - log1p(cases this week)",
        "best_params": best,
        "fixed_params": {k: v for k, v in FIXED.items() if k != "n_jobs"},
        "grid": GRID,
        "n_grid": int(len(results)),
        "features": ALL_FEATURES,
        "climate_only_features": CLIMATE_ONLY_FEATURES,
        "alert_thresholds": thresholds,
        "districts": DISTRICTS,
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "split_ranges": {
            name: [str(p.week_start_date.min().date()), str(p.week_start_date.max().date())]
            for name, p in (("train", train), ("val", val), ("test", test))
        },
        "best_val_mae": float(results.iloc[0]["val_mae"]),
        "val_persistence_mae": float(mean_absolute_error(val["target_cases"], val["cases_lag1"])),
    }
    (ARTIFACTS / "training_info.json").write_text(json.dumps(info, indent=2))
    print("Saved model and training_info.json")


if __name__ == "__main__":
    main()
