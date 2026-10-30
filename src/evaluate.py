"""Test-set metrics and plots for the EBM, baselines and climate-only ablation."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

import joblib

from src.features import DISTRICTS
from src.paths import ARTIFACTS, FIGURES
from src.train import predict_cases

MODELS = {
    "ebm": "EBM (full)",
    "ebm_climate_only": "EBM (climate only)",
    "persistence": "Persistence baseline",
    "seasonal_avg": "Seasonal-average baseline",
}

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})


def regression_metrics(y, p, mase_scale: float | None = None) -> dict:
    """mase_scale: MAE of the naive 'same as last week' forecast on the training data."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    err = p - y
    mae = float(np.abs(err).mean())
    out = {
        "MAE": mae,
        "MSE": float(mean_squared_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
        "MAE_log": float(mean_absolute_error(np.log1p(y), np.log1p(p))),
        "MedAE": float(median_absolute_error(y, p)),
        "WAPE": float(np.abs(err).sum() / y.sum()),
        # symmetric MAPE stays finite in zero-case weeks
        "sMAPE": float(np.mean(2 * np.abs(err) / np.maximum(np.abs(y) + np.abs(p), 1e-9))),
        "bias": float(err.mean()),
        "pearson_r": float(np.corrcoef(y, p)[0, 1]),
        "explained_variance": float(explained_variance_score(y, p)),
    }
    if mase_scale:
        out["MASE"] = mae / mase_scale
    return out


def direction_metrics(df: pd.DataFrame, col: str) -> dict:
    """Does the model get the direction of change (up / down vs this week) right?"""
    actual = np.sign(df["target_cases"] - df["persistence"])
    predicted = np.sign(df[col] - df["persistence"])
    moved = actual != 0
    big = (df["target_cases"] - df["persistence"]).abs() >= 0.25 * df["persistence"].clip(lower=4)
    return {
        "direction_accuracy": float((actual[moved] == predicted[moved]).mean()),
        "direction_accuracy_big_moves": float((actual[big] == predicted[big]).mean()),
        "n_big_moves": int(big.sum()),
    }


def alert_metrics(df: pd.DataFrame, col: str) -> dict:
    actual = (df["target_cases"] > df["threshold"]).astype(int)
    predicted = (df[col] > df["threshold"]).astype(int)
    score = df[col] / df["threshold"]
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(actual, score)),
        "PR_AUC": float(average_precision_score(actual, score)),
        "accuracy": float((tp + tn) / len(df)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "MCC": float(matthews_corrcoef(actual, predicted)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "positive_rate": float(actual.mean()),
    }


def period(df: pd.DataFrame) -> str:
    return f"{df.week_start_date.min():%b %Y} - {df.week_start_date.max():%b %Y}"


def plot_actual_vs_pred(test: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, d in zip(axes, DISTRICTS):
        g = test[test.district == d]
        ax.plot(g.week_start_date, g.target_cases, color="black", lw=1.4, label="Actual")
        ax.plot(g.week_start_date, g.ebm, color="tab:red", lw=1.4, label="EBM forecast")
        ax.plot(g.week_start_date, g.ebm_climate_only, color="tab:blue", lw=1, alpha=0.7,
                ls="--", label="EBM climate-only")
        ax.axhline(g.threshold.iloc[0], color="orange", ls=":", lw=1.2, label="Alert threshold")
        ax.set_title(f"{d} - weekly dengue cases, test period {period(test)}")
        ax.set_ylabel("Cases")
    axes[0].legend(loc="upper left", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "actual_vs_predicted.png")
    plt.close(fig)


def plot_validation(preds: pd.DataFrame) -> None:
    val = preds[preds.split == "val"]
    train = preds[preds.split == "train"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, d in zip(axes, DISTRICTS):
        g = val[val.district == d]
        ax.plot(g.week_start_date, g.target_cases, color="black", lw=1.4, label="Actual")
        ax.plot(g.week_start_date, g.ebm_val_model, color="tab:green", lw=1.4,
                label=f"EBM (trained on {period(train)})")
        ax.set_title(f"{d} - validation period {period(val)}")
        ax.set_ylabel("Cases")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "validation_forecast.png")
    plt.close(fig)


def plot_residuals(test: pd.DataFrame) -> None:
    resid = test.target_cases - test.ebm
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for d in DISTRICTS:
        m = test.district == d
        axes[0].scatter(test.ebm[m], resid[m], s=10, alpha=0.6, label=d)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("Predicted cases")
    axes[0].set_ylabel("Actual - predicted")
    axes[0].set_title("Residuals vs prediction")
    axes[0].legend(fontsize=8)
    axes[1].hist(resid, bins=40, color="tab:red", alpha=0.75)
    axes[1].set_xlabel("Residual (cases)")
    axes[1].set_title("Residual distribution")
    fig.tight_layout()
    fig.savefig(FIGURES / "residuals.png")
    plt.close(fig)


def plot_model_comparison(overall: dict, test: pd.DataFrame) -> None:
    names = [MODELS[k] for k in MODELS]
    mae = [overall[k]["MAE"] for k in MODELS]
    rmse = [overall[k]["RMSE"] for k in MODELS]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, mae, 0.4, label="MAE", color="tab:red")
    ax.bar(x + 0.2, rmse, 0.4, label="RMSE", color="tab:gray")
    for i, (a, b) in enumerate(zip(mae, rmse)):
        ax.text(i - 0.2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + 0.2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("Cases per week (lower is better)")
    ax.set_title(f"Test-set error by model ({period(test)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png")
    plt.close(fig)


def plot_confusion(alert: dict) -> None:
    c = alert["confusion"]
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(mat, cmap="Reds")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1], ["Normal", "High risk"])
    ax.set_yticks([0, 1], ["Normal", "High risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Outbreak alert - EBM (test)")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "alert_confusion.png")
    plt.close(fig)


def plot_scatter(test: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for d in DISTRICTS:
        g = test[test.district == d]
        ax.scatter(g.target_cases, g.ebm, s=10, alpha=0.6, label=d)
    lim = max(test.target_cases.max(), test.ebm.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="black", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual cases")
    ax.set_ylabel("Predicted cases")
    ax.set_title("EBM predicted vs actual (test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "pred_vs_actual_scatter.png")
    plt.close(fig)


def main() -> None:
    preds = pd.read_csv(ARTIFACTS / "predictions.csv", parse_dates=["week_start_date"])
    test = preds[preds.split == "test"].copy()
    val = preds[preds.split == "val"].copy()

    train = preds[preds.split == "train"]
    scale = float(mean_absolute_error(train.target_cases, train.persistence))
    overall = {k: regression_metrics(test.target_cases, test[k], scale) for k in MODELS}
    per_district = {
        d: {k: regression_metrics(g.target_cases, g[k], scale) for k in MODELS}
        for d, g in test.groupby("district")
    }
    alerts = {k: alert_metrics(test, k) for k in MODELS}
    val_metrics = regression_metrics(val.target_cases, val.ebm_val_model)
    val_metrics["persistence_MAE"] = float(mean_absolute_error(val.target_cases, val.persistence))
    direction = {k: direction_metrics(test, k) for k in ("ebm", "ebm_climate_only", "seasonal_avg")}

    feats = pd.read_csv(ARTIFACTS / "features.csv")
    ebm = joblib.load(ARTIFACTS / "ebm_model.pkl")
    tv = feats[feats.split.isin(["train", "val"])]
    tv_pred = predict_cases(ebm, tv)
    trainval_fit = regression_metrics(tv.target_cases, tv_pred)
    n_all = len(feats)
    split_pct = {
        "train": round(100 * (feats.split == "train").sum() / n_all, 1),
        "val": round(100 * (feats.split == "val").sum() / n_all, 1),
        "test": round(100 * (feats.split == "test").sum() / n_all, 1),
    }

    metrics = {
        "test_overall": overall,
        "test_direction": direction,
        "test_per_district": per_district,
        "test_alert": alerts,
        "validation_ebm": val_metrics,
        "trainval_fit_ebm": trainval_fit,
        "split_percentages": split_pct,
        "model_labels": MODELS,
        "n_test": int(len(test)),
        "n_train": int((feats.split == "train").sum()),
        "n_val": int((feats.split == "val").sum()),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_actual_vs_pred(test)
    plot_validation(preds)
    plot_residuals(test)
    plot_model_comparison(overall, test)
    plot_confusion(alerts["ebm"])
    plot_scatter(test)

    table = pd.DataFrame(overall).T.rename(index=MODELS)
    print(table.round(3).to_string())
    print("\nAlert (EBM):", {k: round(v, 3) for k, v in alerts["ebm"].items() if isinstance(v, float)})
    print("Alert (persistence):", {k: round(v, 3) for k, v in alerts["persistence"].items() if isinstance(v, float)})
    print("Direction:", direction)


if __name__ == "__main__":
    main()
