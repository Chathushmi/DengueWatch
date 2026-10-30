"""Explainability: EBM global/local, SHAP, LIME, PDP/ICE and a domain-knowledge check.

All explanations are on the model's output scale: predicted log growth of cases
from this week to next week. A contribution of +0.1 means about +10.5% more cases
(exp(0.1) - 1), -0.1 means about -9.5%.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.features import (
    ALL_FEATURES,
    CATEGORICAL,
    FEATURE_LABELS,
    RAIN_LAGS,
    load_features,
    split,
)
from src.paths import ARTIFACTS, FIGURES

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})
SEED = 42


def label(term: str) -> str:
    if " & " in term:
        a, b = term.split(" & ")
        return f"{FEATURE_LABELS.get(a, a)} x {FEATURE_LABELS.get(b, b)}"
    return FEATURE_LABELS.get(term, term)


# ---------------------------------------------------------------- EBM global
def ebm_global(ebm) -> pd.DataFrame:
    imp = pd.DataFrame({"term": ebm.term_names_, "importance": ebm.term_importances()})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp["label"] = imp["term"].map(label)

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["tab:purple" if " & " in t else "tab:red" for t in top.term]
    ax.barh(top.label, top.importance, color=colors)
    ax.set_xlabel("Mean |contribution| to log growth")
    ax.set_title("EBM global term importance (purple = pairwise interaction)")
    fig.tight_layout()
    fig.savefig(FIGURES / "ebm_global_importance.png")
    plt.close(fig)
    return imp


def shape_data(ebm, term: str) -> dict:
    idx = list(ebm.term_names_).index(term)
    return ebm.explain_global().data(idx)


def plot_shape(ax, ebm, term: str) -> None:
    d = shape_data(ebm, term)
    scores = np.asarray(d["scores"])
    names = d["names"]
    lower, upper = d.get("lower_bounds"), d.get("upper_bounds")
    if term in CATEGORICAL:
        ax.bar(range(len(scores)), scores, color="tab:red", alpha=0.8)
        ax.set_xticks(range(len(scores)), names, rotation=25, fontsize=7, ha="right")
    else:
        edges = np.asarray(names, dtype=float)
        ax.step(edges[:-1], scores, where="post", color="tab:red")
        if lower is not None and upper is not None:
            ax.fill_between(edges[:-1], lower, upper, step="post", color="tab:red", alpha=0.15)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(label(term), fontsize=9)
    ax.set_ylabel("Contribution", fontsize=8)


def ebm_shapes(ebm, imp: pd.DataFrame) -> list[str]:
    mains = [t for t in imp.term if " & " not in t][:9]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, term in zip(axes.ravel(), mains):
        plot_shape(ax, ebm, term)
    fig.suptitle("EBM shape functions - how each feature changes predicted log growth", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "ebm_shape_functions.png")
    plt.close(fig)

    inters = [t for t in imp.term if " & " in t]
    if inters:
        term = inters[0]
        d = shape_data(ebm, term)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        scores = np.asarray(d["scores"])
        im = ax.imshow(scores.T, origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-np.abs(scores).max(), vmax=np.abs(scores).max())
        a, b = term.split(" & ")
        ax.set_xlabel(label(a))
        ax.set_ylabel(label(b))
        ax.set_xticks(range(len(d["left_names"]) - 1)[::max(1, len(d["left_names"]) // 6)])
        ax.set_xticklabels([f"{float(v):.0f}" if a not in CATEGORICAL else v
                            for v in d["left_names"][:-1]][::max(1, len(d["left_names"]) // 6)], fontsize=7)
        ax.set_yticks(range(len(d["right_names"]) - 1)[::max(1, len(d["right_names"]) // 6)])
        ax.set_yticklabels([f"{float(v):.1f}" if b not in CATEGORICAL else v
                            for v in d["right_names"][:-1]][::max(1, len(d["right_names"]) // 6)], fontsize=7)
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="Contribution")
        ax.set_title(f"Top interaction: {label(term)}", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGURES / "ebm_top_interaction.png")
        plt.close(fig)
    return mains


# ---------------------------------------------------------------- EBM local
def local_contributions(ebm, row: pd.DataFrame) -> pd.DataFrame:
    d = ebm.explain_local(row[ALL_FEATURES]).data(0)
    out = pd.DataFrame({"term": d["names"], "contribution": d["scores"]})
    out["label"] = out["term"].map(label)
    out["value"] = [str(v) for v in d["values"]]
    out = out.reindex(out.contribution.abs().sort_values(ascending=False).index)
    return out, float(d["extra"]["scores"][0])


def plot_local(ebm, row: pd.DataFrame, title: str, path) -> dict:
    contrib, intercept = local_contributions(ebm, row)
    top = contrib.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(top.label, top.contribution, color=np.where(top.contribution > 0, "tab:red", "tab:blue"))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Contribution to log growth (red = more cases, blue = fewer)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    pred = intercept + contrib.contribution.sum()
    return {
        "intercept": intercept,
        "predicted_log_growth": float(pred),
        "top": contrib.head(6)[["label", "value", "contribution"]].to_dict("records"),
    }


# ---------------------------------------------------------------- numeric wrapper for SHAP / LIME
class NumericWrapper:
    """Encodes the two categorical columns as integers for SHAP and LIME."""

    def __init__(self, ebm, reference: pd.DataFrame):
        self.ebm = ebm
        self.cats = {c: sorted(reference[c].unique()) for c in CATEGORICAL}

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        df = df[ALL_FEATURES].copy()
        for c, values in self.cats.items():
            df[c] = df[c].map({v: i for i, v in enumerate(values)})
        return df.to_numpy(dtype=float)

    def decode(self, arr: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(arr, columns=ALL_FEATURES)
        for c, values in self.cats.items():
            idx = np.clip(np.round(df[c]).astype(int), 0, len(values) - 1)
            df[c] = [values[i] for i in idx]
        return df

    def predict(self, arr: np.ndarray) -> np.ndarray:
        return self.ebm.predict(self.decode(np.atleast_2d(arr)))


def run_shap(wrapper: NumericWrapper, background: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    bg = shap.sample(wrapper.encode(background), 100, random_state=SEED)
    explainer = shap.Explainer(wrapper.predict, shap.maskers.Independent(bg), algorithm="permutation",
                               feature_names=[label(f) for f in ALL_FEATURES], seed=SEED)
    sv = explainer(wrapper.encode(sample), max_evals=2 * len(ALL_FEATURES) + 1)

    plt.figure()
    shap.plots.bar(sv, max_display=15, show=False)
    plt.title("SHAP mean |value| (test sample)")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_bar.png", dpi=130, bbox_inches="tight")
    plt.close("all")

    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP beeswarm (test sample)")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_beeswarm.png", dpi=130, bbox_inches="tight")
    plt.close("all")

    ranking = pd.DataFrame({"term": ALL_FEATURES, "mean_abs_shap": np.abs(sv.values).mean(axis=0)})
    ranking["label"] = ranking.term.map(label)
    return ranking.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def run_lime(wrapper: NumericWrapper, train: pd.DataFrame, rows: dict[str, pd.DataFrame]) -> dict:
    cat_idx = [ALL_FEATURES.index(c) for c in CATEGORICAL]
    explainer = LimeTabularExplainer(
        wrapper.encode(train),
        feature_names=[label(f) for f in ALL_FEATURES],
        categorical_features=cat_idx,
        categorical_names={i: wrapper.cats[c] for i, c in zip(cat_idx, CATEGORICAL)},
        mode="regression",
        discretize_continuous=True,
        random_state=SEED,
    )
    out = {}
    for name, row in rows.items():
        exp = explainer.explain_instance(wrapper.encode(row)[0], wrapper.predict, num_features=10, num_samples=3000)
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(8, 4.8)
        plt.title(f"LIME - {name} week", fontsize=10)
        plt.tight_layout()
        fig.savefig(FIGURES / f"lime_{name}.png", dpi=130)
        plt.close(fig)
        out[name] = {"local_r2": float(exp.score), "weights": [[r, float(w)] for r, w in exp.as_list()]}
    return out


# ---------------------------------------------------------------- PDP / ICE
PDP_FEATURES = ["growth_1w", "cases_lag1", "rain_sum_8w", "rain_lag8", "temp_lag4", "humidity_lag4"]


def run_pdp(ebm, data: pd.DataFrame) -> dict:
    sample = data.sample(min(300, len(data)), random_state=SEED)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    out = {}
    for ax, feat in zip(axes.ravel(), PDP_FEATURES):
        grid = np.quantile(data[feat], np.linspace(0.02, 0.98, 25))
        ice = np.empty((len(sample), len(grid)))
        for j, v in enumerate(grid):
            tmp = sample[ALL_FEATURES].copy()
            tmp[feat] = v
            ice[:, j] = ebm.predict(tmp)
        pd_curve = ice.mean(axis=0)
        pct = (np.exp(pd_curve) - 1) * 100
        for line in ice[::10]:
            ax.plot(grid, (np.exp(line) - 1) * 100, color="gray", alpha=0.15, lw=0.7)
        ax.plot(grid, pct, color="tab:red", lw=2.2, label="PDP (average)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(label(feat), fontsize=9)
        ax.set_ylabel("Predicted % change in cases", fontsize=8)
        out[feat] = {"grid": grid.round(3).tolist(), "pct_change": pct.round(2).tolist()}
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Partial dependence (red) and ICE curves (grey)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "pdp_ice.png")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- domain check
def domain_check(ebm, feats: pd.DataFrame, imp: pd.DataFrame) -> dict:
    """Which rainfall lag matters most? Literature: cases follow rain by ~6-10 weeks."""
    imp_map = dict(zip(imp.term, imp.importance))
    lag_importance = [imp_map.get(f"rain_lag{k}", 0.0) for k in RAIN_LAGS]

    # data-side evidence: correlation of rainfall k weeks before with log cases, de-seasonalised
    corr = []
    for k in RAIN_LAGS:
        vals = []
        for _, g in feats.groupby("district"):
            y = g["target_log"] - g.groupby("week_of_year")["target_log"].transform("mean")
            x = g[f"rain_lag{k}"] - g.groupby("week_of_year")[f"rain_lag{k}"].transform("mean")
            vals.append(np.corrcoef(x, y)[0, 1])
        corr.append(float(np.mean(vals)))

    # raw (seasonality included) correlation of rainfall with cases, for comparison
    raw_corr = [
        float(np.mean([np.corrcoef(g[f"rain_lag{k}"], g["target_log"])[0, 1] for _, g in feats.groupby("district")]))
        for k in RAIN_LAGS
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(RAIN_LAGS, lag_importance, color="tab:red")
    axes[0].set_xlabel("Rainfall lag (weeks before forecast week)")
    axes[0].set_ylabel("EBM importance")
    axes[0].set_title("Model: importance of each rainfall lag")
    axes[1].plot(RAIN_LAGS, raw_corr, marker="o", label="Raw correlation")
    axes[1].plot(RAIN_LAGS, corr, marker="s", label="De-seasonalised")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axvspan(6, 10, color="orange", alpha=0.15, label="Literature lag 6-10 wks")
    axes[1].set_xlabel("Rainfall lag (weeks)")
    axes[1].set_ylabel("Correlation with log cases")
    axes[1].set_title("Data: rainfall vs later dengue cases")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "domain_rain_lag.png")
    plt.close(fig)

    seasonal_profile = feats.groupby("week_of_year")["target_cases"].mean()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(seasonal_profile.index, seasonal_profile.values, color="tab:red")
    for lo, hi, name in [(18, 39, "SW monsoon"), (40, 48, "2nd inter-monsoon")]:
        ax.axvspan(lo, hi, alpha=0.1, color="tab:blue")
        ax.text((lo + hi) / 2, seasonal_profile.max() * 0.95, name, ha="center", fontsize=8)
    ax.set_xlabel("Week of year")
    ax.set_ylabel("Mean weekly cases")
    ax.set_title("Average dengue cases by week of year (all districts, 2007-2025)")
    fig.tight_layout()
    fig.savefig(FIGURES / "seasonal_profile.png")
    plt.close(fig)

    return {
        "rain_lag_importance": dict(zip(RAIN_LAGS, lag_importance)),
        "most_important_rain_lag": int(RAIN_LAGS[int(np.argmax(lag_importance))]),
        "deseasonalised_corr": dict(zip(RAIN_LAGS, corr)),
        "raw_corr": dict(zip(RAIN_LAGS, raw_corr)),
        "peak_raw_corr_lag": int(RAIN_LAGS[int(np.argmax(raw_corr))]),
        "peak_deseasonalised_corr_lag": int(RAIN_LAGS[int(np.argmax(corr))]),
        "peak_week_of_year": int(seasonal_profile.idxmax()),
    }


def plot_eda(feats: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    for d, g in feats.groupby("district"):
        ax.plot(g.week_start_date, g.target_cases, lw=0.9, label=d)
    for name, key, color in [("train", "train", "tab:green"), ("validation", "val", "tab:orange"),
                             ("test", "test", "tab:purple")]:
        part = feats[feats.split == key]
        a, b = part.week_start_date.min(), part.week_start_date.max()
        ax.axvspan(a, b, color=color, alpha=0.06)
        ax.text(a + (b - a) / 2, ax.get_ylim()[1] * 0.92, name, ha="center", fontsize=9, color=color)
    ax.set_ylabel("Weekly cases")
    ax.set_title("Weekly dengue cases 2007-2025 and the chronological split")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_cases_timeline.png")
    plt.close(fig)


def main() -> None:
    ebm = joblib.load(ARTIFACTS / "ebm_model.pkl")
    feats = load_features()
    train, val, test = split(feats)
    trainval = pd.concat([train, val])

    plot_eda(feats)
    imp = ebm_global(ebm)
    mains = ebm_shapes(ebm, imp)
    imp.to_json(ARTIFACTS / "ebm_importance.json", orient="records", indent=2)

    test = test.copy()
    test["pred_delta"] = ebm.predict(test[ALL_FEATURES])
    # skip weeks right after a reporting gap (near-zero counts) so examples are typical
    typical = test[(test[["cases_lag1", "cases_lag2", "cases_lag4", "target_cases"]] >= 20).all(axis=1)]
    rising = typical.loc[[typical["pred_delta"].idxmax()]]
    falling = typical.loc[[typical["pred_delta"].idxmin()]]
    examples = {}
    for name, row in (("rising", rising), ("falling", falling)):
        title = (f"EBM local explanation - {row.district.iloc[0]}, week of "
                 f"{row.week_start_date.iloc[0].date()} ({name})")
        info = plot_local(ebm, row, title, FIGURES / f"ebm_local_{name}.png")
        info.update({
            "district": row.district.iloc[0],
            "week_start_date": str(row.week_start_date.iloc[0].date()),
            "cases_lag1": int(row.cases_lag1.iloc[0]),
            "actual_cases": int(row.target_cases.iloc[0]),
            "predicted_cases": float(np.expm1(row.base_log.iloc[0] + row.pred_delta.iloc[0])),
        })
        examples[name] = info

    wrapper = NumericWrapper(ebm, feats)
    print("Running SHAP ...")
    shap_rank = run_shap(wrapper, trainval, test.sample(200, random_state=SEED))
    shap_rank.to_json(ARTIFACTS / "shap_ranking.json", orient="records", indent=2)

    print("Running LIME ...")
    lime_out = run_lime(wrapper, trainval, {"rising": rising, "falling": falling})

    print("Running PDP / ICE ...")
    pdp = run_pdp(ebm, trainval)
    dom = domain_check(ebm, feats, imp)

    main_imp = imp[~imp.term.str.contains(" & ")].reset_index(drop=True)
    agreement = {
        "ebm_top10": main_imp.term.head(10).tolist(),
        "shap_top10": shap_rank.term.head(10).tolist(),
        "overlap_top10": len(set(main_imp.term.head(10)) & set(shap_rank.term.head(10))),
        "spearman_rank_corr": float(
            pd.Series(main_imp.set_index("term").importance).rank(ascending=False)
            .corr(shap_rank.set_index("term").mean_abs_shap.rank(ascending=False), method="spearman")
        ),
    }

    summary = {
        "examples": examples,
        "lime": lime_out,
        "pdp": pdp,
        "domain_check": dom,
        "agreement": agreement,
        "shape_terms": mains,
    }
    (ARTIFACTS / "explanations.json").write_text(json.dumps(summary, indent=2, default=str))
    print("Top EBM terms:", imp.term.head(8).tolist())
    print("Top SHAP terms:", shap_rank.term.head(8).tolist())
    print("Domain check:", {k: v for k, v in dom.items() if not isinstance(v, dict)})
    print("Agreement:", agreement["overlap_top10"], round(agreement["spearman_rank_corr"], 3))


if __name__ == "__main__":
    main()
