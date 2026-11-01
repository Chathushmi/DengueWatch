# Sri Lanka Dengue Forecasting with an Explainable Boosting Machine

MSc in AI - Applied Machine Learning assignment.

Forecasts **next week's dengue cases** in Colombo, Gampaha and Kandy from 19 years of
Sri Lankan surveillance and climate data, using an **Explainable Boosting Machine (EBM)**,
a glass-box generalised additive model that was not covered in the lectures. The model is
explained with EBM shape functions, SHAP, LIME and partial dependence plots, and served
through a Streamlit web app.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py   # open the front-end at http://localhost:8501
```

The PDF and screenshots are rendered with the locally installed Microsoft Edge through Playwright,
so Microsoft Word is not needed.

## Results (test period Mar 2022 - Dec 2025)

| Model | MAE (cases/week) | MSE | RMSE | Median error | R2 |
|---|---|---|---|---|---|
| EBM (full) | 32.6 | 2,445 | 49.4 | 21.5 | 0.76 (test) |
| Persistence baseline | 32.2 | 2,400 | 49.0 | 21.0 | 0.77 |
| Seasonal-average baseline | 70.8 | 9,673 | 98.3 | 49.3 | 0.06 |
| EBM (climate only) | 82.0 | 12,449 | 111.6 | 58.6 | -0.21 |

More metrics (WAPE, sMAPE, bias, MASE, Pearson r; alert specificity, balanced accuracy, MCC, PR-AUC) are in
`artifacts/metrics.json`, the report and the app's Model performance page.

Validation R² ≈ 0.89; train+validation fit R² ≈ 0.91. Test R² matches persistence because one-week-ahead counts
are highly autocorrelated. The EBM adds direction accuracy on large moves (~69%) and outbreak alerts (precision ~0.94,
ROC-AUC ~0.93). Rainfall 5–8 weeks earlier is the main climate signal.

## Layout

| Path | Purpose |
|------|---------|
| `data/` | Raw CSV and `DATA_SOURCE.md` (how the data was obtained) |
| `src/features.py` | Cleaning, lag / rolling / seasonal features, time-based split |
| `src/train.py` | EBM hyperparameter tuning, baselines, climate-only ablation |
| `src/evaluate.py` | Test metrics (MAE, RMSE, R2, alert F1/AUC) and plots |
| `src/explain.py` | EBM global/local, SHAP, LIME, PDP/ICE, domain-knowledge check |
| `app/streamlit_app.py` | Front-end: overview, forecast + what-if, explanations, performance |
| `artifacts/` | Generated models, metrics, figures |

## Method summary

- **Target:** reported dengue cases in the following epidemiological week. The EBM learns the log growth
  from the latest known week, `log1p(cases next week) - log1p(cases this week)`.
- **Features:** cases 1, 2 and 4 weeks before plus the 1- and 3-week case trends; rainfall for each of the last 12 weeks;
  4- and 8-week rainfall totals; humidity and temperature at 1/4/8/12-week lags;
  week of year; monsoon season; district.
- **Split (chronological, whole weeks):** 65% train (Mar 2007 - May 2019) / 15% validation (May 2019 - Mar 2022) /
  20% test (Mar 2022 - Dec 2025); exact dates in `artifacts/training_info.json`.
- **Outbreak alert:** predicted cases above the district's training 75th percentile.
