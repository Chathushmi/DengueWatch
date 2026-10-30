# Dataset source

**Name:** Sri Lanka Weekly Dengue + Climate Dataset (2006-2025)
**File:** `srilanka_dengue_climate_weekly_2006_2025.csv`
**Kaggle:** https://www.kaggle.com/datasets/shinthurie/sri-lanka-weekly-dengue-climate-2006-2025

## How it was obtained

The file was downloaded with `scripts/download_data.py`, which calls
`kagglehub.dataset_download("shinthurie/sri-lanka-weekly-dengue-climate-2006-2025")`
and copies the CSV into this folder. No Kaggle login was needed for this public dataset.

The Kaggle dataset is itself a cleaned join of two public sources:

1. **Dengue cases** - Weekly Epidemiological Reports (WER), Epidemiology Unit,
   Ministry of Health, Sri Lanka (https://www.epid.gov.lk), accessed through the
   `denguedatahub` R package by Thiyanga S. Talagala.
2. **Climate** - NASA POWER daily point data (https://power.larc.nasa.gov/) at each
   district centroid, aggregated to epidemiological weeks (rainfall = weekly total,
   temperature and humidity = weekly mean).

## Contents

| Column | Meaning | Unit |
|---|---|---|
| year, week | Epidemiological year and week (1-53) | - |
| week_start_date, week_end_date | First and last day of the epi-week | date |
| district | Colombo, Gampaha or Kandy | - |
| cases | Reported dengue cases that week | count |
| temp_avg_c, temp_max_c, temp_min_c | Weekly mean of daily mean/max/min temperature | deg C |
| humidity_pct | Weekly mean relative humidity | % |
| rainfall_mm | Weekly total rainfall | mm |

2,973 rows (991 weeks x 3 districts), 23 Dec 2006 to 13 Dec 2025, no missing values.

## Ethics and licensing

- Only **aggregated district-level counts** are included. There are no names, IDs,
  addresses or any patient-level information, so no individual can be identified.
- Both upstream sources are published openly by government / NASA for public use.
- Please cite the Epidemiology Unit (Sri Lanka), NASA POWER, and the Kaggle dataset
  author when reusing the data.
