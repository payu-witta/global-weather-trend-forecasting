# Global Weather Trend Forecasting

**PM Accelerator | Data Science Project**

End-to-end pipeline for ingesting global weather data, engineering features, training ensemble forecasting models, and producing an interactive HTML report with spatial, climate, and environmental-impact analyses.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Models](#models)
- [Notebooks](#notebooks)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Data Source](#data-source)

---

## Overview

This project builds a reproducible ML pipeline that:

1. Fetches multi-city weather data from Kaggle (WeatherAPI historical records)
2. Cleans, deduplicates, and aggregates to a daily global time series
3. Engineers lag, rolling, and calendar features
4. Trains five individual models and three ensemble strategies
5. Evaluates via walk-forward backtesting and computes prediction intervals
6. Produces spatial maps, climate trend analysis, regional warming rates, and air-quality correlations
7. Generates a self-contained HTML report with all figures and summary tables

**Target variable:** `temp_c` (mean daily temperature in °C)

---

## Project Structure

```
global-weather-trend-forecasting/
├── data/
│   ├── raw/                  # Downloaded Kaggle CSVs
│   └── processed/            # Parquet files after cleaning
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_forecasting_models.ipynb
│   ├── 04_spatial_analysis.ipynb
│   └── 05_feature_importance.ipynb
├── outputs/
│   ├── figures/              # PNG plots saved by the pipeline
│   ├── models/               # Serialised model artefacts
│   └── reports/              # Generated HTML report
├── src/
│   ├── config.py             # Paths, constants, column lists
│   ├── data_loader.py        # Kaggle download + raw ingestion
│   ├── data_preprocessing.py # Cleaning, dedup, imputation
│   ├── feature_engineering.py# Lag/rolling/calendar features
│   ├── eda.py                # Exploratory plots and summary stats
│   ├── anomaly_detection.py  # Z-score and IQR outlier flagging
│   ├── forecasting_models.py # SARIMA, Prophet, XGBoost, LightGBM, LSTM
│   ├── ensemble_model.py     # Simple avg, weighted avg, stacking
│   ├── feature_importance.py # Tree, permutation, SHAP importances
│   ├── statistical_diagnostics.py # ADF, ACF/PACF, Ljung-Box, residuals
│   ├── backtesting.py        # Walk-forward expanding-window CV
│   ├── prediction_intervals.py # SARIMA CI, Prophet bands, XGB quantile
│   ├── spatial_analysis.py   # Folium maps, Plotly scatter geo, GeoPandas
│   └── regional_analysis.py  # Per-country warming rates (OLS °C/year)
├── generate_report.py        # Standalone HTML report builder
├── run_pipeline.py           # Orchestrates all 16 pipeline stages
└── requirements.txt
```

---

## Installation

Python 3.10+ is recommended.

```bash
# Clone the repository
git clone https://github.com/your-org/global-weather-trend-forecasting.git
cd global-weather-trend-forecasting

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Kaggle credentials

Place your `kaggle.json` API token at `~/.kaggle/kaggle.json` (Linux/macOS) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows), or export the environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

---

## Quick Start

### Run the full pipeline

```bash
python run_pipeline.py
```

This executes all 16 stages in order and writes outputs to `outputs/`.

**Skip slow backtesting** (useful for a fast end-to-end check):

```bash
python run_pipeline.py --skip-backtest
```

### Generate the HTML report

```bash
python generate_report.py
```

Opens `outputs/reports/weather_forecasting_report.html` — a single self-contained file with all figures embedded as base64 data URIs, no external dependencies required.

---

## Pipeline Stages

| # | Stage | Description |
|---|-------|-------------|
| 0 | `stage_load` | Download data from Kaggle and parse raw CSVs |
| 1 | `stage_preprocess` | Clean, deduplicate, impute, save `weather_clean.parquet` |
| 2 | `stage_feature_engineering` | Lag/rolling/calendar features → `daily_global.parquet` |
| 3 | `stage_eda` | Distribution plots, correlation heatmaps, time-series plots |
| 4 | `stage_anomaly` | Flag outliers with Z-score and IQR methods |
| 5 | `stage_forecasting` | Train SARIMA, Prophet, XGBoost, LightGBM, LSTM |
| 6 | `stage_ensemble` | Simple avg, inverse-RMSE weighted avg, Ridge stacking |
| 7 | `stage_feature_importance` | Tree importance, permutation importance, SHAP |
| 8 | `_stage_data_audit` | Row/column counts, missing-value report |
| 9 | `stage_spatial` | Geographic maps and regional comparisons |
| 10 | `stage_climate_analysis` | Global temperature trend (linear regression) |
| 11 | `stage_environmental_impact` | Air quality × weather correlation heatmap |
| 12 | `_stage_statistical_diagnostics` | ADF, ACF/PACF, Ljung-Box, residual diagnostics |
| 13 | `_stage_backtesting` | 5-fold expanding-window cross-validation |
| 14 | `_stage_prediction_intervals` | P10/P90 bands for SARIMA, Prophet, XGBoost |
| 15 | `_stage_regional_analysis` | Per-country OLS warming rate (°C/year) |

---

## Models

### Individual

| Model | Backend | Notes |
|-------|---------|-------|
| SARIMA | statsmodels | Order selected by AIC grid search |
| Prophet | Meta Prophet | Yearly + weekly seasonality, holidays off |
| XGBoost | xgboost | 300 trees, max_depth=6, early stopping |
| LightGBM | lightgbm | 300 iterations, num_leaves=63 |
| LSTM | PyTorch | 2-layer, hidden=64, sequence length=30 |

### Ensemble

| Strategy | Description |
|----------|-------------|
| Simple Average | Equal-weight mean of all individual predictions |
| Weighted Average | Weights proportional to inverse RMSE on validation set |
| Stacking | Ridge meta-learner trained on out-of-fold predictions |

---

## Notebooks

The `notebooks/` directory contains five Jupyter notebooks that mirror the pipeline stages and are designed for interactive exploration:

| Notebook | Content |
|----------|---------|
| `01_data_exploration.ipynb` | Raw data inspection, missing-value analysis, distributions |
| `02_feature_engineering.ipynb` | Feature construction, correlation analysis, feature selection |
| `03_forecasting_models.ipynb` | Model training, metrics comparison, forecast visualisation |
| `04_spatial_analysis.ipynb` | Interactive Plotly/Folium maps, regional bar charts |
| `05_feature_importance.ipynb` | Tree importance, permutation importance, SHAP beeswarm |

Launch JupyterLab:

```bash
jupyter lab
```

---

## Configuration

All configurable constants live in `src/config.py`:

```python
TARGET_VARIABLE   = "temp_c"       # column to forecast
TEST_SIZE         = 0.2            # train/test split ratio
RANDOM_SEED       = 42
LAG_WINDOWS       = [1, 3, 7, 14, 30]
ROLLING_WINDOWS   = [7, 14, 30]
FIGURE_DPI        = 150
```

Directory paths (`DATA_DIR`, `PROCESSED_DATA_DIR`, `FIGURES_DIR`, `REPORTS_DIR`, `MODELS_DIR`) are resolved relative to the project root using `pathlib.Path`, so the project is portable.

---

## Outputs

After a full pipeline run the following artefacts are produced:

```
outputs/
├── figures/
│   ├── eda_*.png
│   ├── anomaly_*.png
│   ├── forecast_*.png
│   ├── ensemble_*.png
│   ├── feature_importance_*.png
│   ├── shap_*.png
│   ├── spatial_*.png
│   ├── climate_trend.png
│   ├── diagnostics_*.png
│   ├── backtest_*.png
│   ├── intervals_*.png
│   └── regional_*.png
├── models/
│   ├── xgboost_model.json
│   ├── lightgbm_model.txt
│   └── lstm_model.pt
└── reports/
    ├── weather_forecasting_report.html   # self-contained report
    ├── metrics_summary.csv
    ├── ensemble_comparison.csv
    └── feature_importance.csv
```

---

## Data Source

Weather data is sourced from the **Global Weather Repository** on Kaggle:

> Kaggle dataset: `nelgiriyewithana/global-weather-repository`

The dataset contains hourly weather observations for locations worldwide, covering temperature, humidity, precipitation, wind speed, pressure, visibility, and air-quality indices (PM2.5, PM10, CO, NO2, O3).

Place `GlobalWeatherRepository.csv` in `data/raw/` before running if not using the Kaggle API downloader.

---

## License

This project is released for educational and portfolio purposes under the MIT License.
