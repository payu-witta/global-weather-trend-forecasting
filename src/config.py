"""
Configuration settings for the Global Weather Forecasting project.
All paths, hyperparameters, and constants are defined here.
"""

from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent

# Directory structure
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FORECASTS_DIR = OUTPUTS_DIR / "forecasts"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"

# Dataset
DATASET_FILENAME = "GlobalWeatherRepository.csv"
KAGGLE_DATASET = "nelgiriyewithana/global-weather-repository"

# Column definitions
TIME_COLUMN = "last_updated"
TARGET_VARIABLE = "temperature_celsius"
LAT_COLUMN = "latitude"
LON_COLUMN = "longitude"
COUNTRY_COLUMN = "country"
LOCATION_COLUMN = "location_name"

NUMERICAL_FEATURES = [
    "temperature_celsius",
    "humidity",
    "wind_kph",
    "pressure_mb",
    "precip_mm",
    "cloud",
    "feels_like_celsius",
    "visibility_km",
    "uv_index",
    "gust_kph",
]

AIR_QUALITY_COLUMNS = [
    "air_quality_Carbon_Monoxide",
    "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5",
    "air_quality_PM10",
]

CATEGORICAL_FEATURES = [
    "country",
    "wind_dir",
    "moon_phase",
]

# All numeric columns used throughout the pipeline
ALL_NUMERIC = NUMERICAL_FEATURES + AIR_QUALITY_COLUMNS

# Time-series / forecasting
FORECAST_HORIZON = 30  # days ahead
RANDOM_SEED = 42
TEST_SIZE = 0.2  # fraction of data used for evaluation

# Rolling / lag feature windows
ROLLING_WINDOWS = [7, 14, 30]
LAG_DAYS = [1, 7, 14, 30]

# SARIMA
ARIMA_ORDER = (1, 1, 1)
ARIMA_SEASONAL_ORDER = (1, 1, 1, 7)

# XGBoost
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# LightGBM
LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# LSTM
LSTM_PARAMS = {
    "sequence_length": 30,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
}

# Anomaly detection
CONTAMINATION = 0.05
ZSCORE_THRESHOLD = 3.0

# Figures
FIGURE_DPI = 150
FIGURE_SIZE_DEFAULT = (12, 6)
FIGURE_SIZE_WIDE = (16, 8)
FIGURE_FORMAT = "png"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
