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

# Figures
FIGURE_DPI = 150
FIGURE_SIZE_DEFAULT = (12, 6)
FIGURE_SIZE_WIDE = (16, 8)
FIGURE_FORMAT = "png"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
