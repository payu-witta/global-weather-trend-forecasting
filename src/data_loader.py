"""
Data loading utilities for the Global Weather Forecasting project.

Supports:
  - Kaggle API download
  - Local CSV from data/raw/
  - Schema validation and basic dataset summary
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from src/ directory directly
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATASET_FILENAME,
    KAGGLE_DATASET,
    LAT_COLUMN,
    LON_COLUMN,
    NUMERICAL_FEATURES,
    RAW_DATA_DIR,
    TIME_COLUMN,
)

logger = logging.getLogger(__name__)


def download_from_kaggle(output_dir=None):
    """Download the dataset from Kaggle. Requires kaggle API credentials."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        return False

    output_dir = Path(output_dir) if output_dir else RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Authenticating with Kaggle API ...")
        kaggle.api.authenticate()
        logger.info("Downloading dataset %s ...", KAGGLE_DATASET)
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(output_dir),
            unzip=True,
        )
        logger.info("Download complete -> %s", output_dir)
        return True
    except Exception as exc:
        logger.error("Kaggle download failed: %s", exc)
        return False


def find_csv(data_dir=None):
    """Return path to the dataset CSV, or None if not found."""
    data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
    preferred = data_dir / DATASET_FILENAME
    if preferred.exists():
        return preferred
    csv_files = sorted(data_dir.glob("*.csv"))
    if csv_files:
        logger.info("Using found CSV: %s", csv_files[0])
        return csv_files[0]
    return None


def load_dataset(data_dir=None, try_kaggle=True):
    """
    Load the Global Weather Repository dataset.

    Checks for local file first; attempts Kaggle download if not present.

    Returns
    -------
    pd.DataFrame
        Raw dataset as loaded from CSV.
    """
    filepath = find_csv(data_dir)

    if filepath is None and try_kaggle:
        logger.info("Dataset not found locally. Attempting Kaggle download ...")
        success = download_from_kaggle(data_dir)
        if success:
            filepath = find_csv(data_dir)

    if filepath is None:
        raise FileNotFoundError(
            f"Dataset not found in {data_dir or RAW_DATA_DIR}. "
            "Please place GlobalWeatherRepository.csv in data/raw/ "
            "or configure Kaggle API credentials (~/.kaggle/kaggle.json)."
        )

    logger.info("Loading: %s", filepath)
    df = pd.read_csv(filepath, low_memory=False)

    # Normalize column names that contain colons (e.g. condition:text)
    df.columns = [c.replace(":", "_") for c in df.columns]

    logger.info("Loaded %d rows x %d columns", *df.shape)
    return df


REQUIRED_COLUMNS = [
    "country",
    "location_name",
    "latitude",
    "longitude",
    "last_updated",
    "temperature_celsius",
    "humidity",
    "wind_kph",
    "pressure_mb",
    "precip_mm",
]


def validate_schema(df):
    """
    Check that all required columns are present.

    Returns True when schema is valid, False otherwise.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("Missing expected columns: %s", missing)
        return False
    logger.info("Schema validation passed")
    return True


def get_dataset_summary(df):
    """Return a dict with key dataset statistics."""
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_per_column": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "countries": int(df["country"].nunique()) if "country" in df.columns else None,
        "locations": int(df["location_name"].nunique()) if "location_name" in df.columns else None,
    }

    if TIME_COLUMN in df.columns:
        dates = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
        summary["date_range"] = {
            "min": str(dates.min()),
            "max": str(dates.max()),
            "unique_dates": int(dates.dt.date.nunique()),
        }

    numeric_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    if numeric_cols:
        summary["numeric_summary"] = df[numeric_cols].describe().round(3).to_dict()

    return summary


def print_summary(df):
    """Pretty-print dataset summary to logger."""
    s = get_dataset_summary(df)
    logger.info("Dataset summary:")
    logger.info("  Shape: %s rows x %s columns", *s["shape"])
    logger.info("  Countries: %s", s.get("countries"))
    logger.info("  Locations: %s", s.get("locations"))
    if "date_range" in s:
        dr = s["date_range"]
        logger.info(
            "  Date range: %s to %s (%s unique dates)",
            dr["min"],
            dr["max"],
            dr["unique_dates"],
        )
    logger.info("  Duplicates: %s", s.get("duplicate_rows"))
    missing = s.get("missing_per_column", {})
    if missing:
        logger.info("  Columns with missing values: %d", len(missing))
    else:
        logger.info("  No missing values detected")
