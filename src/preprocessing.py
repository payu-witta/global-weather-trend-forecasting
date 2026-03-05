"""
Data cleaning and preprocessing pipeline.

Steps performed:
  1. Parse and sort by timestamp
  2. Remove duplicate rows
  3. Normalize country names
  4. Handle missing values (numeric -> median, categorical -> mode)
  5. Clip physical outliers to valid ranges
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CATEGORICAL_FEATURES,
    TIME_COLUMN,
)

# Physical bounds for sanity-check clipping
PHYSICAL_BOUNDS = {
    "temperature_celsius": (-90, 60),
    "temperature_fahrenheit": (-130, 140),
    "feels_like_celsius": (-90, 70),
    "humidity": (0, 100),
    "wind_kph": (0, 400),
    "wind_mph": (0, 250),
    "gust_kph": (0, 500),
    "gust_mph": (0, 320),
    "pressure_mb": (870, 1085),
    "precip_mm": (0, 2000),
    "cloud": (0, 100),
    "visibility_km": (0, 100),
    "uv_index": (0, 20),
    "air_quality_PM2.5": (0, 1000),
    "air_quality_PM10": (0, 2000),
    "air_quality_Carbon_Monoxide": (0, 50000),
    "air_quality_Ozone": (0, 1000),
    "air_quality_Nitrogen_dioxide": (0, 500),
    "air_quality_Sulphur_dioxide": (0, 1000),
}

# Corrects known locale-specific and misspelled country names found in this dataset.
COUNTRY_NAME_MAP = {
    "Marrocos": "Morocco",
    "Inde": "India",
    "Russie": "Russia",
    "Allemagne": "Germany",
    "Espagne": "Spain",
    "Italie": "Italy",
    "Turquie": "Turkey",
    "Grece": "Greece",
    "Suede": "Sweden",
    "Norvege": "Norway",
    "Danemark": "Denmark",
    "Finlande": "Finland",
    "Pologne": "Poland",
    "Autriche": "Austria",
    "Suisse": "Switzerland",
    "Belgique": "Belgium",
    "Pays-Bas": "Netherlands",
    "Royaume-Uni": "United Kingdom",
    "Etats-Unis": "United States",
    "Mexique": "Mexico",
    "Bresil": "Brazil",
    "Argentine": "Argentina",
    "Chili": "Chile",
    "Colombie": "Colombia",
    "Perou": "Peru",
    "Egypte": "Egypt",
    "Algerie": "Algeria",
    "Tunisie": "Tunisia",
    "Libye": "Libya",
    "Ethiopie": "Ethiopia",
    "Chine": "China",
    "Japon": "Japan",
    "Coree du Sud": "South Korea",
    "Thaïlande": "Thailand",
    "Indonesie": "Indonesia",
    "Malaisie": "Malaysia",
}

logger = logging.getLogger(__name__)


def parse_timestamps(df):
    """Parse TIME_COLUMN to datetime and sort chronologically."""
    df = df.copy()
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[TIME_COLUMN])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with unparseable timestamps", dropped)
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    df["date"] = df[TIME_COLUMN].dt.date
    logger.info(
        "Timestamp range: %s to %s", df[TIME_COLUMN].min(), df[TIME_COLUMN].max()
    )
    return df


def remove_duplicates(df):
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        logger.info("Removed %d duplicate rows", dropped)
    else:
        logger.info("No duplicate rows found")
    return df.reset_index(drop=True)


def normalize_country_names(df, country_col="country"):
    """Apply COUNTRY_NAME_MAP to fix known misspellings and locale-specific names."""
    if country_col not in df.columns:
        return df

    df = df.copy()
    df[country_col] = df[country_col].str.strip()

    replaced_total = 0
    for wrong, correct in COUNTRY_NAME_MAP.items():
        mask = df[country_col] == wrong
        n = int(mask.sum())
        if n > 0:
            df.loc[mask, country_col] = correct
            replaced_total += n
            logger.info("Country name fix: '%s' -> '%s' (%d rows)", wrong, correct, n)

    if replaced_total:
        logger.info("Total country name corrections: %d rows", replaced_total)
    else:
        logger.info("Country names: no corrections needed.")
    return df


def impute_missing(df):
    """Fill missing values: numeric with column median, categorical with mode."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug(
                "Imputed %d missing in '%s' with median=%.3f", n_missing, col, median_val
            )

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            logger.debug(
                "Imputed %d missing in '%s' with mode='%s'", n_missing, col, fill_val
            )

    remaining = df.isnull().sum().sum()
    logger.info("Missing values after imputation: %d", remaining)
    return df


def clip_physical_bounds(df):
    """Clip numeric columns to physically valid ranges."""
    df = df.copy()
    clipped_total = 0
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
            if out_of_range > 0:
                df[col] = df[col].clip(lo, hi)
                clipped_total += out_of_range
                logger.debug(
                    "Clipped %d values in '%s' to [%s, %s]", out_of_range, col, lo, hi
                )
    logger.info("Total values clipped to physical bounds: %d", clipped_total)
    return df
