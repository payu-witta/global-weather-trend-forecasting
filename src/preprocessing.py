"""
Data cleaning and preprocessing pipeline.

Steps performed:
  1. Parse and sort by timestamp
  2. Remove duplicate rows
  3. Normalize country names
  4. Handle missing values (numeric -> median, categorical -> mode)
  5. Clip physical outliers to valid ranges
  6. IQR-based outlier replacement
  7. Encode categoricals
  8. Aggregate global daily means for time-series models
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ALL_NUMERIC,
    CATEGORICAL_FEATURES,
    LOCATION_COLUMN,
    NUMERICAL_FEATURES,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TARGET_VARIABLE,
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


def remove_iqr_outliers(df, columns, factor=3.0):
    """
    Replace extreme outliers (beyond factor×IQR) with column median.
    Uses a conservative factor=3.0 to keep genuine extremes.
    """
    df = df.copy()
    total_replaced = 0
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        n = mask.sum()
        if n > 0:
            df.loc[mask, col] = df[col].median()
            total_replaced += n
    logger.info(
        "IQR outlier replacement: %d values across %d columns", total_replaced, len(columns)
    )
    return df


def encode_categoricals(df):
    """Label-encode low-cardinality categorical columns."""
    df = df.copy()
    from sklearn.preprocessing import LabelEncoder

    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def aggregate_daily_global(df):
    """
    Aggregate all locations into a single global daily mean time series.

    This is the primary input for SARIMA and Prophet models.
    """
    numeric_cols = [c for c in ALL_NUMERIC if c in df.columns]
    daily = df.groupby("date")[numeric_cols].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    logger.info("Global daily aggregate: %d time steps", len(daily))
    return daily


def aggregate_daily_by_location(df):
    """Aggregate to daily means per location. Returns a dict {location_name: DataFrame}."""
    numeric_cols = [c for c in ALL_NUMERIC if c in df.columns]
    grouped = {}
    for loc, grp in df.groupby(LOCATION_COLUMN):
        daily = grp.groupby("date")[numeric_cols].mean().reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)
        grouped[loc] = daily
    logger.info("Per-location aggregation: %d locations", len(grouped))
    return grouped


def fit_scaler(df, columns):
    """Fit StandardScaler on given columns. Returns (scaled_df, scaler)."""
    scaler = StandardScaler()
    df = df.copy()
    existing = [c for c in columns if c in df.columns]
    df[existing] = scaler.fit_transform(df[existing])
    return df, scaler


def run_preprocessing(df, save=True):
    """
    Execute the complete preprocessing pipeline.

    Returns
    -------
    df_clean : pd.DataFrame
    daily_global : pd.DataFrame
    scaler : StandardScaler
    """
    logger.info("=== Preprocessing pipeline started ===")

    df = parse_timestamps(df)
    df = normalize_country_names(df)
    df = remove_duplicates(df)
    df = impute_missing(df)
    df = clip_physical_bounds(df)

    numeric_cols = [c for c in ALL_NUMERIC if c in df.columns]
    df = remove_iqr_outliers(df, numeric_cols)

    df, encoders = encode_categoricals(df)

    daily_global = aggregate_daily_global(df)

    scale_cols = [c for c in NUMERICAL_FEATURES if c in daily_global.columns]
    daily_global_scaled, scaler = fit_scaler(daily_global, scale_cols)

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_DATA_DIR / "weather_clean.parquet", index=False)
        daily_global.to_parquet(PROCESSED_DATA_DIR / "daily_global.parquet", index=False)
        logger.info("Saved processed files to %s", PROCESSED_DATA_DIR)

    logger.info("=== Preprocessing complete ===")
    return df, daily_global, scaler
