"""
Feature engineering for the Global Weather Forecasting project.

Creates:
  - Rolling statistics (mean, std) at multiple windows
  - Lag features
  - Temporal / calendar features
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import LAG_DAYS, RANDOM_SEED, ROLLING_WINDOWS, TARGET_VARIABLE

logger = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)


def add_calendar_features(df, date_col="date"):
    """Add year, month, day-of-week, week-of-year, quarter, and season."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col])

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day_of_week"] = dt.dt.dayofweek  # 0 = Monday
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter

    # Season (Northern Hemisphere convention)
    month = dt.dt.month
    df["season"] = pd.cut(
        month,
        bins=[0, 3, 6, 9, 12],
        labels=["Winter", "Spring", "Summer", "Autumn"],
        ordered=False,
    )
    df["season_code"] = pd.cut(
        month,
        bins=[0, 3, 6, 9, 12],
        labels=[0, 1, 2, 3],
        ordered=False,
    ).astype(int)

    # Cyclical encoding for month and day-of-year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    logger.info("Added calendar features (year, month, season, cyclical encodings)")
    return df


def add_rolling_features(df, target_col=TARGET_VARIABLE, windows=None):
    """
    Add rolling mean and rolling std for the target variable at multiple windows.
    Requires df sorted by date ascending.
    """
    if windows is None:
        windows = ROLLING_WINDOWS
    df = df.copy()
    for w in windows:
        df[f"{target_col}_rolling_mean_{w}d"] = (
            df[target_col].shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f"{target_col}_rolling_std_{w}d"] = (
            df[target_col].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
        )
    logger.info("Added rolling features for windows: %s", windows)
    return df


def add_rolling_features_multi(df, columns, windows=None):
    """Add rolling mean for multiple columns at multiple windows."""
    if windows is None:
        windows = ROLLING_WINDOWS
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_roll{w}"] = (
                df[col].shift(1).rolling(window=w, min_periods=1).mean()
            )
    return df


def add_lag_features(df, target_col=TARGET_VARIABLE, lags=None):
    """Add lagged values of the target variable."""
    if lags is None:
        lags = LAG_DAYS
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}d"] = df[target_col].shift(lag)
    logger.info("Added lag features for lags: %s", lags)
    return df


def add_derived_features(df):
    """
    Add physically meaningful derived variables:
      - temperature change rate (day-over-day delta)
      - approximate dew point (Magnus formula)
      - apparent heat index (Steadman formula, simplified)
      - wind components (u, v)
      - pressure tendency
    """
    df = df.copy()

    if "temperature_celsius" in df.columns:
        df["temp_change_rate"] = df["temperature_celsius"].diff().fillna(0)
        df["temp_change_rate_abs"] = df["temp_change_rate"].abs()

    if "temperature_celsius" in df.columns and "humidity" in df.columns:
        T = df["temperature_celsius"]
        RH = df["humidity"].clip(1, 100)
        a, b = 17.27, 237.7
        alpha = (a * T) / (b + T) + np.log(RH / 100.0)
        df["dew_point"] = (b * alpha) / (a - alpha)

    if "temperature_celsius" in df.columns and "humidity" in df.columns:
        T = df["temperature_celsius"]
        RH = df["humidity"]
        df["heat_index"] = (
            -8.78469475556
            + 1.61139411 * T
            + 2.33854883889 * RH
            - 0.14611605 * T * RH
            - 0.012308094 * T**2
            - 0.0164248277778 * RH**2
            + 0.002211732 * T**2 * RH
            + 0.00072546 * T * RH**2
            - 0.000003582 * T**2 * RH**2
        )

    if "wind_kph" in df.columns and "wind_degree" in df.columns:
        wind_rad = np.deg2rad(df["wind_degree"])
        df["wind_u"] = -df["wind_kph"] * np.sin(wind_rad)
        df["wind_v"] = -df["wind_kph"] * np.cos(wind_rad)

    if "pressure_mb" in df.columns:
        df["pressure_tendency"] = df["pressure_mb"].diff().fillna(0)

    if "precip_mm" in df.columns:
        df["is_rainy"] = (df["precip_mm"] > 0.1).astype(int)

    logger.info("Added derived meteorological features")
    return df


def add_monthly_stats(df, target_col=TARGET_VARIABLE):
    """Merge monthly mean and std back onto the daily DataFrame."""
    df = df.copy()
    if "month" not in df.columns:
        df = add_calendar_features(df)

    monthly = (
        df.groupby("month")[target_col]
        .agg(monthly_mean="mean", monthly_std="std")
        .reset_index()
    )
    df = df.merge(monthly, on="month", how="left")
    df["temp_deviation_from_monthly"] = df[target_col] - df["monthly_mean"]
    logger.info("Added monthly climatological statistics")
    return df


def add_yearly_anomaly(df, target_col=TARGET_VARIABLE):
    """Compute deviation from the overall global mean (climate anomaly signal)."""
    df = df.copy()
    global_mean = df[target_col].mean()
    df["climate_anomaly"] = df[target_col] - global_mean
    logger.info(
        "Added climate anomaly feature (deviation from global mean=%.2f)", global_mean
    )
    return df


def run_feature_engineering(daily_df, extra_numeric_cols=None):
    """
    Apply all feature engineering steps to the global daily DataFrame.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame ready for model training.
    """
    logger.info("=== Feature engineering started ===")

    df = daily_df.copy()
    df = add_calendar_features(df)
    df = add_rolling_features(df)
    df = add_lag_features(df)
    df = add_derived_features(df)
    df = add_monthly_stats(df)
    df = add_yearly_anomaly(df)

    if extra_numeric_cols:
        df = add_rolling_features_multi(df, extra_numeric_cols)

    max_lag = max(LAG_DAYS)
    df = df.iloc[max_lag:].reset_index(drop=True)

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    logger.info("=== Feature engineering done ===")
    return df


def get_feature_columns(df, exclude=None):
    """Return list of numeric columns suitable for ML models."""
    if exclude is None:
        exclude = ["date", "year"]
    numeric = df.select_dtypes(include=[float, int]).columns.tolist()
    return [c for c in numeric if c not in exclude]
