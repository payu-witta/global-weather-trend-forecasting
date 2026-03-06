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
