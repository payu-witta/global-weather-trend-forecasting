"""
Data cleaning and preprocessing pipeline.

Steps performed (so far):
  1. Parse and sort by timestamp
  2. Remove duplicate rows
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TIME_COLUMN,
)

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
