"""
Formal data quality audit for the Global Weather Repository dataset.

Produces:
  - Missing-value summary (pre-imputation)
  - Country name quality check (duplicates, known misspellings)

All outputs saved to outputs/reports/.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import REPORTS_DIR, TIME_COLUMN

logger = logging.getLogger(__name__)

# Known misspellings found in this dataset (extend after running audit)
KNOWN_MISSPELLINGS = {
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

SPARSE_OBS_THRESHOLD = 30  # countries with fewer records are flagged


def audit_missing_values(df):
    """Per-column missing value counts and percentages."""
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)

    result = pd.DataFrame({
        "column": missing.index,
        "n_missing": missing.values,
        "pct_missing": pct.values,
    }).sort_values("n_missing", ascending=False).reset_index(drop=True)

    complete = int((result["n_missing"] == 0).sum())
    logger.info(
        "Missing value audit: %d/%d columns fully populated, %d have gaps.",
        complete,
        len(result),
        len(result) - complete,
    )
    return result


def audit_country_names(df, country_col="country"):
    """Detect suspected misspellings and flag low-count country names."""
    if country_col not in df.columns:
        return pd.DataFrame()

    counts = df[country_col].value_counts().reset_index()
    counts.columns = ["country", "n_observations"]

    suspicious = []
    for _, row in counts.iterrows():
        name = row["country"]
        issues = []
        if name in KNOWN_MISSPELLINGS:
            issues.append(f"known misspelling -> {KNOWN_MISSPELLINGS[name]}")
        # Non-ASCII characters can indicate a locale-specific name
        if any(ord(c) > 127 for c in name):
            issues.append("non-ASCII characters")
        if issues:
            suspicious.append({
                "country": name,
                "issues": "; ".join(issues),
            })

    result_df = pd.DataFrame(suspicious) if suspicious else pd.DataFrame(
        columns=["country", "issues"]
    )
    logger.info(
        "Country name audit: %d unique countries, %d flagged as suspicious.",
        counts["country"].nunique(),
        len(result_df),
    )
    return result_df


def audit_obs_counts(df, country_col="country", date_col="date"):
    """Per-country row count and date span; flags sparse countries."""
    if country_col not in df.columns:
        return pd.DataFrame()

    date_series = pd.to_datetime(df.get(date_col, df.get(TIME_COLUMN)), errors="coerce")
    df = df.copy()
    df["_date"] = date_series

    agg = (
        df.groupby(country_col)
        .agg(
            n_observations=("_date", "count"),
            first_date=("_date", "min"),
            last_date=("_date", "max"),
        )
        .reset_index()
    )
    agg["n_days_span"] = (agg["last_date"] - agg["first_date"]).dt.days + 1
    agg["is_sparse"] = agg["n_observations"] < SPARSE_OBS_THRESHOLD
    agg = agg.sort_values("n_observations", ascending=False).reset_index(drop=True)

    n_sparse = int(agg["is_sparse"].sum())
    logger.info(
        "Obs count audit: %d countries total, %d flagged as sparse (<%d obs).",
        len(agg),
        n_sparse,
        SPARSE_OBS_THRESHOLD,
    )
    return agg


def audit_temporal_coverage(df, date_col=None):
    """Detect gaps in the daily time series."""
    col = date_col or TIME_COLUMN
    if col not in df.columns:
        return pd.DataFrame()

    dates = pd.to_datetime(df[col], errors="coerce").dropna().dt.date
    unique_dates = pd.Series(sorted(dates.unique()))

    expected = pd.date_range(unique_dates.iloc[0], unique_dates.iloc[-1], freq="D").date
    missing_dates = sorted(set(expected) - set(unique_dates))

    gaps_df = pd.DataFrame({"missing_date": missing_dates})

    logger.info(
        "Temporal audit: %s to %s, %d unique dates observed, %d dates missing.",
        unique_dates.iloc[0],
        unique_dates.iloc[-1],
        len(unique_dates),
        len(missing_dates),
    )
    return gaps_df


def audit_geographic_coverage(df):
    """Bounding box, unique location/country counts."""
    summary = {
        "n_unique_countries": int(df["country"].nunique()) if "country" in df.columns else None,
        "n_unique_locations": int(df["location_name"].nunique()) if "location_name" in df.columns else None,
    }
    if "latitude" in df.columns and "longitude" in df.columns:
        summary["lat_min"] = round(float(df["latitude"].min()), 3)
        summary["lat_max"] = round(float(df["latitude"].max()), 3)
        summary["lon_min"] = round(float(df["longitude"].min()), 3)
        summary["lon_max"] = round(float(df["longitude"].max()), 3)

    logger.info(
        "Geographic audit: %s countries, %s locations.",
        summary.get("n_unique_countries"),
        summary.get("n_unique_locations"),
    )
    return pd.DataFrame([summary])


def run_data_audit(df, save=True):
    """
    Run all audit checks and optionally save results to outputs/reports/.

    Returns
    -------
    dict with keys: missing, country_names, obs_counts, temporal, geographic
    """
    logger.info("=== Data audit started ===")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df_work = df.copy()
    if "date" not in df_work.columns and TIME_COLUMN in df_work.columns:
        df_work["date"] = pd.to_datetime(df_work[TIME_COLUMN], errors="coerce").dt.date

    missing_df = audit_missing_values(df_work)
    country_df = audit_country_names(df_work)
    obs_df = audit_obs_counts(df_work)
    temporal_df = audit_temporal_coverage(df_work)
    geo_df = audit_geographic_coverage(df_work)

    if save:
        missing_df.to_csv(REPORTS_DIR / "data_audit_missing.csv", index=False)
        country_df.to_csv(REPORTS_DIR / "data_audit_country_names.csv", index=False)
        obs_df.to_csv(REPORTS_DIR / "data_audit_obs_counts.csv", index=False)
        temporal_df.to_csv(REPORTS_DIR / "data_audit_temporal_coverage.csv", index=False)
        geo_df.to_csv(REPORTS_DIR / "data_audit_geographic_coverage.csv", index=False)
        logger.info("Audit reports saved to %s", REPORTS_DIR)

    logger.info("=== Data audit complete ===")
    return {
        "missing": missing_df,
        "country_names": country_df,
        "obs_counts": obs_df,
        "temporal": temporal_df,
        "geographic": geo_df,
    }
