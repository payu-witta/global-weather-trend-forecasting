"""
Anomaly detection for unusual weather events.

Implements three complementary techniques:
  1. Isolation Forest
  2. Local Outlier Factor (LOF)
  3. Z-score / statistical threshold

Anomalies are cross-validated by requiring detection by at least two methods.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CONTAMINATION,
    FIGURES_DIR,
    RANDOM_SEED,
    TARGET_VARIABLE,
    ZSCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)


def isolation_forest_anomalies(df, feature_cols, contamination=CONTAMINATION):
    """
    Detect anomalies using Isolation Forest.

    Returns a boolean Series (True = anomaly).
    """
    X = df[feature_cols].fillna(df[feature_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    preds = model.fit_predict(X_scaled)
    flags = pd.Series(preds == -1, index=df.index, name="if_anomaly")
    logger.info(
        "Isolation Forest: %d anomalies detected (%.1f%%)", flags.sum(), 100 * flags.mean()
    )
    return flags, model


def lof_anomalies(df, feature_cols, contamination=CONTAMINATION, n_neighbors=20):
    """
    Detect anomalies using Local Outlier Factor.

    Returns a boolean Series (True = anomaly).
    """
    X = df[feature_cols].fillna(df[feature_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=-1,
    )
    preds = model.fit_predict(X_scaled)
    flags = pd.Series(preds == -1, index=df.index, name="lof_anomaly")
    logger.info("LOF: %d anomalies detected (%.1f%%)", flags.sum(), 100 * flags.mean())
    return flags, model


def zscore_anomalies(df, target_col=TARGET_VARIABLE, threshold=ZSCORE_THRESHOLD):
    """
    Flag observations whose z-score exceeds the threshold.

    Returns a boolean Series (True = anomaly).
    """
    col = df[target_col]
    zscores = (col - col.mean()) / col.std()
    flags = zscores.abs() > threshold
    flags.name = "zscore_anomaly"
    df = df.copy()
    df["zscore"] = zscores
    logger.info(
        "Z-score (threshold=%.1f): %d anomalies detected (%.1f%%)",
        threshold,
        flags.sum(),
        100 * flags.mean(),
    )
    return flags, zscores


def combine_anomaly_flags(*flag_series, min_votes=2):
    """Return True where at least min_votes methods agree on an anomaly."""
    combined = sum(f.astype(int) for f in flag_series)
    ensemble = combined >= min_votes
    ensemble.name = "ensemble_anomaly"
    logger.info(
        "Ensemble anomaly (min_votes=%d): %d anomalies",
        min_votes,
        ensemble.sum(),
    )
    return ensemble


def run_anomaly_detection(df, target_col=TARGET_VARIABLE, feature_cols=None):
    """
    Run all three anomaly detection methods and combine results.

    Returns
    -------
    df_annotated : pd.DataFrame
    anomaly_summary : dict
    """
    logger.info("=== Anomaly detection started ===")

    df = df.copy()

    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[float, int]).columns.tolist()
        exclude = {"year", "month", "day_of_week", "day_of_year", "week_of_year", "quarter"}
        feature_cols = [c for c in feature_cols if c not in exclude]

    if_flags, if_model = isolation_forest_anomalies(df, feature_cols)
    lof_flags, lof_model = lof_anomalies(df, feature_cols)
    z_flags, zscores = zscore_anomalies(df, target_col)

    ensemble_flags = combine_anomaly_flags(if_flags, lof_flags, z_flags, min_votes=2)

    df["if_anomaly"] = if_flags.values
    df["lof_anomaly"] = lof_flags.values
    df["zscore_anomaly"] = z_flags.values
    df["zscore"] = zscores.values
    df["anomaly"] = ensemble_flags.values

    anomaly_rows = df[df["anomaly"]]
    summary = {
        "isolation_forest_count": int(if_flags.sum()),
        "lof_count": int(lof_flags.sum()),
        "zscore_count": int(z_flags.sum()),
        "ensemble_count": int(ensemble_flags.sum()),
        "anomaly_dates": anomaly_rows["date"].astype(str).tolist() if "date" in anomaly_rows.columns else [],
        "anomaly_temp_values": anomaly_rows[target_col].round(2).tolist() if target_col in anomaly_rows.columns else [],
    }

    logger.info(
        "=== Anomaly detection complete. Ensemble: %d anomalies ===",
        int(ensemble_flags.sum()),
    )
    return df, summary


def describe_anomalies(df_annotated, target_col=TARGET_VARIABLE):
    """Produce a textual description of detected anomalies."""
    if "anomaly" not in df_annotated.columns:
        raise ValueError("Run run_anomaly_detection first to add 'anomaly' column.")

    anom = df_annotated[df_annotated["anomaly"]].copy()
    cols = ["date", target_col, "zscore", "if_anomaly", "lof_anomaly"]
    available = [c for c in cols if c in anom.columns]

    anom = anom[available].sort_values("date").reset_index(drop=True)

    if target_col in anom.columns:
        global_mean = df_annotated[target_col].mean()
        anom["anomaly_type"] = np.where(
            anom[target_col] > global_mean,
            "Extreme High",
            "Extreme Low",
        )

    return anom
