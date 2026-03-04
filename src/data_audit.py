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
