"""
Centralized visualization utilities for the weather forecasting project.

All figures are saved to outputs/figures/ in PNG format.
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    AIR_QUALITY_COLUMNS,
    FIGURE_DPI,
    FIGURE_SIZE_DEFAULT,
    FIGURE_SIZE_WIDE,
    FIGURES_DIR,
    NUMERICAL_FEATURES,
    RANDOM_SEED,
    TARGET_VARIABLE,
)

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, filename, tight=True):
    path = FIGURES_DIR / filename
    if tight:
        plt.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure -> %s", path)
    return path


def plot_distributions(df, columns=None, max_cols=4):
    """Plot histograms with KDE for numeric columns."""
    if columns is None:
        columns = [c for c in NUMERICAL_FEATURES if c in df.columns]
    columns = columns[:16]

    n_rows = (len(columns) + max_cols - 1) // max_cols
    fig, axes = plt.subplots(n_rows, max_cols, figsize=(5 * max_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(columns):
        data = df[col].dropna()
        axes[i].hist(data, bins=50, color="steelblue", edgecolor="white", alpha=0.7, density=True)
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(data)
            xs = np.linspace(data.min(), data.max(), 200)
            axes[i].plot(xs, kde(xs), color="darkorange", lw=2)
        except Exception:
            pass
        axes[i].set_title(col, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Density")

    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    return _save(fig, "distributions.png")


def plot_correlation_heatmap(df, columns=None):
    """Seaborn heatmap of Pearson correlations."""
    if columns is None:
        columns = [c for c in NUMERICAL_FEATURES if c in df.columns]

    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.5,
        square=True,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    return _save(fig, "correlation_heatmap.png")
