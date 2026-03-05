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


def plot_time_series(daily_df, column=TARGET_VARIABLE, rolling_window=30):
    """Plot daily temperature with rolling mean and std band."""
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    dates = pd.to_datetime(daily_df["date"])
    values = daily_df[column]
    roll = values.rolling(rolling_window, min_periods=1)

    ax.plot(dates, values, color="steelblue", alpha=0.5, lw=1, label="Daily")
    roll_mean = roll.mean()
    roll_std = roll.std().fillna(0)
    ax.plot(dates, roll_mean, color="darkorange", lw=2, label=f"{rolling_window}-day MA")
    ax.fill_between(
        dates, roll_mean - roll_std, roll_mean + roll_std, alpha=0.2, color="darkorange"
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.set_title(f"Global Mean {column} Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(column)
    ax.legend()
    return _save(fig, f"timeseries_{column}.png")


def plot_seasonal_decomposition(daily_df, column=TARGET_VARIABLE):
    """STL decomposition of the time series."""
    import matplotlib.dates as mdates

    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        logger.warning("statsmodels not available. Skipping decomposition plot.")
        return

    series = daily_df.set_index("date")[column].dropna()
    series.index = pd.to_datetime(series.index)
    series = series.asfreq("D").interpolate()

    if len(series) < 14:
        return

    period = min(7, len(series) // 3)
    stl = STL(series, period=period, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(series.index, series, color="steelblue", lw=1)
    axes[0].set_ylabel("Observed")
    axes[1].plot(series.index, res.trend, color="darkorange", lw=1.5)
    axes[1].set_ylabel("Trend")
    axes[2].plot(series.index, res.seasonal, color="green", lw=1)
    axes[2].set_ylabel("Seasonal")
    axes[3].plot(series.index, res.resid, color="red", lw=0.8, alpha=0.7)
    axes[3].axhline(0, color="black", lw=0.5)
    axes[3].set_ylabel("Residual")
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    fig.suptitle(f"STL Decomposition - {column}", fontsize=13, fontweight="bold")
    return _save(fig, f"stl_decomposition_{column}.png")


def plot_seasonal_patterns(daily_df, column=TARGET_VARIABLE):
    """Box plot of temperature by month showing seasonal cycles."""
    df = daily_df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(12, 6))
    data_by_month = [df[df["month"] == m][column].dropna().values for m in range(1, 13)]
    bp = ax.boxplot(data_by_month, patch_artist=True, notch=False, vert=True)

    colors = plt.cm.coolwarm(np.linspace(0, 1, 12))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_title("Seasonal Temperature Distribution by Month", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature (°C)")
    return _save(fig, "seasonal_patterns.png")
