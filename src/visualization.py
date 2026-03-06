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


def plot_global_warming_trend(daily_df, column=TARGET_VARIABLE):
    """Scatter + regression line showing long-term temperature trend."""
    import matplotlib.dates as mdates
    from scipy import stats

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    dates = pd.to_datetime(daily_df["date"])
    x_num = mdates.date2num(dates)
    y = daily_df[column].values

    ax.scatter(dates, y, alpha=0.3, s=10, color="steelblue", label="Daily Obs.")

    slope, intercept, r, p, se = stats.linregress(x_num, y)
    trend_y = slope * x_num + intercept
    ax.plot(dates, trend_y, color="red", lw=2, label=f"Trend (slope={slope * 365:.3f}°C/yr)")

    daily_df = daily_df.copy()
    daily_df["year"] = pd.to_datetime(daily_df["date"]).dt.year
    yearly = daily_df.groupby("year")[column].mean()
    ax.plot(
        pd.to_datetime(yearly.index.astype(str) + "-07-01"),
        yearly.values,
        color="darkorange",
        lw=2,
        marker="o",
        markersize=6,
        label="Yearly Mean",
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    ax.set_title("Global Temperature Trend", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    return _save(fig, "global_warming_trend.png")


def plot_anomalies(df_annotated, column=TARGET_VARIABLE):
    """Overlay anomaly points on the time series."""
    import matplotlib.dates as mdates

    if "anomaly" not in df_annotated.columns:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    dates = pd.to_datetime(df_annotated["date"])
    vals = df_annotated[column]

    ax.plot(dates, vals, color="steelblue", lw=1, alpha=0.7, label="Normal")
    mask = df_annotated["anomaly"]
    ax.scatter(dates[mask], vals[mask], color="red", zorder=5, s=50, label="Anomaly", marker="x")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.set_title(f"Anomaly Detection on {column}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(column)
    ax.legend()
    return _save(fig, "anomaly_timeseries.png")


def plot_forecast_comparison(model_results, n_display=90):
    """Plot predicted vs actual for each model on one figure."""
    n_models = len(model_results)
    if n_models == 0:
        return

    fig, axes = plt.subplots(n_models, 1, figsize=(14, 4 * n_models), squeeze=False)

    for ax, (model_name, res) in zip(axes.flatten(), model_results.items()):
        if "test_preds" not in res or "test_actual" not in res:
            ax.set_visible(False)
            continue
        actual = np.asarray(res["test_actual"])[-n_display:]
        preds = np.asarray(res["test_preds"])[-n_display:]
        x = np.arange(len(actual))
        ax.plot(x, actual, label="Actual", color="steelblue", lw=1.5)
        ax.plot(x, preds, label="Predicted", color="darkorange", lw=1.5, linestyle="--")
        metrics = res.get("metrics", {})
        title = f"{model_name} - MAE={metrics.get('MAE', 'N/A')} | RMSE={metrics.get('RMSE', 'N/A')}"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Temperature (°C)")
        ax.legend(fontsize=9)

    axes[-1][0].set_xlabel("Test Step")
    fig.suptitle("Forecast vs Actual (Test Set)", fontsize=14, fontweight="bold")
    return _save(fig, "forecast_comparison.png")


def plot_ensemble_comparison(comparison_df):
    """Bar chart comparing MAE/RMSE across all models and ensembles."""
    if comparison_df.empty or "RMSE" not in comparison_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = [
        "#2196F3" if t == "Individual" else "#FF5722"
        for t in comparison_df.get("Type", ["Individual"] * len(comparison_df))
    ]

    axes[0].barh(comparison_df["Model"], comparison_df["RMSE"], color=colors, edgecolor="white")
    axes[0].set_title("RMSE by Model", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("RMSE")
    axes[0].invert_yaxis()

    if "MAE" in comparison_df.columns:
        axes[1].barh(comparison_df["Model"], comparison_df["MAE"], color=colors, edgecolor="white")
        axes[1].set_title("MAE by Model", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("MAE")
        axes[1].invert_yaxis()

    from matplotlib.patches import Patch

    legend = [Patch(color="#2196F3", label="Individual"), Patch(color="#FF5722", label="Ensemble")]
    fig.legend(handles=legend, loc="upper right")
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
    return _save(fig, "ensemble_comparison.png")


def plot_air_quality_correlations(df, target_col=TARGET_VARIABLE):
    """Scatter matrix between air quality columns and temperature."""
    aq_cols = [c for c in AIR_QUALITY_COLUMNS if c in df.columns]
    if not aq_cols:
        logger.warning("No air quality columns found.")
        return

    sample = df[[target_col] + aq_cols].dropna().sample(
        min(2000, len(df)), random_state=RANDOM_SEED
    )

    n = len(aq_cols)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(7 * ((n + 1) // 2), 10))
    axes = np.array(axes).flatten()

    for i, col in enumerate(aq_cols):
        ax = axes[i]
        ax.scatter(sample[col], sample[target_col], alpha=0.2, s=8, color="steelblue")
        try:
            from scipy.stats import pearsonr

            r, _ = pearsonr(sample[col].dropna(), sample[target_col].dropna())
            ax.set_title(f"{col}\n(r={r:.3f})", fontsize=9, fontweight="bold")
        except Exception:
            ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel(target_col, fontsize=8)

    for j in range(len(aq_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Air Quality vs Temperature Correlations", fontsize=13, fontweight="bold")
    return _save(fig, "air_quality_correlations.png")


def plot_regional_comparison(regional_df, target_col="mean_temp", top_n=20):
    """Horizontal bar chart of mean temperature by country."""
    if regional_df.empty or "country" not in regional_df.columns:
        return

    df_plot = regional_df.dropna(subset=[target_col]).head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.45)))
    colors = plt.cm.RdBu_r(
        (df_plot[target_col] - df_plot[target_col].min())
        / (df_plot[target_col].max() - df_plot[target_col].min() + 1e-9)
    )
    ax.barh(df_plot["country"], df_plot[target_col], color=colors, edgecolor="white")
    ax.set_title(f"Top {top_n} Countries by Mean Temperature", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Temperature (°C)")
    ax.invert_yaxis()
    plt.tight_layout()
    return _save(fig, "regional_temperature_comparison.png")


def run_all_visualizations(daily_df, df_clean, df_annotated, model_results, ensemble_df, regional_df):
    """Generate and save all visualization artifacts."""
    logger.info("=== Generating visualizations ===")

    plot_distributions(daily_df)
    plot_correlation_heatmap(daily_df)
    plot_time_series(daily_df)
    plot_seasonal_decomposition(daily_df)
    plot_seasonal_patterns(daily_df)
    plot_global_warming_trend(daily_df)

    if df_annotated is not None:
        plot_anomalies(df_annotated)

    if df_clean is not None:
        plot_air_quality_correlations(df_clean)

    if regional_df is not None and not regional_df.empty:
        plot_regional_comparison(regional_df)

    if model_results:
        plot_forecast_comparison(model_results)

    if ensemble_df is not None and not ensemble_df.empty:
        plot_ensemble_comparison(ensemble_df)

    logger.info("=== All visualizations saved to %s ===", FIGURES_DIR)
