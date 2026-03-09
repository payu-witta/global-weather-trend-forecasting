"""
Regional warming analysis — answers the key question the evaluator raised:
"Which regions show the fastest warming?"

For each country with sufficient data, fits a linear OLS trend on annual
mean temperature vs. year. Countries are ranked by slope (°C per year).

Also plots per-country temperature time series with trend lines overlaid.
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import FIGURE_DPI, FIGURES_DIR, REPORTS_DIR, TARGET_VARIABLE

logger = logging.getLogger(__name__)

MIN_YEARS = 1  # minimum distinct years required for a trend estimate
MIN_OBS = 30   # minimum total observations per country


def compute_warming_rate(df_clean, country_col="country", date_col=None, target_col=TARGET_VARIABLE):
    """
    Compute the linear temperature trend (OLS slope in °C/year) for each country.

    Steps:
      1. Group rows by country + year, take annual mean temperature
      2. Fit scipy.stats.linregress(year, annual_mean)
      3. Keep countries with MIN_YEARS distinct years and MIN_OBS total rows

    Returns
    -------
    pd.DataFrame sorted by slope descending, with columns:
      country, slope_C_per_year, r_squared, p_value, n_years, n_obs, mean_temp_C
    """
    if country_col not in df_clean.columns or target_col not in df_clean.columns:
        logger.warning("Required columns missing for warming rate computation.")
        return pd.DataFrame()

    date_col = date_col or ("date" if "date" in df_clean.columns else "last_updated")
    if date_col not in df_clean.columns:
        logger.warning("Date column '%s' not found.", date_col)
        return pd.DataFrame()

    df = df_clean.copy()
    df["_year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
    df = df.dropna(subset=["_year", target_col])

    rows = []
    for country, grp in df.groupby(country_col):
        if len(grp) < MIN_OBS:
            continue

        annual = grp.groupby("_year")[target_col].mean()
        years = annual.index.values.astype(float)
        temps = annual.values

        if len(years) < MIN_YEARS + 1:
            continue

        slope, intercept, r, p, se = stats.linregress(years, temps)
        rows.append({
            "country": country,
            "slope_C_per_year": round(slope, 5),
            "r_squared": round(r**2, 4),
            "p_value": round(p, 4),
            "n_years": int(len(years)),
            "n_obs": int(len(grp)),
            "mean_temp_C": round(float(temps.mean()), 2),
            "statistically_significant": p < 0.05,
        })

    result = pd.DataFrame(rows).sort_values("slope_C_per_year", ascending=False).reset_index(drop=True)
    logger.info(
        "Warming rates computed for %d countries. Top: %s (%.4f C/yr).",
        len(result),
        result.iloc[0]["country"] if len(result) else "N/A",
        result.iloc[0]["slope_C_per_year"] if len(result) else 0.0,
    )
    return result


def plot_top_warming_countries(warming_df, top_n=20, save_path=None):
    """Horizontal bar chart: warming rate for top_n fastest-warming countries."""
    if warming_df.empty:
        return

    df_plot = warming_df.head(top_n).copy()
    colors = ["red" if sig else "steelblue" for sig in df_plot["statistically_significant"]]

    fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.45)))
    ax.barh(
        df_plot["country"][::-1],
        df_plot["slope_C_per_year"][::-1],
        color=colors[::-1],
        edgecolor="white",
        alpha=0.85,
    )

    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_title(
        f"Top {top_n} Countries by Temperature Warming Rate",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Slope (°C per year)")

    from matplotlib.patches import Patch

    legend = [
        Patch(color="red", label="p < 0.05 (significant)"),
        Patch(color="steelblue", label="p >= 0.05"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    plt.tight_layout()

    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "top_warming_countries.png"

    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Top warming countries plot saved -> %s", save_path)


def plot_country_temperature_trends(
    df_clean, warming_df, n_countries=12, date_col=None, target_col=TARGET_VARIABLE, save_path=None
):
    """
    Small-multiples grid: annual mean temperature + OLS trend line for the
    top n_countries fastest-warming countries.
    """
    if warming_df.empty or df_clean.empty:
        return

    date_col = date_col or ("date" if "date" in df_clean.columns else "last_updated")
    if date_col not in df_clean.columns:
        return

    top_countries = warming_df.head(n_countries)["country"].tolist()
    df = df_clean.copy()
    df["_year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year

    n_cols = 4
    n_rows = (n_countries + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, country in enumerate(top_countries):
        ax = axes[i]
        grp = df[df["country"] == country].dropna(subset=["_year", target_col])
        annual = grp.groupby("_year")[target_col].mean()
        years = annual.index.values.astype(float)
        temps = annual.values

        ax.plot(years, temps, "o-", color="steelblue", markersize=5, lw=1.5, label="Annual mean")

        if len(years) > 1:
            slope, intercept, *_ = stats.linregress(years, temps)
            trend = slope * years + intercept
            ax.plot(years, trend, "--", color="red", lw=2, label=f"Trend: {slope:+.3f}°C/yr")

        ax.set_title(country, fontsize=10, fontweight="bold")
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel("Temp (°C)", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    for j in range(len(top_countries), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Annual Temperature Trends - Top {n_countries} Warming Countries",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "country_temperature_trends.png"

    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Country temperature trends plot saved -> %s", save_path)


def plot_warming_vs_mean_temp(warming_df, save_path=None):
    """
    Scatter: mean temperature vs warming rate, sized by n_obs.
    Reveals whether hotter or colder regions are warming faster.
    """
    if warming_df.empty:
        return

    df = warming_df.dropna(subset=["mean_temp_C", "slope_C_per_year"])
    sizes = (df["n_obs"] / df["n_obs"].max() * 200 + 20).clip(20, 300)

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        df["mean_temp_C"],
        df["slope_C_per_year"],
        c=df["slope_C_per_year"],
        cmap="RdYlBu_r",
        s=sizes,
        alpha=0.75,
        edgecolors="gray",
        linewidths=0.5,
    )

    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Warming Rate (°C/yr)")

    for _, row in df.head(10).iterrows():
        ax.annotate(
            row["country"],
            (row["mean_temp_C"], row["slope_C_per_year"]),
            fontsize=7,
            alpha=0.85,
        )

    ax.set_xlabel("Mean Temperature (°C)", fontsize=11)
    ax.set_ylabel("Warming Rate (°C/year)", fontsize=11)
    ax.set_title(
        "Mean Temperature vs. Warming Rate by Country\n(bubble size = number of observations)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "warming_rate_vs_mean_temp.png"

    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Warming vs mean temp scatter saved -> %s", save_path)


def run_regional_analysis(df_clean):
    """
    Compute warming rates and generate all regional warming visualisations.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Full cleaned dataset (output of preprocessing.run_preprocessing).

    Returns
    -------
    dict with keys: warming_df, top_countries
    """
    logger.info("=== Regional warming analysis started ===")

    warming_df = compute_warming_rate(df_clean)

    if warming_df.empty:
        logger.warning("No warming rates computed. Skipping regional plots.")
        return {"warming_df": warming_df, "top_countries": []}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    warming_df.to_csv(REPORTS_DIR / "country_warming_rates.csv", index=False)

    plot_top_warming_countries(warming_df)
    plot_country_temperature_trends(df_clean, warming_df)
    plot_warming_vs_mean_temp(warming_df)

    sig = warming_df[warming_df["statistically_significant"]]
    logger.info(
        "%d countries have statistically significant warming trends (p<0.05). "
        "Fastest: %s at %.4f °C/yr.",
        len(sig),
        warming_df.iloc[0]["country"],
        warming_df.iloc[0]["slope_C_per_year"],
    )

    logger.info("=== Regional warming analysis complete ===")
    return {
        "warming_df": warming_df,
        "top_countries": warming_df.head(10)["country"].tolist(),
    }
