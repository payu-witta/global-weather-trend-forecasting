"""
Statistical diagnostics for the time-series forecasting pipeline.

Implements:
  - Augmented Dickey-Fuller (ADF) stationarity test
  - ACF and PACF plots
  - SARIMA residual diagnostics (4-panel: residuals, histogram, Q-Q, residual ACF)
  - Heteroskedasticity check (Ljung-Box on squared residuals)

All figures saved to outputs/figures/; ADF results to outputs/reports/.
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import FIGURE_DPI, FIGURES_DIR, REPORTS_DIR, TARGET_VARIABLE

logger = logging.getLogger(__name__)


def run_adf_test(series, label=TARGET_VARIABLE):
    """
    Augmented Dickey-Fuller test for unit root (non-stationarity).

    H0: series has a unit root (non-stationary)
    Reject H0 at p < 0.05 => stationary

    Returns
    -------
    dict with adf_stat, p_value, n_lags_used, critical_values, is_stationary
    """
    from statsmodels.tsa.stattools import adfuller

    series = series.dropna()
    result = adfuller(series, autolag="AIC")

    adf_stat, p_value, n_lags, nobs, crit_vals, _ = result
    is_stationary = p_value < 0.05

    summary = {
        "series": label,
        "adf_statistic": round(adf_stat, 6),
        "p_value": round(p_value, 6),
        "n_lags_used": int(n_lags),
        "n_observations": int(nobs),
        "critical_value_1pct": round(crit_vals["1%"], 4),
        "critical_value_5pct": round(crit_vals["5%"], 4),
        "critical_value_10pct": round(crit_vals["10%"], 4),
        "is_stationary": bool(is_stationary),
        "conclusion": "Stationary (reject H0)" if is_stationary else "Non-stationary (fail to reject H0)",
    }

    logger.info(
        "ADF test on '%s': stat=%.4f, p=%.4f => %s",
        label,
        adf_stat,
        p_value,
        summary["conclusion"],
    )
    return summary


def save_adf_result(adf_result):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([adf_result]).to_csv(REPORTS_DIR / "adf_stationarity_result.csv", index=False)
    logger.info("ADF result saved -> %s", REPORTS_DIR / "adf_stationarity_result.csv")


def plot_acf_pacf(series, lags=40, save_path=None):
    """
    Side-by-side ACF and PACF plots with 95% confidence bands.
    Useful for identifying AR/MA order for ARIMA.
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    series = series.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_acf(series, lags=lags, ax=axes[0], alpha=0.05, title="")
    axes[0].set_title("Autocorrelation Function (ACF)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Lag (days)")
    axes[0].set_ylabel("Autocorrelation")

    plot_pacf(series, lags=lags, ax=axes[1], alpha=0.05, method="ywm", title="")
    axes[1].set_title("Partial Autocorrelation Function (PACF)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Lag (days)")
    axes[1].set_ylabel("Partial Autocorrelation")

    fig.suptitle(
        f"ACF / PACF - {TARGET_VARIABLE} (lags=1..{lags})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / f"acf_pacf_{TARGET_VARIABLE}.png"

    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("ACF/PACF plot saved -> %s", save_path)
    return save_path


def plot_residual_diagnostics(residuals, model_name="SARIMA", save_path=None):
    """
    Four-panel residual diagnostic plot:
      1. Residuals over time
      2. Histogram with normal overlay
      3. Q-Q plot
      4. ACF of residuals (checking for remaining autocorrelation)
    """
    from scipy import stats
    from statsmodels.graphics.tsaplots import plot_acf

    residuals = pd.Series(residuals).dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(residuals.values, color="steelblue", lw=0.8, alpha=0.8)
    axes[0, 0].axhline(0, color="red", lw=1, linestyle="--")
    axes[0, 0].set_title("Residuals Over Time", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Residual")

    axes[0, 1].hist(residuals, bins=40, density=True, color="steelblue", edgecolor="white", alpha=0.7)
    x = np.linspace(residuals.min(), residuals.max(), 200)
    axes[0, 1].plot(
        x, stats.norm.pdf(x, residuals.mean(), residuals.std()), color="red", lw=2, label="N(0,sigma)"
    )
    axes[0, 1].set_title("Residual Distribution", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Residual Value")
    axes[0, 1].legend()

    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot (Normal)", fontsize=11, fontweight="bold")
    axes[1, 0].get_lines()[0].set(markersize=3, alpha=0.5)

    plot_acf(residuals, lags=30, ax=axes[1, 1], alpha=0.05, title="")
    axes[1, 1].set_title("ACF of Residuals", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Lag")

    fig.suptitle(f"Residual Diagnostics - {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / f"{model_name.lower()}_residual_diagnostics.png"

    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Residual diagnostics saved -> %s", save_path)
    return save_path


def ljung_box_test(residuals, lags=10):
    """
    Ljung-Box test on residuals and squared residuals.
    Significant p-value on squared residuals indicates heteroskedasticity.

    Returns DataFrame with columns: lag, lb_stat, lb_pvalue, squared_lb_stat, squared_lb_pvalue
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    residuals = pd.Series(residuals).dropna()
    lb_raw = acorr_ljungbox(residuals, lags=lags, return_df=True)
    lb_sq = acorr_ljungbox(residuals**2, lags=lags, return_df=True)

    result = pd.DataFrame({
        "lag": lb_raw.index,
        "lb_stat": lb_raw["lb_stat"].values.round(4),
        "lb_pvalue": lb_raw["lb_pvalue"].values.round(4),
        "squared_lb_stat": lb_sq["lb_stat"].values.round(4),
        "squared_lb_pvalue": lb_sq["lb_pvalue"].values.round(4),
    })

    heteroskedastic = (result["squared_lb_pvalue"] < 0.05).any()
    logger.info(
        "Ljung-Box: residuals autocorrelated=%s, heteroskedastic=%s",
        (result["lb_pvalue"] < 0.05).any(),
        heteroskedastic,
    )
    return result


def run_statistical_diagnostics(series, model_results=None):
    """
    Run all statistical diagnostics.

    Parameters
    ----------
    series : pd.Series
        Global daily temperature time series (DatetimeIndex).
    model_results : dict, optional
        Output of forecasting_models.run_all_models(). Used to extract SARIMA residuals.

    Returns
    -------
    dict with keys: adf, ljung_box (if model available)
    """
    logger.info("=== Statistical diagnostics started ===")

    adf = run_adf_test(series)
    save_adf_result(adf)

    # ACF / PACF on the raw series and on the first difference
    plot_acf_pacf(series, lags=40)

    series_diff = series.diff().dropna()
    plot_acf_pacf(series_diff, lags=40, save_path=FIGURES_DIR / "acf_pacf_differenced.png")

    results = {"adf": adf}

    if model_results is not None:
        for model_name in ("SARIMA", "Prophet"):
            if model_name in model_results:
                res_obj = model_results[model_name].get("model_obj")
                residuals = None

                if model_name == "SARIMA" and res_obj is not None:
                    residuals = getattr(res_obj, "residuals", None)
                    if residuals is None and hasattr(res_obj, "model_fit"):
                        residuals = res_obj.model_fit.resid

                if residuals is not None:
                    plot_residual_diagnostics(residuals, model_name)
                    lb = ljung_box_test(residuals)
                    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                    lb.to_csv(REPORTS_DIR / f"ljung_box_{model_name}.csv", index=False)
                    results[f"ljung_box_{model_name}"] = lb

    logger.info("=== Statistical diagnostics complete ===")
    return results
