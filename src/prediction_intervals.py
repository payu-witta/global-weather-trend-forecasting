"""
Forecast prediction intervals for all three model families.

Sources of uncertainty:
  - SARIMA   : statsmodels get_forecast() 95% confidence intervals
  - Prophet  : built-in yhat_lower / yhat_upper (80% by default)
  - XGBoost  : quantile regression (P10 / P90 bands)

All plots saved to outputs/figures/.
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ARIMA_ORDER,
    ARIMA_SEASONAL_ORDER,
    FIGURE_DPI,
    FIGURES_DIR,
    FORECASTS_DIR,
    RANDOM_SEED,
    TARGET_VARIABLE,
    TEST_SIZE,
    XGBOOST_PARAMS,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)


def _plot_interval(actual, point_pred, lower, upper, model_name, interval_label, save_path):
    """Actual vs predicted with shaded uncertainty band."""
    n = min(len(actual), len(point_pred), len(lower), len(upper))
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, actual[:n], color="steelblue", lw=1.5, label="Actual")
    ax.plot(x, point_pred[:n], color="darkorange", lw=1.5, linestyle="--", label="Predicted")
    ax.fill_between(
        x,
        lower[:n],
        upper[:n],
        alpha=0.25,
        color="darkorange",
        label=f"{interval_label} interval",
    )
    ax.set_title(
        f"{model_name} Forecast with {interval_label} Prediction Interval",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Test Step")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    plt.tight_layout()

    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Prediction interval plot saved -> %s", save_path)


def sarima_prediction_intervals(ts_train, ts_test, alpha=0.05):
    """
    Fit SARIMA on ts_train, produce forecast + confidence intervals for ts_test.

    Returns DataFrame: index, mean, lower_ci, upper_ci
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        ts_train,
        order=ARIMA_ORDER,
        seasonal_order=ARIMA_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    forecast_obj = fit.get_forecast(steps=len(ts_test))
    summary = forecast_obj.summary_frame(alpha=alpha)

    result = pd.DataFrame(
        {
            "mean": summary["mean"].values,
            "lower_ci": summary["mean_ci_lower"].values,
            "upper_ci": summary["mean_ci_upper"].values,
        },
        index=ts_test.index,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_interval(
        ts_test.values,
        result["mean"].values,
        result["lower_ci"].values,
        result["upper_ci"].values,
        "SARIMA",
        f"{int((1 - alpha) * 100)}%",
        FIGURES_DIR / "sarima_prediction_intervals.png",
    )
    return result


def prophet_prediction_intervals(ts_train, ts_test, interval_width=0.80):
    """
    Fit Prophet on ts_train; predict with built-in uncertainty intervals.

    Returns DataFrame: ds, yhat, yhat_lower, yhat_upper
    """
    from prophet import Prophet

    prophet_df = pd.DataFrame({"ds": ts_train.index, "y": ts_train.values})
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=interval_width,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(prophet_df)

    future = pd.DataFrame({"ds": ts_test.index})
    forecast = m.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_interval(
        ts_test.values,
        forecast["yhat"].values,
        forecast["yhat_lower"].values,
        forecast["yhat_upper"].values,
        "Prophet",
        f"{int(interval_width * 100)}%",
        FIGURES_DIR / "prophet_prediction_intervals.png",
    )
    return result


def xgboost_quantile_intervals(feature_df, target_col=TARGET_VARIABLE, lower_q=0.10, upper_q=0.90):
    """
    Train lower- and upper-quantile XGBoost models (P10/P90 band).

    Requires XGBoost >= 1.7 for objective='reg:quantileerror'.
    Falls back to training a single median model if quantile objective unavailable.

    Returns DataFrame: lower, upper (test-set rows only)
    """
    from xgboost import XGBRegressor

    exclude = {"date", "season", "anomaly", "if_anomaly", "lof_anomaly",
               "zscore_anomaly", "zscore", "anomaly_type", "ensemble_anomaly"}
    feature_cols = [
        c for c in feature_df.columns
        if c != target_col and c not in exclude
        and feature_df[c].dtype in [float, int, np.float64, np.int64]
    ]

    n = len(feature_df)
    n_train = int(n * (1 - TEST_SIZE))
    df_train = feature_df.iloc[:n_train]
    df_test = feature_df.iloc[n_train:]

    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train[target_col]
    X_test = df_test[feature_cols].fillna(0)
    y_actual = df_test[target_col].values

    q_params = {k: v for k, v in XGBOOST_PARAMS.items() if k != "random_state"}
    q_params["random_state"] = RANDOM_SEED

    try:
        lower_model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=lower_q, **q_params)
        lower_model.fit(X_train, y_train)
        lower_preds = lower_model.predict(X_test)

        upper_model = XGBRegressor(objective="reg:quantileerror", quantile_alpha=upper_q, **q_params)
        upper_model.fit(X_train, y_train)
        upper_preds = upper_model.predict(X_test)

        point_model = XGBRegressor(**XGBOOST_PARAMS)
        point_model.fit(X_train, y_train)
        point_preds = point_model.predict(X_test)

    except Exception as exc:
        logger.warning("Quantile XGBoost failed (%s). Using ±1.5*std heuristic band.", exc)
        point_model = XGBRegressor(**XGBOOST_PARAMS)
        point_model.fit(X_train, y_train)
        point_preds = point_model.predict(X_test)
        residuals_std = np.std(y_train.values - point_model.predict(X_train))
        lower_preds = point_preds - 1.645 * residuals_std
        upper_preds = point_preds + 1.645 * residuals_std

    result = pd.DataFrame(
        {"lower": lower_preds, "point": point_preds, "upper": upper_preds},
        index=df_test.index,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_interval(
        y_actual,
        point_preds,
        lower_preds,
        upper_preds,
        "XGBoost",
        f"P{int(lower_q*100)}-P{int(upper_q*100)}",
        FIGURES_DIR / "xgboost_prediction_intervals.png",
    )
    return result


def run_prediction_intervals(series, feature_df):
    """
    Generate prediction intervals for SARIMA, Prophet, and XGBoost.

    Parameters
    ----------
    series : pd.Series
        Full global daily temperature series (DatetimeIndex).
    feature_df : pd.DataFrame
        Enriched feature DataFrame.

    Returns
    -------
    dict {model_name: interval DataFrame}
    """
    logger.info("=== Prediction intervals started ===")

    series = series.dropna()
    n_test = int(len(series) * TEST_SIZE)
    ts_train = series.iloc[:-n_test]
    ts_test = series.iloc[-n_test:]

    results = {}

    try:
        results["SARIMA"] = sarima_prediction_intervals(ts_train, ts_test)
    except Exception as exc:
        logger.error("SARIMA intervals failed: %s", exc)

    try:
        results["Prophet"] = prophet_prediction_intervals(ts_train, ts_test)
    except Exception as exc:
        logger.error("Prophet intervals failed: %s", exc)

    try:
        results["XGBoost"] = xgboost_quantile_intervals(feature_df)
    except Exception as exc:
        logger.error("XGBoost quantile intervals failed: %s", exc)

    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for name, df in results.items():
        row = {"model": name, "n_test_points": len(df)}
        if "lower_ci" in df.columns:
            row["mean_interval_width"] = round(float((df["upper_ci"] - df["lower_ci"]).mean()), 4)
        elif "yhat_lower" in df.columns:
            row["mean_interval_width"] = round(float((df["yhat_upper"] - df["yhat_lower"]).mean()), 4)
        elif "lower" in df.columns:
            row["mean_interval_width"] = round(float((df["upper"] - df["lower"]).mean()), 4)
        summary_rows.append(row)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            FORECASTS_DIR / "prediction_intervals_summary.csv", index=False
        )

    logger.info("=== Prediction intervals complete ===")
    return results
