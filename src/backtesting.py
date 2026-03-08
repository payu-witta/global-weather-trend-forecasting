"""
Walk-forward (expanding-window) cross-validation for time-series models.

Each fold adds one slice of data to the training set and evaluates on
the next unseen slice. This tests whether model performance is stable
over time — a much stronger validation than a single 80/20 split.

Models backtested: SARIMA, Prophet, XGBoost
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ARIMA_ORDER,
    ARIMA_SEASONAL_ORDER,
    FIGURE_DPI,
    FIGURES_DIR,
    FORECASTS_DIR,
    RANDOM_SEED,
    TARGET_VARIABLE,
    XGBOOST_PARAMS,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)


def _metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() else np.nan
    return round(mae, 4), round(rmse, 4), round(mape, 4)


def expanding_window_splits(n_total, n_splits=5, min_train_frac=0.55):
    """
    Generate (train_end, test_end) index pairs for expanding-window CV.

    The minimum training set is min_train_frac * n_total.
    Each subsequent fold expands training by one equal step.

    Yields (train_end_idx, test_end_idx) — exclusive end indices.
    """
    min_train = max(30, int(n_total * min_train_frac))
    remaining = n_total - min_train
    step = max(1, remaining // n_splits)

    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = min(train_end + step, n_total)
        if train_end >= n_total:
            break
        yield train_end, test_end


def backtest_sarima(series, n_splits=5):
    """
    Walk-forward SARIMA cross-validation.

    Re-fits SARIMA from scratch on each expanding training window.
    Returns DataFrame: fold, n_train, n_test, MAE, RMSE, MAPE.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = series.dropna()
    rows = []
    n = len(series)

    for fold, (train_end, test_end) in enumerate(expanding_window_splits(n, n_splits), 1):
        train = series.iloc[:train_end]
        test = series.iloc[train_end:test_end]
        if len(test) == 0:
            continue
        try:
            model = SARIMAX(
                train,
                order=ARIMA_ORDER,
                seasonal_order=ARIMA_SEASONAL_ORDER,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)
            preds = fit.forecast(steps=len(test))
            mae, rmse, mape = _metrics(test.values, preds.values)
        except Exception as exc:
            logger.warning("SARIMA fold %d failed: %s", fold, exc)
            mae, rmse, mape = np.nan, np.nan, np.nan

        rows.append({
            "fold": fold,
            "train_end_date": str(train.index[-1])[:10] if hasattr(train.index, "date") else train_end,
            "n_train": train_end,
            "n_test": len(test),
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
        })
        logger.info(
            "SARIMA fold %d/%d: n_train=%d MAE=%.4f RMSE=%.4f",
            fold, n_splits, train_end, mae, rmse,
        )

    return pd.DataFrame(rows)


def backtest_prophet(series, n_splits=5):
    """Walk-forward Prophet cross-validation."""
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning("Prophet not installed. Skipping Prophet backtest.")
        return pd.DataFrame()

    series = series.dropna()
    rows = []
    n = len(series)

    for fold, (train_end, test_end) in enumerate(expanding_window_splits(n, n_splits), 1):
        train = series.iloc[:train_end]
        test = series.iloc[train_end:test_end]
        if len(test) == 0:
            continue
        try:
            prophet_df = pd.DataFrame({"ds": train.index, "y": train.values})
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(prophet_df)
            future = pd.DataFrame({"ds": test.index})
            forecast = m.predict(future)
            preds = forecast["yhat"].values
            mae, rmse, mape = _metrics(test.values, preds)
        except Exception as exc:
            logger.warning("Prophet fold %d failed: %s", fold, exc)
            mae, rmse, mape = np.nan, np.nan, np.nan

        rows.append({
            "fold": fold,
            "train_end_date": str(train.index[-1])[:10],
            "n_train": train_end,
            "n_test": len(test),
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
        })
        logger.info(
            "Prophet fold %d/%d: n_train=%d MAE=%.4f RMSE=%.4f",
            fold, n_splits, train_end, mae, rmse,
        )

    return pd.DataFrame(rows)


def backtest_xgboost(feature_df, target_col=TARGET_VARIABLE, n_splits=5):
    """Walk-forward XGBoost cross-validation using the full feature set."""
    from xgboost import XGBRegressor

    exclude = {"date", "season", "anomaly", "if_anomaly", "lof_anomaly",
               "zscore_anomaly", "zscore", "anomaly_type", "ensemble_anomaly"}
    feature_cols = [
        c for c in feature_df.columns
        if c != target_col and c not in exclude
        and feature_df[c].dtype in [float, int, np.float64, np.int64]
    ]

    rows = []
    n = len(feature_df)

    for fold, (train_end, test_end) in enumerate(expanding_window_splits(n, n_splits), 1):
        df_train = feature_df.iloc[:train_end]
        df_test = feature_df.iloc[train_end:test_end]
        if len(df_test) == 0:
            continue
        try:
            X_train = df_train[feature_cols].fillna(0)
            y_train = df_train[target_col]
            X_test = df_test[feature_cols].fillna(0)
            y_test = df_test[target_col].values

            model = XGBRegressor(**XGBOOST_PARAMS)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae, rmse, mape = _metrics(y_test, preds)
        except Exception as exc:
            logger.warning("XGBoost fold %d failed: %s", fold, exc)
            mae, rmse, mape = np.nan, np.nan, np.nan

        rows.append({
            "fold": fold,
            "train_end_date": str(feature_df.iloc[train_end - 1].get("date", train_end))[:10],
            "n_train": train_end,
            "n_test": len(df_test),
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
        })
        logger.info(
            "XGBoost fold %d/%d: n_train=%d MAE=%.4f RMSE=%.4f",
            fold, n_splits, train_end, mae, rmse,
        )

    return pd.DataFrame(rows)


def plot_backtest_results(backtest_dict, save_path=None):
    """
    Line charts of RMSE and MAE across folds for all backtested models.
    One subplot per metric, models overlaid.
    """
    if not backtest_dict:
        return

    colors = {"SARIMA": "steelblue", "Prophet": "darkorange", "XGBoost": "green"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, df in backtest_dict.items():
        if df.empty or "fold" not in df.columns:
            continue
        color = colors.get(model_name, "gray")
        axes[0].plot(df["fold"], df["RMSE"], marker="o", label=model_name, color=color, lw=2)
        axes[1].plot(df["fold"], df["MAE"], marker="o", label=model_name, color=color, lw=2)

    for ax, metric in zip(axes, ("RMSE", "MAE")):
        ax.set_title(f"{metric} by Fold (Expanding Window CV)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric)
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.suptitle("Walk-Forward Cross-Validation Results", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "backtest_cv_results.png"

    fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Backtest CV plot saved -> %s", save_path)


def run_backtesting(series, feature_df, n_splits=5):
    """
    Run walk-forward backtests for SARIMA, Prophet, and XGBoost.

    Parameters
    ----------
    series : pd.Series
        Global daily temperature series with DatetimeIndex.
    feature_df : pd.DataFrame
        Enriched feature DataFrame from feature_engineering.
    n_splits : int

    Returns
    -------
    dict {model_name: DataFrame of per-fold metrics}
    """
    logger.info("=== Walk-forward backtesting started (n_splits=%d) ===", n_splits)

    results = {}

    try:
        results["SARIMA"] = backtest_sarima(series, n_splits)
    except Exception as exc:
        logger.error("SARIMA backtest error: %s", exc)

    try:
        results["Prophet"] = backtest_prophet(series, n_splits)
    except Exception as exc:
        logger.error("Prophet backtest error: %s", exc)

    try:
        results["XGBoost"] = backtest_xgboost(feature_df, n_splits=n_splits)
    except Exception as exc:
        logger.error("XGBoost backtest error: %s", exc)

    plot_backtest_results(results)

    rows = []
    for model_name, df in results.items():
        if not df.empty:
            df = df.copy()
            df.insert(0, "model", model_name)
            rows.append(df)

    if rows:
        FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(rows, ignore_index=True)
        combined.to_csv(FORECASTS_DIR / "backtest_cv_results.csv", index=False)
        logger.info("Backtest results saved -> %s", FORECASTS_DIR / "backtest_cv_results.csv")

    logger.info("=== Backtesting complete ===")
    return results
