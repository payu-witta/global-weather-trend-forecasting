"""
Forecasting models for global temperature prediction.

Models implemented (so far):
  1. SARIMA   - statsmodels
  2. Prophet  - Meta Prophet

Each model exposes a train() and predict() interface.
Evaluation metrics: MAE, RMSE, MAPE.
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    ARIMA_ORDER,
    ARIMA_SEASONAL_ORDER,
    FORECAST_HORIZON,
    FORECASTS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_VARIABLE,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)


def compute_metrics(y_true, y_pred, model_name="Model"):
    """Compute MAE, RMSE, MAPE."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    metrics = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}
    logger.info("%s -> MAE=%.4f | RMSE=%.4f | MAPE=%.2f%%", model_name, mae, rmse, mape)
    return metrics


def time_series_split(series, test_size=TEST_SIZE):
    """Split a Series into train/test maintaining temporal order."""
    n = len(series)
    split = int(n * (1 - test_size))
    return series.iloc[:split], series.iloc[split:]


def df_train_test_split(df, test_size=TEST_SIZE, date_col="date"):
    """Split a DataFrame maintaining temporal order."""
    n = len(df)
    split = int(n * (1 - test_size))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class SARIMAModel:
    """Seasonal ARIMA wrapper using statsmodels."""

    def __init__(self, order=ARIMA_ORDER, seasonal_order=ARIMA_SEASONAL_ORDER):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None
        self.train_series = None

    def train(self, series):
        """Fit SARIMA on a pd.Series indexed by date."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        logger.info("Training SARIMA%s x %s ...", self.order, self.seasonal_order)
        self.train_series = series
        model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_fit = model.fit(disp=False)
        self.residuals = self.model_fit.resid
        logger.info("SARIMA training complete. AIC=%.2f", self.model_fit.aic)
        return self

    def predict_with_intervals(self, steps, alpha=0.05):
        """Return point forecast + confidence intervals as a DataFrame."""
        if self.model_fit is None:
            raise RuntimeError("Model not trained. Call train() first.")
        forecast_obj = self.model_fit.get_forecast(steps=steps)
        summary = forecast_obj.summary_frame(alpha=alpha)
        return summary[["mean", "mean_ci_lower", "mean_ci_upper"]].rename(
            columns={"mean_ci_lower": "lower_ci", "mean_ci_upper": "upper_ci"}
        )

    def predict(self, steps):
        """Return in-sample + out-of-sample forecast as a pd.Series."""
        if self.model_fit is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model_fit.forecast(steps=steps)

    def evaluate(self, test_series):
        preds = self.predict(len(test_series))
        preds.index = test_series.index
        return compute_metrics(test_series.values, preds.values, "SARIMA")

    def save(self, path=None):
        if path is None:
            path = MODELS_DIR / "sarima_model.pkl"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model_fit.save(str(path))
        logger.info("SARIMA model saved -> %s", path)


class ProphetModel:
    """Meta Prophet wrapper."""

    def __init__(self, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None
        self.forecast_df = None

    def _to_prophet_df(self, series):
        """Convert pd.Series to Prophet-formatted DataFrame with ds/y columns."""
        return pd.DataFrame({"ds": series.index, "y": series.values})

    def train(self, series):
        """Fit Prophet on a pd.Series with a DatetimeIndex."""
        from prophet import Prophet

        logger.info("Training Prophet ...")
        prophet_df = self._to_prophet_df(series)
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=0.05,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(prophet_df)
        logger.info("Prophet training complete.")
        return self

    def predict(self, periods, freq="D"):
        """Return forecast DataFrame for future periods."""
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        self.forecast_df = self.model.predict(future)
        return self.forecast_df

    def predict_series(self, index):
        """Predict for specific dates given as a DatetimeIndex."""
        df_future = pd.DataFrame({"ds": index})
        forecast = self.model.predict(df_future)
        return pd.Series(forecast["yhat"].values, index=index)

    def evaluate(self, test_series):
        preds = self.predict_series(test_series.index)
        return compute_metrics(test_series.values, preds.values, "Prophet")

    def save(self, path=None):
        import json
        from prophet.serialize import model_to_json

        if path is None:
            path = MODELS_DIR / "prophet_model.json"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(model_to_json(self.model), f)
        logger.info("Prophet model saved -> %s", path)
