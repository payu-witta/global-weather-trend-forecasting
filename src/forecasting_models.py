"""
Forecasting models for global temperature prediction.

Models implemented:
  1. SARIMA   - statsmodels
  2. Prophet  - Meta Prophet
  3. XGBoost  - gradient boosting with lag/calendar features
  4. LightGBM - gradient boosting with lag/calendar features
  5. LSTM     - PyTorch sequence model

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
    LIGHTGBM_PARAMS,
    LSTM_PARAMS,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_VARIABLE,
    TEST_SIZE,
    XGBOOST_PARAMS,
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


class GradientBoostingModel:
    """XGBoost and LightGBM gradient boosting wrapper for tabular forecasting."""

    def __init__(self, backend="xgboost", params=None):
        self.backend = backend
        self.params = params or (XGBOOST_PARAMS if backend == "xgboost" else LIGHTGBM_PARAMS)
        self.model = None
        self.feature_cols = None

    def _build_model(self):
        if self.backend == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(**self.params)
        else:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**self.params)

    def train(self, df_train, target_col=TARGET_VARIABLE, exclude_cols=None):
        """Train on tabular features."""
        if exclude_cols is None:
            exclude_cols = {"date", "season", "anomaly", "if_anomaly", "lof_anomaly",
                            "zscore_anomaly", "zscore", "anomaly_type", "ensemble_anomaly"}

        self.feature_cols = [
            c for c in df_train.columns
            if c != target_col and c not in exclude_cols
            and df_train[c].dtype in [float, int, np.float64, np.int64]
        ]

        X = df_train[self.feature_cols].fillna(0)
        y = df_train[target_col]

        logger.info(
            "Training %s on %d features, %d samples ...",
            self.backend.upper(), len(self.feature_cols), len(X),
        )
        self.model = self._build_model()
        self.model.fit(X, y)
        logger.info("%s training complete.", self.backend.upper())
        return self

    def predict(self, df):
        X = df[self.feature_cols].fillna(0)
        return pd.Series(self.model.predict(X), index=df.index)

    def evaluate(self, df_test, target_col=TARGET_VARIABLE):
        preds = self.predict(df_test)
        return compute_metrics(df_test[target_col].values, preds.values, self.backend.upper())

    def get_feature_importance(self):
        scores = self.model.feature_importances_
        return pd.Series(scores, index=self.feature_cols).sort_values(ascending=False)

    def save(self, path=None):
        import joblib
        if path is None:
            path = MODELS_DIR / f"{self.backend}_model.pkl"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("%s model saved -> %s", self.backend.upper(), path)


class LSTMDataset:
    """PyTorch Dataset for sequence-to-one prediction."""

    def __init__(self, data, sequence_length):
        import torch
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = sequence_length

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len, 0]
        return x, y


class LSTMNet:
    """Simple stacked LSTM implemented with PyTorch."""

    def __init__(self, input_size, params=None):
        self.params = params or LSTM_PARAMS
        self.input_size = input_size
        self.net = None
        self.scaler = None

    def _build_net(self):
        import torch.nn as nn
        p = self.params

        class Net(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        return Net(self.input_size, p["hidden_size"], p["num_layers"], p["dropout"])

    def train(self, data_array, target_col_idx=0):
        """Train LSTM on a 2D numpy array [timesteps x features]."""
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import MinMaxScaler
        from torch.utils.data import DataLoader

        logger.info("Training LSTM ...")
        p = self.params

        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(data_array)

        dataset = LSTMDataset(data_scaled, p["sequence_length"])
        loader = DataLoader(dataset, batch_size=p["batch_size"], shuffle=False)

        self.net = self._build_net()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=p["learning_rate"])
        criterion = nn.MSELoss()

        self.net.train()
        for epoch in range(p["epochs"]):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self.net(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info(
                    "LSTM epoch %d/%d | loss=%.6f",
                    epoch + 1, p["epochs"], total_loss / len(loader),
                )

        logger.info("LSTM training complete.")
        return self

    def predict(self, data_array):
        """Predict on new data."""
        import torch
        p = self.params
        data_scaled = self.scaler.transform(data_array)
        sequences = []
        for i in range(len(data_scaled) - p["sequence_length"]):
            sequences.append(data_scaled[i : i + p["sequence_length"]])

        if not sequences:
            return np.array([])

        X = torch.tensor(np.array(sequences), dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            preds_scaled = self.net(X).numpy()

        dummy = np.zeros((len(preds_scaled), data_array.shape[1]))
        dummy[:, 0] = preds_scaled
        preds = self.scaler.inverse_transform(dummy)[:, 0]
        return preds

    def save(self, path=None):
        import torch
        if path is None:
            path = MODELS_DIR / "lstm_model.pth"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), str(path))
        logger.info("LSTM weights saved -> %s", path)


def run_all_models(daily_df, feature_df, target_col=TARGET_VARIABLE):
    """
    Train all forecasting models and return their predictions and metrics.

    Returns
    -------
    results : dict
        Keys: model name -> {"metrics": dict, "test_preds": Series, "test_actual": Series}
    """
    logger.info("=== Training all forecasting models ===")
    results = {}

    ts = daily_df.set_index("date")[target_col].dropna()
    ts.index = pd.to_datetime(ts.index)
    ts = ts.asfreq("D").interpolate()

    n_test = int(len(ts) * TEST_SIZE)
    ts_train, ts_test = ts.iloc[:-n_test], ts.iloc[-n_test:]

    # SARIMA
    try:
        sarima = SARIMAModel()
        sarima.train(ts_train)
        sarima_preds = sarima.predict(n_test)
        sarima_preds.index = ts_test.index
        metrics = compute_metrics(ts_test.values, sarima_preds.values, "SARIMA")
        results["SARIMA"] = {
            "metrics": metrics, "test_preds": sarima_preds,
            "test_actual": ts_test, "model_obj": sarima,
        }
        sarima.save()
    except Exception as exc:
        logger.error("SARIMA failed: %s", exc)

    # Prophet
    try:
        prophet_model = ProphetModel()
        prophet_model.train(ts_train)
        prophet_preds = prophet_model.predict_series(ts_test.index)
        metrics = compute_metrics(ts_test.values, prophet_preds.values, "Prophet")
        results["Prophet"] = {
            "metrics": metrics, "test_preds": prophet_preds, "test_actual": ts_test,
        }
        prophet_model.save()
    except Exception as exc:
        logger.error("Prophet failed: %s", exc)

    # XGBoost
    try:
        df_train_feat, df_test_feat = df_train_test_split(feature_df)
        xgb_model = GradientBoostingModel(backend="xgboost")
        xgb_model.train(df_train_feat, target_col)
        xgb_preds = xgb_model.predict(df_test_feat)
        metrics = xgb_model.evaluate(df_test_feat, target_col)
        results["XGBoost"] = {
            "metrics": metrics,
            "test_preds": pd.Series(xgb_preds.values, index=df_test_feat.index),
            "test_actual": pd.Series(df_test_feat[target_col].values, index=df_test_feat.index),
            "model": xgb_model,
        }
        xgb_model.save()
    except Exception as exc:
        logger.error("XGBoost failed: %s", exc)

    # LightGBM
    try:
        lgb_model = GradientBoostingModel(backend="lightgbm")
        lgb_model.train(df_train_feat, target_col)
        lgb_preds = lgb_model.predict(df_test_feat)
        metrics = lgb_model.evaluate(df_test_feat, target_col)
        results["LightGBM"] = {
            "metrics": metrics,
            "test_preds": pd.Series(lgb_preds.values, index=df_test_feat.index),
            "test_actual": pd.Series(df_test_feat[target_col].values, index=df_test_feat.index),
            "model": lgb_model,
        }
        lgb_model.save()
    except Exception as exc:
        logger.error("LightGBM failed: %s", exc)

    # LSTM
    try:
        feature_cols_lstm = [
            c for c in feature_df.select_dtypes(include=[float, int]).columns
            if c != "year" and "anomaly" not in c
        ]
        if target_col in feature_cols_lstm:
            feature_cols_lstm = [target_col] + [c for c in feature_cols_lstm if c != target_col]

        arr = feature_df[feature_cols_lstm].fillna(0).values
        n_train_lstm = int(len(arr) * (1 - TEST_SIZE))
        arr_train = arr[:n_train_lstm]
        arr_test = arr[n_train_lstm - LSTM_PARAMS["sequence_length"]:]

        lstm = LSTMNet(input_size=arr_train.shape[1])
        lstm.train(arr_train)
        lstm_preds_raw = lstm.predict(arr_test)
        actual_lstm = arr[n_train_lstm:, 0]
        min_len = min(len(lstm_preds_raw), len(actual_lstm))
        metrics = compute_metrics(actual_lstm[:min_len], lstm_preds_raw[:min_len], "LSTM")
        results["LSTM"] = {
            "metrics": metrics,
            "test_preds": pd.Series(lstm_preds_raw[:min_len]),
            "test_actual": pd.Series(actual_lstm[:min_len]),
        }
        lstm.save()
    except Exception as exc:
        logger.error("LSTM failed: %s", exc)

    logger.info("=== All models trained ===")

    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    for name, res in results.items():
        row = {"Model": name}
        row.update(res.get("metrics", {}))
        metrics_rows.append(row)
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(FORECASTS_DIR / "model_metrics.csv", index=False)
        logger.info("Metrics saved -> %s", FORECASTS_DIR / "model_metrics.csv")

    return results
