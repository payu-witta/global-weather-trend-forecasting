"""
Ensemble model combining predictions from multiple base forecasters.

Strategies implemented:
  1. Simple average
  2. Weighted average (inverse-RMSE weights)
  3. Stacking with Ridge regression as meta-learner

The ensemble is evaluated against individual models on the held-out test set.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))
from config import FORECASTS_DIR, MODELS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)


def _metrics(y_true, y_pred, name="Ensemble"):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    logger.info("%s -> MAE=%.4f | RMSE=%.4f | MAPE=%.2f%%", name, mae, rmse, mape)
    return {"Model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}


def simple_average(predictions_dict):
    """Average predictions from all models."""
    arrays = [np.asarray(v) for v in predictions_dict.values()]
    min_len = min(len(a) for a in arrays)
    stacked = np.vstack([a[:min_len] for a in arrays])
    return stacked.mean(axis=0)


def weighted_average(predictions_dict, metrics_dict):
    """Weight each model by 1/RMSE so that more accurate models get higher weight."""
    weights = {}
    for name in predictions_dict:
        rmse = metrics_dict.get(name, {}).get("RMSE", None)
        if rmse and rmse > 0:
            weights[name] = 1.0 / rmse
        else:
            weights[name] = 1.0

    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    arrays = []
    for name, arr in predictions_dict.items():
        arrays.append(np.asarray(arr) * weights[name])

    min_len = min(len(a) for a in arrays)
    combined = sum(a[:min_len] for a in arrays)

    logger.info("Weighted ensemble weights: %s", {k: f"{v:.3f}" for k, v in weights.items()})
    return combined, weights


class StackingEnsemble:
    """
    Train a Ridge regression meta-learner on out-of-fold base predictions.
    Splits test set in half: first half fits meta-learner, second half evaluates.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.meta = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        self.base_model_names = None

    def fit(self, predictions_dict, y_true):
        y_true = np.asarray(y_true)
        min_len = min(len(np.asarray(v)) for v in predictions_dict.values())
        y_true = y_true[:min_len]

        X_meta = np.column_stack([np.asarray(v)[:min_len] for v in predictions_dict.values()])
        self.base_model_names = list(predictions_dict.keys())

        split = max(1, min_len // 2)
        self.meta.fit(X_meta[:split], y_true[:split])
        logger.info("Stacking meta-learner fitted on %d samples.", split)
        return self

    def predict(self, predictions_dict):
        min_len = min(len(np.asarray(v)) for v in predictions_dict.values())
        X_meta = np.column_stack([np.asarray(v)[:min_len] for v in predictions_dict.values()])
        return self.meta.predict(X_meta)

    def coef_summary(self):
        if self.base_model_names is None:
            return {}
        return dict(zip(self.base_model_names, self.meta.coef_))


def run_ensemble(model_results, target_col="temperature_celsius"):
    """
    Build and evaluate all ensemble strategies using test-set predictions.

    Returns
    -------
    ensemble_results : dict
    comparison_df : pd.DataFrame
    """
    logger.info("=== Building ensemble models ===")

    predictions = {}
    actuals = {}

    for name, res in model_results.items():
        if "test_preds" in res and "test_actual" in res:
            preds = np.asarray(res["test_preds"])
            acts = np.asarray(res["test_actual"])
            min_len = min(len(preds), len(acts))
            predictions[name] = preds[:min_len]
            actuals[name] = acts[:min_len]

    if len(predictions) < 2:
        logger.warning("Need at least 2 models for ensemble. Skipping.")
        return {}, pd.DataFrame()

    min_len = min(len(v) for v in predictions.values())
    predictions = {k: v[:min_len] for k, v in predictions.items()}
    y_true = np.vstack([v[:min_len] for v in actuals.values()]).mean(axis=0)

    individual_metrics = {name: res.get("metrics", {}) for name, res in model_results.items()}
    ensemble_results = {}

    avg_preds = simple_average(predictions)
    m_avg = _metrics(y_true, avg_preds, "Ensemble (Simple Avg)")
    ensemble_results["Simple Average"] = {"metrics": m_avg, "predictions": avg_preds}

    weighted_preds, weights = weighted_average(predictions, individual_metrics)
    m_w = _metrics(y_true, weighted_preds, "Ensemble (Weighted Avg)")
    ensemble_results["Weighted Average"] = {
        "metrics": m_w, "predictions": weighted_preds, "weights": weights,
    }

    try:
        stacker = StackingEnsemble()
        stacker.fit(predictions, y_true)
        stacked_preds = stacker.predict(predictions)
        m_stack = _metrics(y_true, stacked_preds, "Ensemble (Stacking)")
        ensemble_results["Stacking"] = {
            "metrics": m_stack, "predictions": stacked_preds,
            "meta_coefs": stacker.coef_summary(),
        }
    except Exception as exc:
        logger.error("Stacking ensemble failed: %s", exc)

    rows = []
    for name, res in model_results.items():
        row = {"Model": name, "Type": "Individual"}
        row.update(res.get("metrics", {}))
        rows.append(row)
    for name, res in ensemble_results.items():
        row = {"Model": name, "Type": "Ensemble"}
        row.update(res.get("metrics", {}))
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    if "RMSE" in comparison_df.columns:
        comparison_df = comparison_df.sort_values("RMSE").reset_index(drop=True)

    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(FORECASTS_DIR / "ensemble_comparison.csv", index=False)
    logger.info("Ensemble comparison saved -> %s", FORECASTS_DIR / "ensemble_comparison.csv")
    logger.info("=== Ensemble evaluation complete ===")

    return ensemble_results, comparison_df


def summarize_ensemble(comparison_df):
    """Log a formatted summary of the model comparison table."""
    logger.info("\n%s", comparison_df.to_string(index=False))
    if "RMSE" in comparison_df.columns:
        best = comparison_df.iloc[0]
        logger.info("Best model: %s (RMSE=%.4f)", best["Model"], best["RMSE"])
