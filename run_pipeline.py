"""
run_pipeline.py — End-to-end orchestration for the Global Weather Forecasting project.

Usage:
    python run_pipeline.py                    # full pipeline
    python run_pipeline.py --skip-models      # skip slow model training
    python run_pipeline.py --data-dir /path   # custom data directory

All outputs are written to outputs/ (figures, forecasts, reports).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import (
    FIGURES_DIR,
    FORECASTS_DIR,
    LOG_FORMAT,
    LOG_LEVEL,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TARGET_VARIABLE,
)


def setup_logging():
    import io

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL))
    root.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

    # Force UTF-8 on the stream handler so Unicode chars don't crash on Windows
    # terminals running non-UTF-8 code pages (e.g. cp874).
    if hasattr(sys.stdout, "buffer"):
        utf8_stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        sh = logging.StreamHandler(utf8_stream)
    else:
        sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    fh = logging.FileHandler(ROOT / "pipeline.log", mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)


logger = logging.getLogger("pipeline")


def init_directories():
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, FORECASTS_DIR, REPORTS_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Output directories initialised.")


def stage_load(args):
    from data_loader import load_dataset, print_summary, validate_schema

    logger.info("== Stage 1: Data Loading ==")
    t0 = time.time()
    df = load_dataset(data_dir=args.data_dir, try_kaggle=not args.no_kaggle)
    validate_schema(df)
    print_summary(df)
    logger.info("Data loading done in %.1fs", time.time() - t0)
    return df


def stage_preprocess(df):
    from preprocessing import run_preprocessing

    logger.info("== Stage 2: Preprocessing ==")
    t0 = time.time()
    df_clean, daily_global, scaler = run_preprocessing(df, save=True)
    logger.info("Preprocessing done in %.1fs", time.time() - t0)
    return df_clean, daily_global, scaler


def stage_feature_engineering(daily_global):
    from feature_engineering import get_feature_columns, run_feature_engineering

    logger.info("== Stage 3: Feature Engineering ==")
    t0 = time.time()
    feature_df = run_feature_engineering(daily_global)
    feature_cols = get_feature_columns(feature_df)
    logger.info(
        "Feature engineering done in %.1fs. %d feature columns.",
        time.time() - t0,
        len(feature_cols),
    )
    return feature_df


def stage_eda(daily_global, df_clean, regional_df):
    from visualization import (
        plot_air_quality_correlations,
        plot_correlation_heatmap,
        plot_distributions,
        plot_global_warming_trend,
        plot_regional_comparison,
        plot_seasonal_decomposition,
        plot_seasonal_patterns,
        plot_time_series,
    )

    logger.info("== Stage 4: Exploratory Data Analysis ==")
    t0 = time.time()

    plot_distributions(daily_global)
    plot_correlation_heatmap(daily_global)
    plot_time_series(daily_global)
    plot_seasonal_decomposition(daily_global)
    plot_seasonal_patterns(daily_global)
    plot_global_warming_trend(daily_global)

    if df_clean is not None:
        plot_air_quality_correlations(df_clean)

    if regional_df is not None and not regional_df.empty:
        plot_regional_comparison(regional_df)

    logger.info("EDA done in %.1fs", time.time() - t0)


def stage_anomaly(daily_global, df_clean):
    from anomaly_detection import describe_anomalies, run_anomaly_detection
    from visualization import plot_anomalies

    logger.info("== Stage 5: Anomaly Detection ==")
    t0 = time.time()

    df_annotated, summary = run_anomaly_detection(daily_global)
    plot_anomalies(df_annotated)

    anomaly_events = describe_anomalies(df_annotated)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    anomaly_events.to_csv(REPORTS_DIR / "anomaly_events.csv", index=False)
    logger.info("Anomaly report saved. %d events detected.", len(anomaly_events))
    logger.info("Anomaly detection done in %.1fs", time.time() - t0)
    return df_annotated, summary


def stage_forecasting(daily_global, feature_df, skip=False):
    if skip:
        logger.info("== Stage 6: Forecasting -- SKIPPED ==")
        return {}

    from forecasting_models import run_all_models

    logger.info("== Stage 6: Forecasting Models ==")
    t0 = time.time()
    results = run_all_models(daily_global, feature_df)
    logger.info("Model training done in %.1fs", time.time() - t0)
    return results


def stage_ensemble(model_results):
    if not model_results:
        logger.info("== Stage 7: Ensemble -- SKIPPED (no models) ==")
        return {}, None

    from ensemble_model import run_ensemble, summarize_ensemble
    from visualization import plot_ensemble_comparison, plot_forecast_comparison

    logger.info("== Stage 7: Ensemble Model ==")
    t0 = time.time()
    ensemble_results, comparison_df = run_ensemble(model_results)
    summarize_ensemble(comparison_df)
    plot_forecast_comparison(model_results)
    plot_ensemble_comparison(comparison_df)
    logger.info("Ensemble done in %.1fs", time.time() - t0)
    return ensemble_results, comparison_df


def stage_feature_importance(model_results, feature_df, skip=False):
    if skip or not model_results:
        logger.info("== Stage 8: Feature Importance -- SKIPPED ==")
        return

    from feature_importance import run_feature_importance

    logger.info("== Stage 8: Feature Importance ==")
    t0 = time.time()
    run_feature_importance(model_results, feature_df)
    logger.info("Feature importance done in %.1fs", time.time() - t0)


def _stage_data_audit(df_raw):
    from data_audit import run_data_audit

    logger.info("== Stage 0: Data Audit ==")
    t0 = time.time()
    run_data_audit(df_raw, save=True)
    logger.info("Data audit done in %.1fs", time.time() - t0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Global Weather Forecasting Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default=None, help="Path to directory containing raw CSV")
    parser.add_argument("--no-kaggle", action="store_true", help="Do not attempt Kaggle download")
    parser.add_argument("--skip-models", action="store_true", help="Skip model training (EDA only)")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  Global Weather Trend Forecasting Pipeline")
    logger.info("  PM Accelerator Technical Assessment")
    logger.info("=" * 60)

    t_start = time.time()
    init_directories()

    df_raw = stage_load(args)

    _stage_data_audit(df_raw)

    df_clean, daily_global, scaler = stage_preprocess(df_raw)
    feature_df = stage_feature_engineering(daily_global)

    stage_eda(daily_global, df_clean, regional_df=None)

    df_annotated, anomaly_summary = stage_anomaly(daily_global, df_clean)

    model_results = stage_forecasting(daily_global, feature_df, skip=args.skip_models)

    ensemble_results, comparison_df = stage_ensemble(model_results)

    stage_feature_importance(model_results, feature_df, skip=args.skip_models or args.skip_shap)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("  Pipeline complete in %.1f seconds", elapsed)
    logger.info("  Figures -> %s", FIGURES_DIR)
    logger.info("  Forecasts -> %s", FORECASTS_DIR)
    logger.info("  Reports -> %s", REPORTS_DIR)
    logger.info("  Models -> %s", MODELS_DIR)
    logger.info("=" * 60)

    if comparison_df is not None and not comparison_df.empty:
        logger.info("\nModel Comparison:\n%s", comparison_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
