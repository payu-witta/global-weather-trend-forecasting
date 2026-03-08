"""
Feature importance analysis using three complementary techniques:
  1. Tree-model built-in feature importance (XGBoost / LightGBM)
  2. Permutation importance (sklearn)
  3. SHAP values

Results are visualized and saved to outputs/figures/.
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
from config import FIGURES_DIR, RANDOM_SEED, TARGET_VARIABLE

logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)

_DPI = 150


def _save(fig, path):
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved -> %s", path)


def tree_feature_importance(
    model,
    feature_cols,
    top_n=25,
    dominant_n=4,
    dominant_save_path=None,
    logscale_save_path=None,
):
    """Extract built-in feature importance and save two charts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    top_all = importance.head(top_n)
    top_dom = importance.head(dominant_n)

    if dominant_save_path is None:
        dominant_save_path = FIGURES_DIR / "xgboost_tree_importance_dominant.png"
    if logscale_save_path is None:
        logscale_save_path = FIGURES_DIR / "xgboost_tree_importance_logscale.png"

    fig, ax = plt.subplots(figsize=(10, max(4, dominant_n * 0.55)))
    colors = plt.cm.Blues_r(np.linspace(0.3, 0.8, dominant_n))
    top_dom[::-1].plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title(
        f"Top {dominant_n} Feature Importances - Tree Model (Linear Scale)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Importance Score (Gain)")
    ax.tick_params(axis="y", labelsize=10)
    for bar, val in zip(ax.patches, top_dom[::-1].values):
        ax.text(
            val + max(top_dom) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9,
        )
    plt.tight_layout()
    _save(fig, dominant_save_path)

    vals = top_all.clip(lower=1e-8)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    colors_all = ["steelblue" if i < dominant_n else "lightskyblue" for i in range(top_n)]
    vals[::-1].plot(kind="barh", ax=ax, color=colors_all[::-1], edgecolor="white")
    ax.set_xscale("log")
    ax.set_title(
        f"Top {top_n} Feature Importances - Tree Model (Log Scale)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Importance Score (Gain) - log scale")
    ax.tick_params(axis="y", labelsize=9)
    ax.axvline(
        vals.iloc[dominant_n - 1], color="red", linewidth=1.2,
        linestyle="--", alpha=0.7, label=f"Top-{dominant_n} threshold",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, logscale_save_path)

    return top_all


def permutation_importance(
    model,
    X_test,
    y_test,
    feature_cols,
    n_repeats=10,
    top_n=25,
    dominant_n=8,
    dominant_save_path=None,
    logscale_save_path=None,
):
    """Compute sklearn permutation importance and save two charts."""
    from sklearn.inspection import permutation_importance as sklearn_perm

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Computing permutation importance (n_repeats=%d) ...", n_repeats)

    result = sklearn_perm(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=RANDOM_SEED, n_jobs=-1,
    )
    imp = pd.Series(result.importances_mean, index=feature_cols)
    imp = imp.sort_values(ascending=False)
    top_all = imp.head(top_n)
    top_dom = imp.head(dominant_n)

    if dominant_save_path is None:
        dominant_save_path = FIGURES_DIR / "xgboost_permutation_importance_dominant.png"
    if logscale_save_path is None:
        logscale_save_path = FIGURES_DIR / "xgboost_permutation_importance_logscale.png"

    fig, ax = plt.subplots(figsize=(10, max(4, dominant_n * 0.55)))
    colors = plt.cm.Oranges_r(np.linspace(0.3, 0.8, dominant_n))
    top_dom[::-1].plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title(
        f"Top {dominant_n} Permutation Importances (Linear Scale)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Mean Decrease in Accuracy")
    ax.tick_params(axis="y", labelsize=10)
    for bar, val in zip(ax.patches, top_dom[::-1].values):
        ax.text(
            val + max(top_dom) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9,
        )
    plt.tight_layout()
    _save(fig, dominant_save_path)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    colors_all = ["darkorange" if i < dominant_n else "moccasin" for i in range(top_n)]
    top_all[::-1].plot(kind="barh", ax=ax, color=colors_all[::-1], edgecolor="white")
    linthresh = max(abs(top_all).min(), 1e-4)
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_title(
        f"Top {top_n} Permutation Importances (Symlog Scale)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Mean Decrease in Accuracy - symlog scale")
    ax.tick_params(axis="y", labelsize=9)
    ax.axvline(
        top_all.iloc[dominant_n - 1], color="red", linewidth=1.2,
        linestyle="--", alpha=0.7, label=f"Top-{dominant_n} threshold",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, logscale_save_path)

    return top_all


def shap_analysis(model, X_train, X_test, feature_cols, top_n=20, dominant_n=4, save_dir=None):
    """
    Compute SHAP values and generate beeswarm, bar, and dependence plots.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed. Skipping SHAP analysis.")
        return None

    logger.info("Computing SHAP values ...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if save_dir is None:
        save_dir = FIGURES_DIR

    explainer = shap.TreeExplainer(model)
    X_test_df = (
        pd.DataFrame(X_test, columns=feature_cols)
        if not isinstance(X_test, pd.DataFrame)
        else X_test
    )
    shap_values = explainer.shap_values(X_test_df)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, max_display=top_n, show=False, plot_type="dot")
    plt.title("SHAP Value Summary (Beeswarm)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_dir / "shap_beeswarm.png")

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.Series(mean_abs, index=feature_cols).sort_values(ascending=False)
    top_all = shap_importance.head(top_n)
    top_dom = shap_importance.head(dominant_n)

    fig, ax = plt.subplots(figsize=(10, max(4, dominant_n * 0.55)))
    colors = plt.cm.Greens_r(np.linspace(0.3, 0.8, dominant_n))
    top_dom[::-1].plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title(
        f"Top {dominant_n} SHAP Feature Importance - Mean |SHAP| (Linear Scale)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Mean |SHAP| Value")
    ax.tick_params(axis="y", labelsize=10)
    for bar, val in zip(ax.patches, top_dom[::-1].values):
        ax.text(
            val + max(top_dom) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9,
        )
    plt.tight_layout()
    _save(fig, save_dir / "shap_bar_dominant.png")

    vals = top_all.clip(lower=1e-8)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    colors_all = ["mediumseagreen" if i < dominant_n else "lightgreen" for i in range(top_n)]
    vals[::-1].plot(kind="barh", ax=ax, color=colors_all[::-1], edgecolor="white")
    ax.set_xscale("log")
    ax.set_title(
        f"Top {top_n} SHAP Feature Importance - Mean |SHAP| (Log Scale)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Mean |SHAP| Value - log scale")
    ax.tick_params(axis="y", labelsize=9)
    ax.axvline(
        vals.iloc[dominant_n - 1], color="red", linewidth=1.2,
        linestyle="--", alpha=0.7, label=f"Top-{dominant_n} threshold",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, save_dir / "shap_bar_logscale.png")

    top_feature = feature_cols[np.argmax(mean_abs)]
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(top_feature, shap_values, X_test_df, ax=ax, show=False)
    plt.title(f"SHAP Dependence: {top_feature}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_dir / "shap_dependence_top.png")

    logger.info("SHAP analysis complete. Top feature: '%s'", top_feature)
    return shap_importance, shap_values


def run_feature_importance(model_results, feature_df, target_col=TARGET_VARIABLE):
    """Run all three feature importance analyses for the XGBoost model."""
    logger.info("=== Feature importance analysis started ===")

    for model_name in ("XGBoost", "LightGBM"):
        if model_name not in model_results:
            continue

        gb_model_wrapper = model_results[model_name].get("model")
        if gb_model_wrapper is None:
            continue

        feature_cols = gb_model_wrapper.feature_cols
        fitted_model = gb_model_wrapper.model

        from config import TEST_SIZE

        n = len(feature_df)
        split = int(n * (1 - TEST_SIZE))
        df_train = feature_df.iloc[:split]
        df_test = feature_df.iloc[split:]

        X_train = df_train[feature_cols].fillna(0).values
        X_test = df_test[feature_cols].fillna(0).values
        y_test = df_test[target_col].values

        stem = model_name.lower()

        tree_feature_importance(
            fitted_model, feature_cols,
            dominant_save_path=FIGURES_DIR / f"{stem}_tree_importance_dominant.png",
            logscale_save_path=FIGURES_DIR / f"{stem}_tree_importance_logscale.png",
        )

        permutation_importance(
            fitted_model, X_test, y_test, feature_cols,
            dominant_save_path=FIGURES_DIR / f"{stem}_permutation_importance_dominant.png",
            logscale_save_path=FIGURES_DIR / f"{stem}_permutation_importance_logscale.png",
        )

        if model_name == "XGBoost":
            shap_analysis(fitted_model, X_train, X_test, feature_cols)

        logger.info("%s feature importance analysis complete.", model_name)
        break

    logger.info("=== Feature importance analysis done ===")
