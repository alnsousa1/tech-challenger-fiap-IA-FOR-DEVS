from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import seaborn as sns

from womens_health_ai.settings import paths
from womens_health_ai.utils.io import ensure_directories


sns.set_theme(style="whitegrid")


def _to_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def clean_feature_name(name: str) -> str:
    cleaned = name.replace("num__", "").replace("cat__", "")
    return cleaned.replace("onehot__", "")


def get_transformed_data(pipeline: Any, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = _to_dense(preprocessor.transform(features))
    feature_names = [clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    return transformed, feature_names


def extract_feature_importance(pipeline: Any) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    feature_names = [
        clean_feature_name(name)
        for name in pipeline.named_steps["preprocessor"].get_feature_names_out()
    ]

    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        raise ValueError("Unsupported model for feature importance extraction.")

    importance_frame = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance_frame


def save_feature_importance_plot(
    model_name: str,
    importance_frame: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, str]:
    output_dir = output_dir or paths.reports_explainability_dir
    ensure_directories(output_dir)

    csv_path = output_dir / f"{model_name}_feature_importance.csv"
    plot_path = output_dir / f"{model_name}_feature_importance.png"

    importance_frame.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    top_features = importance_frame.head(15)
    sns.barplot(
        data=top_features,
        x="importance",
        y="feature",
        hue="feature",
        legend=False,
        palette="crest",
    )
    plt.title(f"{model_name.replace('_', ' ').title()} Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {"csv": str(csv_path), "plot": str(plot_path)}


def select_positive_class_explanation(
    explanation: shap.Explanation,
    sample_matrix: np.ndarray,
    feature_names: list[str],
) -> shap.Explanation:
    values = explanation.values
    base_values = explanation.base_values

    if values.ndim == 3:
        selected_values = values[:, :, 1]
        if np.asarray(base_values).ndim == 2:
            selected_base_values = base_values[:, 1]
        else:
            selected_base_values = np.repeat(np.asarray(base_values)[1], sample_matrix.shape[0])
        return shap.Explanation(
            values=selected_values,
            base_values=selected_base_values,
            data=sample_matrix,
            feature_names=feature_names,
        )

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=sample_matrix,
        feature_names=feature_names,
    )


def generate_shap_report(
    model_name: str,
    pipeline: Any,
    background_frame: pd.DataFrame,
    sample_frame: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or paths.reports_explainability_dir
    ensure_directories(output_dir)

    background_matrix, feature_names = get_transformed_data(pipeline, background_frame)
    sample_matrix, _ = get_transformed_data(pipeline, sample_frame)
    model = pipeline.named_steps["model"]

    explainer = shap.Explainer(model, background_matrix, feature_names=feature_names)
    raw_explanation = explainer(sample_matrix)
    positive_explanation = select_positive_class_explanation(
        raw_explanation,
        sample_matrix,
        feature_names,
    )

    summary_bar_path = output_dir / f"{model_name}_shap_summary_bar.png"
    beeswarm_path = output_dir / f"{model_name}_shap_beeswarm.png"
    waterfall_path = output_dir / f"{model_name}_shap_waterfall.png"

    shap.summary_plot(positive_explanation, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(summary_bar_path, dpi=200, bbox_inches="tight")
    plt.close()

    shap.summary_plot(positive_explanation, show=False)
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
    plt.close()

    shap.plots.waterfall(positive_explanation[0], show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "background_matrix": background_matrix,
        "feature_names": feature_names,
        "summary_bar_plot": str(summary_bar_path),
        "beeswarm_plot": str(beeswarm_path),
        "waterfall_plot": str(waterfall_path),
    }
