from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from womens_health_ai.data.dataset import load_dataset
from womens_health_ai.features.engineering import RAW_FEATURES
from womens_health_ai.settings import paths, settings
from womens_health_ai.utils.io import ensure_directories


sns.set_theme(style="whitegrid")


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_eda_reports(dataset: pd.DataFrame | None = None) -> dict[str, str]:
    ensure_directories(paths.reports_dir, paths.reports_eda_dir)

    dataset = dataset if dataset is not None else load_dataset()
    feature_frame = dataset[RAW_FEATURES].copy()

    descriptive_stats = feature_frame.describe().transpose()
    descriptive_stats.to_csv(paths.reports_eda_dir / "descriptive_statistics.csv")

    missing_summary = feature_frame.isna().sum().to_frame("missing_values")
    missing_summary["missing_ratio"] = missing_summary["missing_values"] / len(feature_frame)
    missing_summary.to_csv(paths.reports_eda_dir / "missing_value_summary.csv")

    correlation_matrix = feature_frame.corr(numeric_only=True)
    correlation_matrix.to_csv(paths.reports_eda_dir / "correlation_matrix.csv")

    plt.figure(figsize=(7, 5))
    sns.countplot(
        data=dataset,
        x=settings.target_label_column,
        hue=settings.target_label_column,
        legend=False,
        palette=["#2b6cb0", "#c53030"],
    )
    plt.title("Diagnosis Class Distribution")
    plt.xlabel("Diagnosis")
    plt.ylabel("Cases")
    _save_figure(paths.reports_eda_dir / "class_distribution.png")

    feature_subset = [
        "mean_radius",
        "mean_texture",
        "mean_perimeter",
        "mean_area",
        "mean_compactness",
        "mean_concavity",
        "worst_radius",
        "worst_perimeter",
        "worst_area",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for axis, feature in zip(axes.flat, feature_subset, strict=False):
        sns.histplot(
            data=dataset,
            x=feature,
            hue=settings.target_label_column,
            kde=True,
            stat="density",
            common_norm=False,
            ax=axis,
            palette=["#2b6cb0", "#c53030"],
        )
        axis.set_title(feature.replace("_", " ").title())
    _save_figure(paths.reports_eda_dir / "feature_distributions.png")

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        linewidths=0.1,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix")
    _save_figure(paths.reports_eda_dir / "correlation_matrix.png")

    boxplot_features = [
        "mean_radius",
        "mean_texture",
        "mean_perimeter",
        "mean_area",
        "worst_radius",
        "worst_area",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for axis, feature in zip(axes.flat, boxplot_features, strict=False):
        sns.boxplot(
            data=dataset,
            x=settings.target_label_column,
            y=feature,
            hue=settings.target_label_column,
            legend=False,
            dodge=False,
            ax=axis,
            palette=["#2b6cb0", "#c53030"],
        )
        axis.set_title(feature.replace("_", " ").title())
        axis.set_xlabel("Diagnosis")
    _save_figure(paths.reports_eda_dir / "diagnosis_boxplots.png")

    return {
        "descriptive_statistics": str(paths.reports_eda_dir / "descriptive_statistics.csv"),
        "missing_value_summary": str(paths.reports_eda_dir / "missing_value_summary.csv"),
        "correlation_matrix_csv": str(paths.reports_eda_dir / "correlation_matrix.csv"),
        "class_distribution_plot": str(paths.reports_eda_dir / "class_distribution.png"),
        "feature_distribution_plot": str(paths.reports_eda_dir / "feature_distributions.png"),
        "correlation_heatmap": str(paths.reports_eda_dir / "correlation_matrix.png"),
        "diagnosis_boxplots": str(paths.reports_eda_dir / "diagnosis_boxplots.png"),
    }


def main() -> None:
    report_paths = generate_eda_reports()
    print("EDA reports generated:")
    for name, path in report_paths.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
