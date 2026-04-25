from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from womens_health_ai.data.dataset import load_dataset
from womens_health_ai.explainability.shap_analysis import (
    extract_feature_importance,
    generate_shap_report,
    save_feature_importance_plot,
)
from womens_health_ai.features.engineering import (
    CATEGORICAL_FEATURES,
    DERIVED_NUMERIC_FEATURES,
    FEATURE_GROUPS,
    RAW_FEATURES,
    engineer_features,
)
from womens_health_ai.reporting.eda import generate_eda_reports
from womens_health_ai.settings import paths, settings
from womens_health_ai.utils.io import ensure_directories, read_json, to_serializable, write_json


TARGET_LABELS = [settings.negative_class_label, settings.positive_class_label]

#build_preprocessor serve para criar um pipeline de pré-processamento que lida tanto com features numéricas quanto categóricas. Ele define transformações específicas para cada tipo de feature, como imputação e escalonamento para numéricas, e imputação seguida de codificação one-hot para categóricas. O resultado é um ColumnTransformer que pode ser integrado a pipelines de modelagem para garantir que os dados sejam adequadamente preparados antes do treinamento do modelo.
def build_preprocessor() -> ColumnTransformer:
    numeric_features = RAW_FEATURES + DERIVED_NUMERIC_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def candidate_models() -> dict[str, dict[str, Any]]:
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=settings.random_state,
            ),
            "params": {
                "model__C": [0.1, 1.0, 5.0],
            },
        },
        "decision_tree": {
            "estimator": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=settings.random_state,
            ),
            "params": {
                "model__max_depth": [3, 5, 7, None],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
    }


def _create_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def _evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=candidate_models()[model_name]["params"],
        scoring="recall",
        cv=StratifiedKFold(
            n_splits=settings.cv_folds,
            shuffle=True,
            random_state=settings.random_state,
        ),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    predictions = best_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=TARGET_LABELS,
        output_dict=True,
        zero_division=0,
    )

    print(f"\n=== {model_name.replace('_', ' ').title()} ===")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=TARGET_LABELS,
            zero_division=0,
        )
    )

    return {
        "pipeline": best_pipeline,
        "metrics": {
            "accuracy": accuracy,
            "recall": recall,
            "f1_score": f1,
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "best_cv_recall": float(search.best_score_),
            "best_params": search.best_params_,
        },
    }


def _select_champion(evaluations: dict[str, dict[str, Any]]) -> str:
    return max(
        evaluations,
        key=lambda name: (
            evaluations[name]["metrics"]["recall"],
            evaluations[name]["metrics"]["f1_score"],
            evaluations[name]["metrics"]["accuracy"],
        ),
    )


def _serialize_sample(frame: pd.DataFrame) -> dict[str, float | None]:
    row = frame.iloc[0].to_dict()
    return {key: (None if pd.isna(value) else float(value)) for key, value in row.items()}


def train_and_persist(force: bool = False) -> dict[str, Any]:
    ensure_directories(
        paths.artifacts_dir,
        paths.reports_dir,
        paths.reports_eda_dir,
        paths.reports_explainability_dir,
    )

    if not force and paths.bundle_path.exists() and paths.model_info_path.exists():
        return read_json(paths.model_info_path)

    dataset = load_dataset()
    eda_report_paths = generate_eda_reports(dataset)

    raw_features = dataset[RAW_FEATURES].copy()
    engineered_features = engineer_features(raw_features)
    target = dataset[settings.target_column].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        engineered_features,
        target,
        test_size=settings.test_size,
        stratify=target,
        random_state=settings.random_state,
    )

    evaluations: dict[str, dict[str, Any]] = {}
    for model_name, configuration in candidate_models().items():
        pipeline = _create_pipeline(configuration["estimator"])
        evaluations[model_name] = _evaluate_model(
            model_name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
        )

    champion_name = _select_champion(evaluations)
    champion_pipeline = evaluations[champion_name]["pipeline"]

    feature_importance_reports: dict[str, dict[str, str]] = {}
    importance_tables: dict[str, list[dict[str, Any]]] = {}
    for model_name, evaluation in evaluations.items():
        importance_frame = extract_feature_importance(evaluation["pipeline"])
        feature_importance_reports[model_name] = save_feature_importance_plot(
            model_name,
            importance_frame,
        )
        importance_tables[model_name] = importance_frame.head(20).to_dict(orient="records")

    background_frame = X_train.sample(
        n=min(settings.shap_background_size, len(X_train)),
        random_state=settings.random_state,
    )
    sample_frame = X_test.sample(
        n=min(settings.shap_sample_size, len(X_test)),
        random_state=settings.random_state,
    )
    shap_report = generate_shap_report(
        champion_name,
        champion_pipeline,
        background_frame,
        sample_frame,
    )

    champion_metrics = evaluations[champion_name]["metrics"]

    model_info = {
        "project_name": settings.project_name,
        "project_version": settings.project_version,
        "dataset": {
            "name": settings.dataset_name,
            "rows": int(len(dataset)),
            "columns": int(len(dataset.columns) - 2),
            "target_distribution": dataset[settings.target_label_column].value_counts().to_dict(),
        },
        "feature_groups": FEATURE_GROUPS,
        "engineered_features": {
            "numeric": DERIVED_NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "models": {
            model_name: evaluation["metrics"]
            for model_name, evaluation in evaluations.items()
        },
        "champion_model": {
            "name": champion_name,
            "selection_rule": "Highest malignant-class recall, then F1-score, then accuracy.",
            "metrics": champion_metrics,
        },
        "metric_rationale": settings.malignant_recall_rationale,
        "feature_importance_reports": feature_importance_reports,
        "feature_importance_top_features": importance_tables,
        "shap_reports": {
            key: value
            for key, value in shap_report.items()
            if key not in {"background_matrix", "feature_names"}
        },
        "eda_reports": eda_report_paths,
        "sample_payloads": {
            "malignant_example": _serialize_sample(
                raw_features[target == 1].head(1)
            ),
            "benign_example": _serialize_sample(
                raw_features[target == 0].head(1)
            ),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    bundle = {
        "metadata": to_serializable(model_info),
        "champion_model_name": champion_name,
        "models": {
            model_name: evaluation["pipeline"]
            for model_name, evaluation in evaluations.items()
        },
        "explainability": {
            "background_matrix": shap_report["background_matrix"],
            "feature_names": shap_report["feature_names"],
        },
    }

    joblib.dump(bundle, paths.bundle_path)
    write_json(paths.model_info_path, model_info)

    return model_info


def ensure_training_artifacts(force: bool = False) -> dict[str, Any]:
    return train_and_persist(force=force)


def load_model_bundle() -> dict[str, Any]:
    ensure_training_artifacts(force=False)
    return joblib.load(paths.bundle_path)


def main() -> None:
    model_info = train_and_persist(force=True)
    print("\nChampion model:")
    print(f"  - {model_info['champion_model']['name']}")
    print(
        f"  - recall={model_info['champion_model']['metrics']['recall']:.4f}, "
        f"f1={model_info['champion_model']['metrics']['f1_score']:.4f}, "
        f"accuracy={model_info['champion_model']['metrics']['accuracy']:.4f}"
    )
    print(f"Artifacts saved to: {paths.artifacts_dir}")


if __name__ == "__main__":
    main()
