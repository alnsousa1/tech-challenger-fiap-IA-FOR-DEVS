from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
import shap

from womens_health_ai.explainability.shap_analysis import (
    get_transformed_data,
    select_positive_class_explanation,
)
from womens_health_ai.features.engineering import RAW_FEATURES, engineer_features
from womens_health_ai.modeling.training import load_model_bundle
from womens_health_ai.settings import settings


class PredictionService:
    def __init__(self) -> None:
        self.bundle = load_model_bundle()
        self.metadata: dict[str, Any] = self.bundle["metadata"]
        self.model_name: str = self.bundle["champion_model_name"]
        self.pipeline = self.bundle["models"][self.model_name]
        self.feature_names: list[str] = self.bundle["explainability"]["feature_names"]
        self.background_matrix = np.asarray(
            self.bundle["explainability"]["background_matrix"]
        )

    @cached_property
    def explainer(self) -> shap.Explainer:
        model = self.pipeline.named_steps["model"]
        return shap.Explainer(model, self.background_matrix, feature_names=self.feature_names)

    def _prepare_input_frame(self, payload: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
        raw_row = {feature: payload.get(feature) for feature in RAW_FEATURES}
        missing_features = [
            feature for feature, value in raw_row.items() if value is None
        ]

        frame = pd.DataFrame([raw_row]).apply(pd.to_numeric, errors="coerce")
        engineered = engineer_features(frame)
        return engineered, missing_features

    def _risk_level(self, malignant_probability: float) -> str:
        if malignant_probability >= 0.70:
            return "high"
        if malignant_probability >= 0.40:
            return "moderate"
        return "low"

    def _humanize_feature(self, feature_name: str) -> str:
        return feature_name.replace("_", " ").title()

    def _fallback_contributors(self, transformed_row: np.ndarray) -> list[dict[str, Any]]:
        model = self.pipeline.named_steps["model"]
        if hasattr(model, "coef_"):
            impacts = transformed_row[0] * model.coef_[0]
        else:
            impacts = transformed_row[0] * model.feature_importances_

        ranking = pd.DataFrame(
            {
                "feature": self.feature_names,
                "impact": impacts,
                "abs_impact": np.abs(impacts),
            }
        ).sort_values("abs_impact", ascending=False)

        return [
            {
                "feature": self._humanize_feature(row.feature),
                "impact": float(row.impact),
                "direction": "increases_risk" if row.impact >= 0 else "decreases_risk",
            }
            for row in ranking.head(5).itertuples()
        ]

    def _top_contributors(self, engineered_frame: pd.DataFrame) -> list[dict[str, Any]]:
        transformed_row, feature_names = get_transformed_data(self.pipeline, engineered_frame)

        try:
            explanation = self.explainer(transformed_row)
            positive_explanation = select_positive_class_explanation(
                explanation,
                transformed_row,
                feature_names,
            )
            impacts = positive_explanation.values[0]
            ranking = pd.DataFrame(
                {
                    "feature": feature_names,
                    "impact": impacts,
                    "abs_impact": np.abs(impacts),
                }
            ).sort_values("abs_impact", ascending=False)

            return [
                {
                    "feature": self._humanize_feature(row.feature),
                    "impact": float(row.impact),
                    "direction": (
                        "increases_risk" if row.impact >= 0 else "decreases_risk"
                    ),
                }
                for row in ranking.head(5).itertuples()
            ]
        except Exception:
            return self._fallback_contributors(transformed_row)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        engineered_frame, missing_features = self._prepare_input_frame(payload)

        malignant_probability = float(
            self.pipeline.predict_proba(engineered_frame)[0][1]
        )
        predicted_class = (
            settings.positive_class_label
            if malignant_probability >= settings.model_threshold
            else settings.negative_class_label
        )

        return {
            "predicted_class": predicted_class,
            "malignant_probability": malignant_probability,
            "benign_probability": 1.0 - malignant_probability,
            "risk_level": self._risk_level(malignant_probability),
            "model_name": self.model_name,
            "threshold": settings.model_threshold,
            "missing_features_imputed": missing_features,
            "top_contributors": self._top_contributors(engineered_frame),
            "disclaimer": (
                "This API supports risk screening and explainability analysis. "
                "It must not replace clinical judgement or diagnostic confirmation."
            ),
        }

