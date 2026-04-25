from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


FEATURE_GROUPS: Final[dict[str, list[str]]] = {
    "mean_measurements": [
        "mean_radius",
        "mean_texture",
        "mean_perimeter",
        "mean_area",
        "mean_smoothness",
        "mean_compactness",
        "mean_concavity",
        "mean_concave_points",
        "mean_symmetry",
        "mean_fractal_dimension",
    ],
    "measurement_error": [
        "radius_error",
        "texture_error",
        "perimeter_error",
        "area_error",
        "smoothness_error",
        "compactness_error",
        "concavity_error",
        "concave_points_error",
        "symmetry_error",
        "fractal_dimension_error",
    ],
    "worst_case_measurements": [
        "worst_radius",
        "worst_texture",
        "worst_perimeter",
        "worst_area",
        "worst_smoothness",
        "worst_compactness",
        "worst_concavity",
        "worst_concave_points",
        "worst_symmetry",
        "worst_fractal_dimension",
    ],
}

RAW_FEATURES: Final[list[str]] = [
    feature
    for group_features in FEATURE_GROUPS.values()
    for feature in group_features
]

DERIVED_NUMERIC_FEATURES: Final[list[str]] = [
    "area_perimeter_ratio",
    "compactness_to_concavity_ratio",
    "radius_error_pct",
    "worst_radius_delta",
]

CATEGORICAL_FEATURES: Final[list[str]] = [
    "radius_band",
    "texture_band",
    "concavity_band",
]

MODEL_FEATURES: Final[list[str]] = RAW_FEATURES + DERIVED_NUMERIC_FEATURES + CATEGORICAL_FEATURES


def normalize_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    engineered = frame.copy()

    engineered["area_perimeter_ratio"] = _safe_divide(
        engineered["mean_area"],
        engineered["mean_perimeter"],
    )
    engineered["compactness_to_concavity_ratio"] = _safe_divide(
        engineered["mean_compactness"],
        engineered["mean_concavity"],
    )
    engineered["radius_error_pct"] = _safe_divide(
        engineered["radius_error"],
        engineered["mean_radius"],
    )
    engineered["worst_radius_delta"] = engineered["worst_radius"] - engineered["mean_radius"]

    engineered["radius_band"] = pd.cut(
        engineered["mean_radius"],
        bins=[-np.inf, 12.0, 18.0, np.inf],
        labels=["small", "medium", "large"],
    )
    engineered["texture_band"] = pd.cut(
        engineered["mean_texture"],
        bins=[-np.inf, 15.0, 22.0, np.inf],
        labels=["low", "moderate", "high"],
    )
    engineered["concavity_band"] = pd.cut(
        engineered["mean_concavity"],
        bins=[-np.inf, 0.05, 0.15, np.inf],
        labels=["mild", "elevated", "severe"],
    )

    return engineered

