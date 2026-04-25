from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer

from womens_health_ai.features.engineering import RAW_FEATURES, normalize_column_name
from womens_health_ai.settings import settings


def load_dataset() -> pd.DataFrame:
    dataset = load_breast_cancer(as_frame=True)

    features = dataset.data.rename(columns=normalize_column_name)
    features = features[RAW_FEATURES].copy()

    malignant_target = dataset.target.map({0: 1, 1: 0}).astype(int)

    features[settings.target_column] = malignant_target
    features[settings.target_label_column] = malignant_target.map(
        {
            0: settings.negative_class_label,
            1: settings.positive_class_label,
        }
    )

    return features


def load_features_and_target() -> tuple[pd.DataFrame, pd.Series]:
    dataset = load_dataset()
    return dataset[RAW_FEATURES].copy(), dataset[settings.target_column].copy()
