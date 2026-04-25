from __future__ import annotations

from functools import lru_cache

from womens_health_ai.modeling.training import ensure_training_artifacts
from womens_health_ai.services.prediction_service import PredictionService


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    ensure_training_artifacts(force=False)
    return PredictionService()

