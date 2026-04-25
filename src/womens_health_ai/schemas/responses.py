from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    champion_model: str


class ContributorResponse(BaseModel):
    feature: str
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    predicted_class: str
    malignant_probability: float
    benign_probability: float
    risk_level: str
    model_name: str
    threshold: float
    missing_features_imputed: list[str]
    top_contributors: list[ContributorResponse]
    disclaimer: str


class ModelInfoResponse(BaseModel):
    payload: dict[str, Any]

