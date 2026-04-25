from __future__ import annotations

from fastapi import APIRouter, Depends

from womens_health_ai.core.bootstrap import get_prediction_service
from womens_health_ai.schemas.patient import PatientFeatures
from womens_health_ai.schemas.responses import HealthResponse, PredictionResponse
from womens_health_ai.services.prediction_service import PredictionService


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_ready=True,
        champion_model=prediction_service.model_name,
    )


@router.get("/model-info")
def model_info(
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> dict:
    return prediction_service.metadata


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PatientFeatures,
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    return PredictionResponse(**prediction_service.predict(payload.model_dump()))

