from __future__ import annotations

from dataclasses import dataclass

from womens_health_ai.settings import settings


@dataclass(frozen=True)
class ApiMetadata:
    title: str = settings.project_name
    version: str = settings.project_version
    description: str = (
        "FastAPI service for breast cancer risk classification, model evaluation, "
        "and explainability outputs in a women's healthcare context."
    )


api_metadata = ApiMetadata()

