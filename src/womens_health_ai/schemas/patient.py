from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean_radius: Optional[float] = Field(default=None)
    mean_texture: Optional[float] = Field(default=None)
    mean_perimeter: Optional[float] = Field(default=None)
    mean_area: Optional[float] = Field(default=None)
    mean_smoothness: Optional[float] = Field(default=None)
    mean_compactness: Optional[float] = Field(default=None)
    mean_concavity: Optional[float] = Field(default=None)
    mean_concave_points: Optional[float] = Field(default=None)
    mean_symmetry: Optional[float] = Field(default=None)
    mean_fractal_dimension: Optional[float] = Field(default=None)

    radius_error: Optional[float] = Field(default=None)
    texture_error: Optional[float] = Field(default=None)
    perimeter_error: Optional[float] = Field(default=None)
    area_error: Optional[float] = Field(default=None)
    smoothness_error: Optional[float] = Field(default=None)
    compactness_error: Optional[float] = Field(default=None)
    concavity_error: Optional[float] = Field(default=None)
    concave_points_error: Optional[float] = Field(default=None)
    symmetry_error: Optional[float] = Field(default=None)
    fractal_dimension_error: Optional[float] = Field(default=None)

    worst_radius: Optional[float] = Field(default=None)
    worst_texture: Optional[float] = Field(default=None)
    worst_perimeter: Optional[float] = Field(default=None)
    worst_area: Optional[float] = Field(default=None)
    worst_smoothness: Optional[float] = Field(default=None)
    worst_compactness: Optional[float] = Field(default=None)
    worst_concavity: Optional[float] = Field(default=None)
    worst_concave_points: Optional[float] = Field(default=None)
    worst_symmetry: Optional[float] = Field(default=None)
    worst_fractal_dimension: Optional[float] = Field(default=None)

    @model_validator(mode="after")
    def validate_payload_is_not_empty(self) -> "PatientFeatures":
        if not any(value is not None for value in self.model_dump().values()):
            raise ValueError("At least one measurement must be provided.")
        return self

