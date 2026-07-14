"""Pydantic request and response schemas for the REST API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Request body for creating a user account."""

    email: EmailStr
    password: str = Field(min_length=8)


class TokenResponse(BaseModel):
    """JWT token response for authenticated clients."""

    access_token: str
    token_type: str = "bearer"


class UploadResponse(BaseModel):
    """Response returned after a satellite image upload."""

    job_id: str
    status: str


class PredictRequest(BaseModel):
    """Request body for cloud-removal inference."""

    job_id: str


class PredictResponse(BaseModel):
    """Response returned when inference completes."""

    job_id: str
    status: str
    cloud_mask_url: str | None
    download_url: str | None
    metrics: dict[str, float]
    confidence_score: float | None


class JobResponse(BaseModel):
    """Serialized inference job for history screens."""

    id: str
    status: str
    input_uri: str
    mask_uri: str | None
    output_uri: str | None
    metrics: dict[str, float]
    confidence_score: float | None
    created_at: datetime
    updated_at: datetime


class PaginatedJobsResponse(BaseModel):
    """Paginated list of inference jobs."""

    items: list[JobResponse]
    total: int
    offset: int
    limit: int
