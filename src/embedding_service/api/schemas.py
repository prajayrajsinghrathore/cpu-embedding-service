from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class EmbeddingMetadata(BaseModel):
    project_id: Optional[str] = Field(default=None, max_length=256)
    correlation_id: Optional[str] = Field(default=None, max_length=256)


class EmbeddingRequest(BaseModel):
    model: Optional[str] = Field(default=None, max_length=256)
    texts: List[str] = Field(..., min_length=1)
    normalize: Optional[bool] = Field(default=None)
    truncate: Optional[bool] = Field(default=None)
    metadata: Optional[EmbeddingMetadata] = Field(default=None)

    @field_validator("texts")
    @classmethod
    def validate_texts_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("texts list cannot be empty")
        return v


class UsageStats(BaseModel):
    texts: int = Field(..., ge=0)
    chars: int = Field(..., ge=0)
    ms: int = Field(..., ge=0)


class EmbeddingResponse(BaseModel):
    model: str
    dim: int
    embeddings: List[List[float]]
    usage: UsageStats


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    config_valid: bool
    details: Optional[Dict[str, str]] = Field(default=None)


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
