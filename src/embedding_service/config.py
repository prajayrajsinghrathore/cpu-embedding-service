from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, field_validator


class ServiceConfig(BaseModel):
    host: str = Field(default="0.0.0.0", min_length=1)
    port: int = Field(default=8008, ge=1, le=65535)


class EmbeddingsConfig(BaseModel):
    default_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        min_length=1
    )
    allow_model_override: bool = Field(default=False)
    normalize_default: bool = Field(default=True)
    truncate_default: bool = Field(default=True)
    batch_max_texts: int = Field(default=64, ge=1, le=1024)
    max_chars_per_text: int = Field(default=8000, ge=1, le=100000)
    request_timeout_seconds: int = Field(default=60, ge=1, le=600)

    @field_validator("default_model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not v or v.isspace():
            raise ValueError("default_model cannot be empty or whitespace")
        return v.strip()


class ObservabilityConfig(BaseModel):
    enabled: bool = Field(default=False)
    service_name: str = Field(default="embedding-service", min_length=1)
    exporter: str = Field(default="otlp", pattern="^(otlp|console)$")
    otlp_endpoint: str = Field(default="http://otel-collector:4317")
    sampling_ratio: float = Field(default=1.0, ge=0.0, le=1.0)


class SecurityConfig(BaseModel):
    allowed_origins: List[str] = Field(default=["*"])
    request_id_header: str = Field(default="X-Request-ID", min_length=1)

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("allowed_origins must contain at least one origin")
        return v


class AppConfig(BaseModel):
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)


def _apply_env_overrides(config_dict: dict) -> dict:
    env_mappings = {
        "EMBEDDING_SERVICE_HOST": ("service", "host"),
        "EMBEDDING_SERVICE_PORT": ("service", "port", int),
        "EMBEDDING_DEFAULT_MODEL": ("embeddings", "default_model"),
        "EMBEDDING_ALLOW_MODEL_OVERRIDE": ("embeddings", "allow_model_override", _parse_bool),
        "EMBEDDING_NORMALIZE_DEFAULT": ("embeddings", "normalize_default", _parse_bool),
        "EMBEDDING_TRUNCATE_DEFAULT": ("embeddings", "truncate_default", _parse_bool),
        "EMBEDDING_BATCH_MAX_TEXTS": ("embeddings", "batch_max_texts", int),
        "EMBEDDING_MAX_CHARS_PER_TEXT": ("embeddings", "max_chars_per_text", int),
        "EMBEDDING_REQUEST_TIMEOUT": ("embeddings", "request_timeout_seconds", int),
        "OTEL_ENABLED": ("observability", "enabled", _parse_bool),
        "OTEL_SERVICE_NAME": ("observability", "service_name"),
        "OTEL_EXPORTER": ("observability", "exporter"),
        "OTEL_ENDPOINT": ("observability", "otlp_endpoint"),
        "OTEL_SAMPLING_RATIO": ("observability", "sampling_ratio", float),
        "SECURITY_ALLOWED_ORIGINS": ("security", "allowed_origins", _parse_list),
        "SECURITY_REQUEST_ID_HEADER": ("security", "request_id_header"),
    }

    for env_var, mapping in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            section = mapping[0]
            key = mapping[1]
            converter = mapping[2] if len(mapping) > 2 else str

            if section not in config_dict:
                config_dict[section] = {}

            config_dict[section][key] = converter(value)

    return config_dict


def _parse_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes", "on")


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_config(config_path: str | None = None) -> AppConfig:
    config_dict: dict = {}

    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "config.json")

    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

    config_dict = _apply_env_overrides(config_dict)

    return AppConfig(**config_dict)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return load_config()


def validate_config(config: AppConfig) -> tuple[bool, str | None]:
    try:
        if config.embeddings.batch_max_texts < 1:
            return False, "batch_max_texts must be at least 1"

        if config.embeddings.max_chars_per_text < 1:
            return False, "max_chars_per_text must be at least 1"

        if not config.embeddings.default_model:
            return False, "default_model is required"

        return True, None
    except Exception as e:
        return False, str(e)
