from __future__ import annotations

import uuid
from typing import List

from fastapi import Request, Response

from embedding_service.config import EmbeddingsConfig, SecurityConfig


def generate_correlation_id() -> str:
    return str(uuid.uuid4())


def get_correlation_id(request: Request, header_name: str) -> str:
    correlation_id = request.headers.get(header_name)
    if correlation_id:
        return correlation_id[:256]
    return generate_correlation_id()


def set_correlation_id_header(
    response: Response,
    correlation_id: str,
    header_name: str
) -> None:
    response.headers[header_name] = correlation_id


class InputValidator:
    def __init__(self, config: EmbeddingsConfig) -> None:
        self._max_texts = config.batch_max_texts
        self._max_chars = config.max_chars_per_text

    def validate_texts(self, texts: List[str]) -> tuple[bool, str | None]:
        if not texts:
            return False, "texts list cannot be empty"

        if len(texts) > self._max_texts:
            return False, f"batch size {len(texts)} exceeds maximum {self._max_texts}"

        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                return False, f"text at index {idx} is not a string"

            if len(text) > self._max_chars:
                return False, f"text at index {idx} exceeds maximum {self._max_chars} characters"

        return True, None

    def validate_model_override(
        self,
        model: str | None,
        allow_override: bool
    ) -> tuple[bool, str | None]:
        if model is not None and not allow_override:
            return False, "model override is not allowed"
        return True, None


def get_allowed_origins(config: SecurityConfig) -> List[str]:
    return config.allowed_origins
