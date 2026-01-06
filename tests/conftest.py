from __future__ import annotations

from typing import Generator, List
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from embedding_service.api.routes import create_routes
from embedding_service.config import (
    AppConfig,
    EmbeddingsConfig,
    ObservabilityConfig,
    SecurityConfig,
    ServiceConfig,
)
from embedding_service.engine.base import EmbeddingEngine


class MockEmbeddingEngine(EmbeddingEngine):
    def __init__(self, should_fail: bool = False, dimension: int = 384) -> None:
        self._loaded = not should_fail
        self._should_fail = should_fail
        self._model_name = "mock-model"
        self._dimension = dimension

    def load_model(self, model_name: str) -> None:
        if self._should_fail:
            raise RuntimeError("Mock model load failure")
        self._model_name = model_name
        self._loaded = True

    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        truncate: bool = True
    ) -> List[List[float]]:
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        return [[0.1] * self._dimension for _ in texts]

    def get_dimension(self) -> int:
        return self._dimension

    def get_model_name(self) -> str:
        return self._model_name

    def is_loaded(self) -> bool:
        return self._loaded

    def supports_model(self, model_name: str) -> bool:
        return True


@pytest.fixture
def default_config() -> AppConfig:
    return AppConfig(
        service=ServiceConfig(host="0.0.0.0", port=8008),
        embeddings=EmbeddingsConfig(
            default_model="test-model",
            allow_model_override=False,
            normalize_default=True,
            truncate_default=True,
            batch_max_texts=64,
            max_chars_per_text=8000,
            request_timeout_seconds=60
        ),
        observability=ObservabilityConfig(enabled=False),
        security=SecurityConfig(allowed_origins=["*"], request_id_header="X-Request-ID")
    )


@pytest.fixture
def mock_engine() -> MockEmbeddingEngine:
    return MockEmbeddingEngine()


@pytest.fixture
def failing_engine() -> MockEmbeddingEngine:
    return MockEmbeddingEngine(should_fail=True)


@pytest.fixture
def test_app(default_config: AppConfig, mock_engine: MockEmbeddingEngine) -> FastAPI:
    app = FastAPI()
    routes = create_routes(default_config, mock_engine, lambda: default_config)
    app.include_router(routes)
    return app


@pytest.fixture
def client(test_app: FastAPI) -> Generator[TestClient, None, None]:
    with TestClient(test_app) as c:
        yield c


@pytest.fixture
def failing_app(default_config: AppConfig, failing_engine: MockEmbeddingEngine) -> FastAPI:
    app = FastAPI()
    routes = create_routes(default_config, failing_engine, lambda: default_config)
    app.include_router(routes)
    return app


@pytest.fixture
def failing_client(failing_app: FastAPI) -> Generator[TestClient, None, None]:
    with TestClient(failing_app) as c:
        yield c
