from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from embedding_service.api.routes import create_routes
from embedding_service.config import AppConfig, EmbeddingsConfig
from tests.conftest import MockEmbeddingEngine


class TestBatchLimits:
    def test_batch_within_limit(self, client: TestClient) -> None:
        response = client.post(
            "/embeddings",
            json={"texts": ["text1", "text2", "text3"]}
        )
        assert response.status_code == 200
        assert len(response.json()["embeddings"]) == 3

    def test_batch_exceeds_limit(self, default_config: AppConfig) -> None:
        default_config.embeddings.batch_max_texts = 5
        engine = MockEmbeddingEngine()
        
        app = FastAPI()
        routes = create_routes(default_config, engine, lambda: default_config)
        app.include_router(routes)
        
        with TestClient(app) as client:
            response = client.post(
                "/embeddings",
                json={"texts": ["text"] * 10}
            )
            assert response.status_code == 400
            assert "exceeds maximum" in response.json()["detail"]["error"]["message"]

    def test_batch_at_limit(self, default_config: AppConfig) -> None:
        default_config.embeddings.batch_max_texts = 5
        engine = MockEmbeddingEngine()
        
        app = FastAPI()
        routes = create_routes(default_config, engine, lambda: default_config)
        app.include_router(routes)
        
        with TestClient(app) as client:
            response = client.post(
                "/embeddings",
                json={"texts": ["text"] * 5}
            )
            assert response.status_code == 200


class TestCharacterLimits:
    def test_text_within_limit(self, client: TestClient) -> None:
        response = client.post(
            "/embeddings",
            json={"texts": ["short text"]}
        )
        assert response.status_code == 200

    def test_text_exceeds_limit(self, default_config: AppConfig) -> None:
        default_config.embeddings.max_chars_per_text = 100
        engine = MockEmbeddingEngine()
        
        app = FastAPI()
        routes = create_routes(default_config, engine, lambda: default_config)
        app.include_router(routes)
        
        with TestClient(app) as client:
            long_text = "x" * 200
            response = client.post(
                "/embeddings",
                json={"texts": [long_text]}
            )
            assert response.status_code == 400
            assert "exceeds maximum" in response.json()["detail"]["error"]["message"]

    def test_second_text_exceeds_limit(self, default_config: AppConfig) -> None:
        default_config.embeddings.max_chars_per_text = 100
        engine = MockEmbeddingEngine()
        
        app = FastAPI()
        routes = create_routes(default_config, engine, lambda: default_config)
        app.include_router(routes)
        
        with TestClient(app) as client:
            response = client.post(
                "/embeddings",
                json={"texts": ["short", "x" * 200]}
            )
            assert response.status_code == 400
            assert "index 1" in response.json()["detail"]["error"]["message"]


class TestEmptyInput:
    def test_empty_texts_list(self, client: TestClient) -> None:
        response = client.post(
            "/embeddings",
            json={"texts": []}
        )
        assert response.status_code == 422

    def test_missing_texts_field(self, client: TestClient) -> None:
        response = client.post(
            "/embeddings",
            json={}
        )
        assert response.status_code == 422


class TestModelOverride:
    def test_model_override_disabled_by_default(self, client: TestClient) -> None:
        response = client.post(
            "/embeddings",
            json={"texts": ["test"], "model": "other-model"}
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"]["code"] == "MODEL_OVERRIDE_DISABLED"

    def test_model_override_enabled(self, default_config: AppConfig) -> None:
        default_config.embeddings.allow_model_override = True
        engine = MockEmbeddingEngine()
        
        app = FastAPI()
        routes = create_routes(default_config, engine, lambda: default_config)
        app.include_router(routes)
        
        with TestClient(app) as client:
            response = client.post(
                "/embeddings",
                json={"texts": ["test"], "model": "other-model"}
            )
            assert response.status_code == 200
