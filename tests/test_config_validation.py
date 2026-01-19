from __future__ import annotations

import json
import os
import tempfile

import pytest
from pydantic import ValidationError

from cpu_embedding_service.config import (
    AppConfig,
    EmbeddingsConfig,
    ObservabilityConfig,
    SecurityConfig,
    ServiceConfig,
    load_config,
    validate_config,
)


class TestServiceConfig:
    def test_valid_config(self) -> None:
        config = ServiceConfig(host="0.0.0.0", port=8080)
        assert config.host == "0.0.0.0"
        assert config.port == 8080

    def test_default_values(self) -> None:
        config = ServiceConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8008

    def test_invalid_port_too_high(self) -> None:
        with pytest.raises(ValidationError):
            ServiceConfig(host="0.0.0.0", port=70000)

    def test_invalid_port_too_low(self) -> None:
        with pytest.raises(ValidationError):
            ServiceConfig(host="0.0.0.0", port=0)

    def test_empty_host(self) -> None:
        with pytest.raises(ValidationError):
            ServiceConfig(host="", port=8080)


class TestEmbeddingsConfig:
    def test_valid_config(self) -> None:
        config = EmbeddingsConfig(
            default_model="test-model",
            batch_max_texts=100,
            max_chars_per_text=5000
        )
        assert config.default_model == "test-model"
        assert config.batch_max_texts == 100

    def test_default_values(self) -> None:
        config = EmbeddingsConfig()
        assert config.default_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.allow_model_override is False
        assert config.normalize_default is True
        assert config.batch_max_texts == 64
        assert config.max_chars_per_text == 8000

    def test_invalid_batch_max_texts_zero(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingsConfig(batch_max_texts=0)

    def test_invalid_batch_max_texts_too_high(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingsConfig(batch_max_texts=2000)

    def test_invalid_max_chars_zero(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingsConfig(max_chars_per_text=0)

    def test_empty_model_name(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingsConfig(default_model="")

    def test_whitespace_model_name(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingsConfig(default_model="   ")


class TestObservabilityConfig:
    def test_valid_config(self) -> None:
        config = ObservabilityConfig(
            enabled=True,
            service_name="test-service",
            exporter="otlp",
            otlp_endpoint="http://localhost:4317",
            sampling_ratio=0.5
        )
        assert config.enabled is True
        assert config.sampling_ratio == 0.5

    def test_invalid_exporter(self) -> None:
        with pytest.raises(ValidationError):
            ObservabilityConfig(exporter="invalid")

    def test_invalid_sampling_ratio_too_high(self) -> None:
        with pytest.raises(ValidationError):
            ObservabilityConfig(sampling_ratio=1.5)

    def test_invalid_sampling_ratio_negative(self) -> None:
        with pytest.raises(ValidationError):
            ObservabilityConfig(sampling_ratio=-0.1)


class TestSecurityConfig:
    def test_valid_config(self) -> None:
        config = SecurityConfig(
            allowed_origins=["http://localhost:3000"],
            request_id_header="X-Request-ID"
        )
        assert len(config.allowed_origins) == 1

    def test_empty_origins_list(self) -> None:
        with pytest.raises(ValidationError):
            SecurityConfig(allowed_origins=[])


class TestAppConfig:
    def test_full_config(self) -> None:
        config = AppConfig(
            service=ServiceConfig(host="127.0.0.1", port=9000),
            embeddings=EmbeddingsConfig(default_model="custom-model"),
            observability=ObservabilityConfig(enabled=True),
            security=SecurityConfig(allowed_origins=["*"])
        )
        assert config.service.port == 9000
        assert config.embeddings.default_model == "custom-model"


class TestLoadConfig:
    def test_load_from_file(self) -> None:
        config_data = {
            "service": {"host": "127.0.0.1", "port": 9000},
            "embeddings": {"default_model": "file-model"}
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config.service.port == 9000
            assert config.embeddings.default_model == "file-model"
        finally:
            os.unlink(config_path)

    def test_load_with_env_override(self) -> None:
        config_data = {"service": {"port": 8000}}
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            os.environ["EMBEDDING_SERVICE_PORT"] = "9999"
            config = load_config(config_path)
            assert config.service.port == 9999
        finally:
            os.unlink(config_path)
            del os.environ["EMBEDDING_SERVICE_PORT"]

    def test_load_missing_file_uses_defaults(self) -> None:
        config = load_config("/nonexistent/path.json")
        assert config.service.port == 8008


class TestValidateConfig:
    def test_valid_config(self) -> None:
        config = AppConfig()
        valid, error = validate_config(config)
        assert valid is True
        assert error is None

    def test_invalid_batch_max_texts(self) -> None:
        config = AppConfig()
        config.embeddings.batch_max_texts = 0
        valid, error = validate_config(config)
        assert valid is False
        assert "batch_max_texts" in error
