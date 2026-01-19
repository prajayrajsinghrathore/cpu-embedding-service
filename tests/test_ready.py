from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestReadyEndpoint:
    def test_ready_when_model_loaded(self, client: TestClient) -> None:
        response = client.get("/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["ready"] is True
        assert data["model_loaded"] is True
        assert data["config_valid"] is True
        assert data["details"] is None

    def test_not_ready_when_model_fails(self, failing_client: TestClient) -> None:
        response = failing_client.get("/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["ready"] is False
        assert data["model_loaded"] is False
        assert data["details"] is not None
        assert "model_error" in data["details"]


class TestReadyResponseFormat:
    def test_ready_response_has_required_fields(self, client: TestClient) -> None:
        response = client.get("/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "ready" in data
        assert "model_loaded" in data
        assert "config_valid" in data

    def test_ready_response_types(self, client: TestClient) -> None:
        response = client.get("/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data["ready"], bool)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["config_valid"], bool)
