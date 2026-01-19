from __future__ import annotations

from fastapi.testclient import TestClient


class TestEmbeddingsEndpoint:
    def test_successful_embedding_single_text(self, client: TestClient) -> None:
        response = client.post(
            "/embeddings",
            json={"texts": ["Hello world"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model" in data
        assert "dim" in data
        assert "embeddings" in data
        assert "usage" in data
        
        assert data["dim"] == 384
        assert len(data["embeddings"]) == 1
        assert len(data["embeddings"][0]) == 384

    def test_successful_embedding_multiple_texts(self, client: TestClient) -> None:
        texts = ["First text", "Second text", "Third text"]
        response = client.post(
            "/embeddings",
            json={"texts": texts}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["embeddings"]) == 3
        for embedding in data["embeddings"]:
            assert len(embedding) == data["dim"]

    def test_usage_stats(self, client: TestClient) -> None:
        texts = ["Hello", "World"]
        response = client.post(
            "/embeddings",
            json={"texts": texts}
        )
        
        assert response.status_code == 200
        usage = response.json()["usage"]
        
        assert usage["texts"] == 2
        assert usage["chars"] == 10
        assert usage["ms"] >= 0

    def test_normalize_parameter(self, client: TestClient) -> None:
        response = client.post(
            "/embeddings",
            json={"texts": ["test"], "normalize": False}
        )
        
        assert response.status_code == 200

    def test_truncate_parameter(self, client: TestClient) -> None:
        response = client.post(
            "/embeddings",
            json={"texts": ["test"], "truncate": True}
        )
        
        assert response.status_code == 200

    def test_metadata_included(self, client: TestClient) -> None:
        response = client.post(
            "/embeddings",
            json={
                "texts": ["test"],
                "metadata": {
                    "project_id": "proj-123",
                    "correlation_id": "corr-456"
                }
            }
        )
        
        assert response.status_code == 200

    def test_correlation_id_header_returned(self, client: TestClient) -> None:
        response = client.post(
            "/embeddings",
            json={"texts": ["test"]},
            headers={"X-Request-ID": "test-correlation-id"}
        )
        
        assert response.status_code == 200
        assert response.headers.get("X-Request-ID") == "test-correlation-id"

    def test_correlation_id_generated_if_missing(self, client: TestClient) -> None:
        response = client.post(
            "/embeddings",
            json={"texts": ["test"]}
        )
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert len(response.headers.get("X-Request-ID", "")) > 0


class TestEmbeddingsDimension:
    def test_consistent_dimension(self, client: TestClient) -> None:
        response1 = client.post("/embeddings", json={"texts": ["text1"]})
        response2 = client.post("/embeddings", json={"texts": ["text2", "text3"]})
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        assert response1.json()["dim"] == response2.json()["dim"]
        
        dim = response1.json()["dim"]
        for embedding in response2.json()["embeddings"]:
            assert len(embedding) == dim


class TestEmbeddingsResponseFormat:
    def test_embeddings_are_lists_of_floats(self, client: TestClient) -> None:
        response = client.post(
            "/embeddings",
            json={"texts": ["test"]}
        )
        
        assert response.status_code == 200
        embeddings = response.json()["embeddings"]
        
        assert isinstance(embeddings, list)
        assert isinstance(embeddings[0], list)
        for value in embeddings[0]:
            assert isinstance(value, float)

    def test_model_name_in_response(self, client: TestClient) -> None:
        response = client.post(
            "/embeddings",
            json={"texts": ["test"]}
        )
        
        assert response.status_code == 200
        assert response.json()["model"] == "mock-model"
