# Embedding Service

A production-ready, CPU-only embedding microservice using SentenceTransformers and FastAPI. Designed to generate vector embeddings for text, typically called by an ingestion worker that stores vectors in Milvus.

## Features

- **CPU-Only Inference**: Optimized for CPU deployment without GPU dependencies
- **Batch Processing**: Process multiple texts in a single request
- **Request Limits**: Configurable limits to protect CPU resources
- **OpenTelemetry Support**: Optional distributed tracing with OTLP exporter
- **Kubernetes Ready**: Helm chart with Istio support for AKS deployment
- **Model Caching**: Persistent volume support for faster restarts

## Quick Start

### Local Development with Poetry

```bash
# Install dependencies
poetry install

# Copy and configure
cp config.json.template config.json

# Run the service
poetry run python -m embedding_service.app
```

### Local Development with Docker

```bash
# Build the image
docker build -t embedding-service:latest .

# Run with default configuration
docker run -p 8008:8008 embedding-service:latest

# Run with custom configuration
docker run -p 8008:8008 \
  -e EMBEDDING_DEFAULT_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
  -v /path/to/models:/models \
  embedding-service:latest
```

### Running Tests

```bash
# Install dev dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=embedding_service
```

## API Reference

### POST /embeddings

Generate embeddings for input texts.

**Request Body:**

```json
{
  "model": "optional-model-override",
  "texts": ["First text to embed", "Second text to embed"],
  "normalize": true,
  "truncate": true,
  "metadata": {
    "project_id": "my-project",
    "correlation_id": "request-123"
  }
}
```

**Response:**

```json
{
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "dim": 384,
  "embeddings": [[0.123, -0.456, ...], [0.789, -0.012, ...]],
  "usage": {
    "texts": 2,
    "chars": 42,
    "ms": 125
  }
}
```

**Example curl:**

```bash
curl -X POST http://localhost:8008/embeddings \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: my-correlation-id" \
  -d '{
    "texts": ["Hello, world!", "How are you?"],
    "normalize": true
  }'
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:8008/health
# {"status": "ok"}
```

### GET /ready

Readiness check endpoint. Returns false if model failed to load or configuration is invalid.

```bash
curl http://localhost:8008/ready
# {"ready": true, "model_loaded": true, "config_valid": true, "details": null}
```

## Configuration

### Configuration File

Copy `config.json.template` to `config.json` and customize:

```json
{
  "service": {
    "host": "0.0.0.0",
    "port": 8008
  },
  "embeddings": {
    "default_model": "sentence-transformers/all-MiniLM-L6-v2",
    "allow_model_override": false,
    "normalize_default": true,
    "truncate_default": true,
    "batch_max_texts": 64,
    "max_chars_per_text": 8000,
    "request_timeout_seconds": 60
  },
  "observability": {
    "enabled": false,
    "service_name": "embedding-service",
    "exporter": "otlp",
    "otlp_endpoint": "http://otel-collector:4317",
    "sampling_ratio": 1.0
  },
  "security": {
    "allowed_origins": ["*"],
    "request_id_header": "X-Request-ID"
  }
}
```

### Environment Variable Overrides

All configuration options can be overridden via environment variables:

| Environment Variable | Config Path | Type |
|---------------------|-------------|------|
| `EMBEDDING_SERVICE_HOST` | service.host | string |
| `EMBEDDING_SERVICE_PORT` | service.port | int |
| `EMBEDDING_DEFAULT_MODEL` | embeddings.default_model | string |
| `EMBEDDING_ALLOW_MODEL_OVERRIDE` | embeddings.allow_model_override | bool |
| `EMBEDDING_NORMALIZE_DEFAULT` | embeddings.normalize_default | bool |
| `EMBEDDING_TRUNCATE_DEFAULT` | embeddings.truncate_default | bool |
| `EMBEDDING_BATCH_MAX_TEXTS` | embeddings.batch_max_texts | int |
| `EMBEDDING_MAX_CHARS_PER_TEXT` | embeddings.max_chars_per_text | int |
| `EMBEDDING_REQUEST_TIMEOUT` | embeddings.request_timeout_seconds | int |
| `OTEL_ENABLED` | observability.enabled | bool |
| `OTEL_SERVICE_NAME` | observability.service_name | string |
| `OTEL_EXPORTER` | observability.exporter | string |
| `OTEL_ENDPOINT` | observability.otlp_endpoint | string |
| `OTEL_SAMPLING_RATIO` | observability.sampling_ratio | float |
| `SECURITY_ALLOWED_ORIGINS` | security.allowed_origins | comma-separated |
| `SECURITY_REQUEST_ID_HEADER` | security.request_id_header | string |

## Model Selection

### Default Model

The default model is configurable in `config.json` or via `EMBEDDING_DEFAULT_MODEL` environment variable.

Recommended models for different use cases:

| Model | Dimension | Use Case |
|-------|-----------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | General purpose, fast |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Higher quality, slower |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual |

### Per-Request Model Override

By default, per-request model override is disabled. To enable:

```json
{
  "embeddings": {
    "allow_model_override": true
  }
}
```

Then specify the model in requests:

```bash
curl -X POST http://localhost:8008/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-mpnet-base-v2",
    "texts": ["Hello world"]
  }'
```

## Model Cache

### Mounting Model Cache for Faster Startup

Mount a volume to cache downloaded models:

```bash
# Docker
docker run -p 8008:8008 \
  -v /path/to/model-cache:/models \
  embedding-service:latest

# The following directories are used:
# - TRANSFORMERS_CACHE=/models/cache
# - HF_HOME=/models
# - SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers
```

### Kubernetes PVC

In Helm values.yaml:

```yaml
modelCache:
  enabled: true
  size: 5Gi
  storageClass: "managed-premium"  # AKS storage class
```

## CPU-Only Deployment

The service is designed for CPU-only inference:

1. **Dockerfile** sets `CUDA_VISIBLE_DEVICES=""` to disable GPU
2. **SentenceTransformer** is initialized with `device="cpu"`
3. No CUDA/cuDNN dependencies in `pyproject.toml`

To verify CPU-only operation:

```python
import torch
print(torch.cuda.is_available())  # Should be False
```

## Integration with Ingestion Worker

### Configuration

Set the following environment variable in your ingestion worker:

```bash
EMBEDDING_SERVICE_URL=http://embedding-service:8008
```

### Usage Example

```python
import httpx

async def get_embeddings(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{EMBEDDING_SERVICE_URL}/embeddings",
            json={"texts": texts},
            headers={"X-Request-ID": correlation_id}
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"]
```

### Milvus Collection Dimension

Ensure your Milvus collection dimension matches the embedding dimension:

```python
# For all-MiniLM-L6-v2
collection_schema = CollectionSchema(
    fields=[
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
)
```

## Kubernetes Deployment

### Helm Installation

```bash
# Add your values
cp helm/embedding-service/values.yaml my-values.yaml
# Edit my-values.yaml

# Install
helm install embedding-service ./helm/embedding-service \
  -f my-values.yaml \
  -n your-namespace
```

### Key Helm Values

```yaml
# Image configuration
image:
  repository: your-registry.azurecr.io/embedding-service
  tag: "1.0.0"

# Resource limits (important for CPU inference)
resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 2Gi

# Enable OpenTelemetry
config:
  observability:
    enabled: true
    otlpEndpoint: "http://otel-collector.observability:4317"

# Enable Istio VirtualService
istio:
  enabled: true
  virtualService:
    enabled: true
```

### Istio Integration

The Helm chart includes:

- Pod annotation for Istio sidecar injection
- VirtualService for traffic routing
- Optional Gateway for external access
- Retry and timeout configuration

## Observability

### OpenTelemetry

Enable distributed tracing:

```json
{
  "observability": {
    "enabled": true,
    "service_name": "embedding-service",
    "exporter": "otlp",
    "otlp_endpoint": "http://otel-collector:4317",
    "sampling_ratio": 1.0
  }
}
```

### Trace Attributes

The encode span includes:

- `model.name`: Model used for encoding
- `batch.size`: Number of texts in batch
- `total.chars`: Total characters processed
- `encoding.elapsed_ms`: Encoding duration
- `encoding.dimension`: Embedding dimension

### Correlation ID

The service accepts and propagates correlation IDs:

- Header: `X-Request-ID` (configurable)
- Generated automatically if not provided
- Returned in response headers
- Included in structured logs

## Security

### Input Validation

- Empty texts list: Rejected
- Texts exceeding `max_chars_per_text`: Rejected
- Batch exceeding `batch_max_texts`: Rejected
- Model override when disabled: Rejected with 400

### Safe Logging

- Raw text contents are never logged
- Only metadata (batch size, character counts) is logged
- Error messages are sanitized

### CORS

Configure allowed origins in `config.json`:

```json
{
  "security": {
    "allowed_origins": ["https://your-app.com"]
  }
}
```

## Project Structure

```
embedding-service/
├── pyproject.toml
├── README.md
├── Dockerfile
├── config.json.template
├── src/embedding_service/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── observability.py
│   │   └── security.py
│   └── engine/
│       ├── __init__.py
│       ├── base.py
│       └── sentence_transformers.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_config_validation.py
│   ├── test_limits.py
│   ├── test_embeddings_endpoint.py
│   └── test_ready.py
└── helm/embedding-service/
    ├── Chart.yaml
    ├── values.yaml
    └── templates/
        ├── _helpers.tpl
        ├── deployment.yaml
        ├── service.yaml
        ├── configmap.yaml
        ├── serviceaccount.yaml
        ├── pvc.yaml
        ├── hpa.yaml
        ├── istio-virtualservice.yaml
        └── istio-gateway.yaml
```

## License

MIT
