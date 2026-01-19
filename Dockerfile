# Simple pip-based build - no Poetry, no apt-get needed
# Requirements exported on host via: poetry export -f requirements.txt --only main --without-hashes -o requirements.txt
#
# Corporate CA cert handling:
#   - Tilt injects cert at build time (see Tiltfile inject_ca_cert_cmd)
#   - Manual build: docker build --build-arg CA_CERT=path/to/cert.crt .
#   - Non-corp envs: no cert needed, glob pattern fails silently

FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Optional: Copy corporate CA cert for SSL/TLS (ZScaler/corporate proxy)
# The [t] glob pattern makes COPY succeed even if file is missing
COPY corporate-ca.cr[t] /tmp/

# Install CA cert if present (corporate environment)
RUN if [ -f /tmp/corporate-ca.crt ]; then \
        mkdir -p /usr/local/share/ca-certificates && \
        cp /tmp/corporate-ca.crt /usr/local/share/ca-certificates/; \
    fi

# Set SSL env vars only if cert exists
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Copy wheels (pilot-common built on host)
COPY wheels/ ./wheels/

# Copy requirements.txt (exported from Poetry on host)
COPY requirements.txt .

# Install dependencies using pip - use cert if present
RUN if [ -f /usr/local/share/ca-certificates/corporate-ca.crt ]; then \
        pip install --upgrade pip --cert /usr/local/share/ca-certificates/corporate-ca.crt && \
        pip install -r requirements.txt --cert /usr/local/share/ca-certificates/corporate-ca.crt; \
    else \
        pip install --upgrade pip && \
        pip install -r requirements.txt; \
    fi

# Copy application code
COPY cpu_embedding_service/ ./cpu_embedding_service/



# Runtime stage - minimal image
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Create non-root user
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY --from=builder /app/cpu_embedding_service ./cpu_embedding_service


RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["cpu_embedding_service.app:app", "--host", "0.0.0.0", "--port", "8080"]
