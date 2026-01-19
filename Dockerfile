# Simple pip-based build - no Poetry, no apt-get needed
# Requirements exported on host via: poetry export -f requirements.txt --only main --without-hashes -o requirements.txt

FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Copy requirements.txt (exported from Poetry on host)
COPY requirements.txt .

# Copy wheel files
COPY wheels/ ./wheels/

# Install dependencies using pip
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install wheels/*.whl

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
