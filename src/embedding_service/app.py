from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from embedding_service.api.routes import create_routes
from embedding_service.config import AppConfig, get_config, validate_config
from embedding_service.core.logging import setup_logging
from embedding_service.core.observability import instrument_fastapi, setup_observability
from embedding_service.core.security import get_allowed_origins
from embedding_service.engine import EmbeddingEngine, SentenceTransformerEngine

logger = logging.getLogger(__name__)

_engine: EmbeddingEngine | None = None
_config: AppConfig | None = None


def get_engine() -> EmbeddingEngine:
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine


def get_app_config() -> AppConfig:
    if _config is None:
        raise RuntimeError("Config not initialized")
    return _config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _engine, _config

    setup_logging()
    logger.info("Starting embedding service")

    try:
        _config = get_config()
        valid, error = validate_config(_config)
        if not valid:
            logger.error("Configuration validation failed", extra={"error": error})
            raise RuntimeError(f"Configuration validation failed: {error}")

        setup_observability(_config.observability)

        _engine = SentenceTransformerEngine()
        _engine.load_model(_config.embeddings.default_model)

        logger.info(
            "Service initialized",
            extra={
                "model": _config.embeddings.default_model,
                "dimension": _engine.get_dimension(),
                "host": _config.service.host,
                "port": _config.service.port
            }
        )

    except Exception as e:
        logger.error(
            "Failed to initialize service",
            extra={"error_type": type(e).__name__}
        )
        _engine = SentenceTransformerEngine()

    yield

    logger.info("Shutting down embedding service")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Embedding Service",
        description="CPU-only embedding microservice using SentenceTransformers",
        version="1.0.0",
        lifespan=lifespan
    )

    config = get_config()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(config.security),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    instrument_fastapi(app)

    routes = create_routes(config, get_engine, get_app_config)
    app.include_router(routes)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    config = get_config()
    uvicorn.run(
        "embedding_service.app:app",
        host=config.service.host,
        port=config.service.port,
        reload=False,
        workers=1
    )


if __name__ == "__main__":
    main()
