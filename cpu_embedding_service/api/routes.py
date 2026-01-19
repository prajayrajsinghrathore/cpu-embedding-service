from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import APIRouter, HTTPException, Request, Response

from cpu_embedding_service.api.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ReadyResponse,
    UsageStats,
)
from cpu_embedding_service.config import AppConfig, validate_config
from cpu_embedding_service.core.observability import add_span_attributes, trace_encode_operation
from cpu_embedding_service.core.security import (
    InputValidator,
    get_correlation_id,
    set_correlation_id_header,
)
from cpu_embedding_service.engine.base import EmbeddingEngine

logger = logging.getLogger(__name__)

router = APIRouter()


def create_routes(
    config: AppConfig,
    get_engine: Callable[[], EmbeddingEngine],
    get_config: Callable[[], AppConfig],
) -> APIRouter:
    router = APIRouter()
    validator = InputValidator(config.embeddings)
    request_id_header = config.security.request_id_header

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @router.get("/ready", response_model=ReadyResponse)
    async def ready() -> ReadyResponse:
        current_config = get_config()
        config_valid, config_error = validate_config(current_config)
        eng = get_engine()
        model_loaded = eng.is_loaded()

        details = None
        if not config_valid or not model_loaded:
            details = {}
            if not config_valid:
                details["config_error"] = config_error or "Unknown configuration error"
            if not model_loaded:
                details["model_error"] = "Model failed to load"

        return ReadyResponse(
            ready=model_loaded and config_valid,
            model_loaded=model_loaded,
            config_valid=config_valid,
            details=details
        )

    @router.post("/embeddings", response_model=EmbeddingResponse)
    async def create_embeddings(
        request: Request,
        response: Response,
        body: EmbeddingRequest
    ) -> EmbeddingResponse:
        correlation_id = get_correlation_id(request, request_id_header)
        set_correlation_id_header(response, correlation_id, request_id_header)

        log_extra = {"correlation_id": correlation_id, "batch_size": len(body.texts)}

        valid, error = validator.validate_model_override(
            body.model,
            config.embeddings.allow_model_override
        )
        if not valid:
            logger.warning("Model override rejected", extra=log_extra)
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error=ErrorDetail(code="MODEL_OVERRIDE_DISABLED", message=error or "")
                ).model_dump()
            )

        valid, error = validator.validate_texts(body.texts)
        if not valid:
            logger.warning("Input validation failed", extra=log_extra)
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error=ErrorDetail(code="VALIDATION_ERROR", message=error or "")
                ).model_dump()
            )

        model_name = body.model or config.embeddings.default_model
        normalize = body.normalize if body.normalize is not None else config.embeddings.normalize_default
        truncate = body.truncate if body.truncate is not None else config.embeddings.truncate_default
        eng= get_engine()
        if body.model and body.model != eng.get_model_name():
            try:
                eng.load_model(body.model)
            except Exception:
                logger.error(
                    "Failed to load requested model",
                    extra={**log_extra, "requested_model": body.model}
                )
                raise HTTPException(
                    status_code=500,
                    detail=ErrorResponse(
                        error=ErrorDetail(
                            code="MODEL_LOAD_ERROR",
                            message="Failed to load requested model"
                        )
                    ).model_dump()
                )

        total_chars = sum(len(text) for text in body.texts)
        start_time = time.perf_counter()

        try:
            with trace_encode_operation(model_name, len(body.texts), total_chars):
                embeddings = eng.encode(body.texts, normalize=normalize, truncate=truncate)

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            add_span_attributes({
                "encoding.elapsed_ms": elapsed_ms,
                "encoding.dimension": eng.get_dimension()
            })

            logger.info(
                "Embeddings generated",
                extra={
                    **log_extra,
                    "total_chars": total_chars,
                    "elapsed_ms": elapsed_ms,
                    "model": eng.get_model_name(),
                    "dimension": eng.get_dimension()
                }
            )

            return EmbeddingResponse(
                model=eng.get_model_name(),
                dim=eng.get_dimension(),
                embeddings=embeddings,
                usage=UsageStats(
                    texts=len(body.texts),
                    chars=total_chars,
                    ms=elapsed_ms
                )
            )

        except Exception as e:
            logger.error(
                "Encoding failed",
                extra={**log_extra, "error_type": type(e).__name__}
            )
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error=ErrorDetail(code="ENCODING_ERROR", message="Failed to generate embeddings")
                ).model_dump()
            )

    return router
