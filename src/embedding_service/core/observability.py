from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from embedding_service.config import ObservabilityConfig

logger = logging.getLogger(__name__)

_tracer: Optional[Any] = None
_otel_enabled: bool = False


def setup_observability(config: ObservabilityConfig) -> None:
    global _tracer, _otel_enabled

    if not config.enabled:
        logger.info("OpenTelemetry disabled by configuration")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        resource = Resource.create({
            "service.name": config.service_name,
        })

        sampler = TraceIdRatioBased(config.sampling_ratio)
        provider = TracerProvider(resource=resource, sampler=sampler)

        if config.exporter == "otlp":
            exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint, insecure=True)
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(config.service_name)
        _otel_enabled = True

        logger.info(
            "OpenTelemetry initialized",
            extra={
                "service_name": config.service_name,
                "exporter": config.exporter,
                "endpoint": config.otlp_endpoint,
                "sampling_ratio": config.sampling_ratio
            }
        )

    except ImportError:
        logger.warning("OpenTelemetry packages not installed, tracing disabled")
    except Exception as e:
        logger.error(
            "Failed to initialize OpenTelemetry",
            extra={"error_type": type(e).__name__}
        )


def instrument_fastapi(app: Any) -> None:
    if not _otel_enabled:
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented with OpenTelemetry")
    except ImportError:
        logger.warning("FastAPI instrumentation not available")
    except Exception as e:
        logger.error(
            "Failed to instrument FastAPI",
            extra={"error_type": type(e).__name__}
        )


@contextmanager
def trace_encode_operation(
    model_name: str,
    batch_size: int,
    total_chars: int
) -> Generator[None, None, None]:
    if not _otel_enabled or _tracer is None:
        yield
        return

    try:
        from opentelemetry import trace as otel_trace

        with _tracer.start_as_current_span("encode_embeddings") as span:
            span.set_attribute("model.name", model_name)
            span.set_attribute("batch.size", batch_size)
            span.set_attribute("total.chars", total_chars)
            yield
    except Exception:
        yield


def add_span_attributes(attributes: Dict[str, Any]) -> None:
    if not _otel_enabled:
        return

    try:
        from opentelemetry import trace as otel_trace
        span = otel_trace.get_current_span()
        for key, value in attributes.items():
            span.set_attribute(key, value)
    except Exception:
        pass
