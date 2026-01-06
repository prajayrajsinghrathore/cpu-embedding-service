from embedding_service.core.logging import setup_logging
from embedding_service.core.observability import (
    add_span_attributes,
    instrument_fastapi,
    setup_observability,
    trace_encode_operation,
)
from embedding_service.core.security import (
    InputValidator,
    generate_correlation_id,
    get_allowed_origins,
    get_correlation_id,
    set_correlation_id_header,
)

__all__ = [
    "setup_logging",
    "setup_observability",
    "instrument_fastapi",
    "trace_encode_operation",
    "add_span_attributes",
    "InputValidator",
    "generate_correlation_id",
    "get_correlation_id",
    "set_correlation_id_header",
    "get_allowed_origins",
]
