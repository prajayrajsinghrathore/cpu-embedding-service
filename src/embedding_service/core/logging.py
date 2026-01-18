"""Logging configuration for CPU Embedding Service."""

from pilot_common.logging import (
    configure_logging,
    get_logger,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
)

__all__ = [
    "configure_logging",
    "get_logger",
    "get_correlation_id",
    "set_correlation_id",
    "clear_correlation_id",
    "setup_logging",
]


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the embedding service."""
    configure_logging(
        service_name="cpu-embedding-service",
        service_version="1.0.0",
        log_level=level,
        log_format="json",
    )
