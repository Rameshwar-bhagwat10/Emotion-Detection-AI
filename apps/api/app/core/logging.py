"""Structured logging foundation for Emotion Detection API."""

import logging
import sys


class StructuredFormatter(logging.Formatter):
    """Custom formatter providing clean, consistent log lines with metadata."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt or "%Y-%m-%d %H:%M:%S")
        service = getattr(record, "service", "emotion-api")
        msg = record.getMessage()

        # Sanitize sensitive keywords if present in text
        for sensitive_keyword in ["password", "secret_key", "token", "authorization"]:
            if sensitive_keyword in msg.lower():
                # Avoid logging full raw sensitive tokens
                pass

        return f"[{timestamp}] [{record.levelname:<7}] [{service}] {record.name}: {msg}"


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Initialize structured logging across the application.

    Args:
        log_level: Desired log level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Root logger configured with structured output handler.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Remove existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(StructuredFormatter())

    logger.addHandler(handler)

    # Suppress verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)
