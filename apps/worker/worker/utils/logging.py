"""Worker logging utility foundation."""

import logging
import sys


def setup_worker_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure Celery worker structured logging."""
    logger = logging.getLogger("emotion-worker")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [emotion-worker] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
