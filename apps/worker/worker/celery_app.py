"""Celery application instance and task registration."""

import os
from typing import Any

from celery import Celery

# Read broker and backend URLs from environment
broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

# Create Celery application instance
celery_app = Celery(
    "emotion_worker",
    broker=broker_url,
    backend=result_backend,
    include=[
        "worker.celery_app",
    ],
)

# Configure Celery settings
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    worker_prefetch_multiplier=1,
)


@celery_app.task(name="worker.tasks.health_check_task", bind=True)
def health_check_task(self, x: int = 1, y: int = 1) -> dict[str, Any]:
    """Trivial health check task to verify Celery broker and worker execution flow."""
    return {
        "status": "success",
        "task_id": self.request.id,
        "result": x + y,
        "service": "emotion-worker",
    }
