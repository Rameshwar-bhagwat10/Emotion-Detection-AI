"""Tests for Celery worker tasks."""

from worker.celery_app import health_check_task


def test_health_check_task_execution():
    """Verify that health_check_task executes correctly and returns expected payload."""
    result = health_check_task(x=5, y=10)

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["result"] == 15
    assert result["service"] == "emotion-worker"
