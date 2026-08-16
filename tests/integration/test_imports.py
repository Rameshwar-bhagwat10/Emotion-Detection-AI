"""Import verification tests for Phase 01 environment dependencies."""

import importlib

import pytest


@pytest.mark.parametrize(
    "package_name",
    [
        # Machine Learning / Deep Learning Frameworks
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "PIL",
        "cv2",
        "mediapipe",
        "sklearn",
        "matplotlib",
        "seaborn",
        "timm",
        "pytorch_grad_cam",
        "yaml",
        # Backend / Web API & Background Processing
        "fastapi",
        "uvicorn",
        "pydantic",
        "pydantic_settings",
        "sqlalchemy",
        "asyncpg",
        "alembic",
        "redis",
        "celery",
        "jwt",
        "argon2",
        "httpx",
    ],
)
def test_package_import(package_name: str):
    """Verify that all core environment dependencies can be imported cleanly."""
    module = importlib.import_module(package_name)
    assert module is not None
