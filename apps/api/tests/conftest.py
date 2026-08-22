"""Pytest test configuration and fixtures for FastAPI backend."""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import AsyncGenerator

import cv2
import numpy as np
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import app.db.models  # noqa: F401
from app.api.dependencies import get_db, get_inference_engine
from app.db.base import Base
from app.main import create_application
from ml.inference.config import FaceDetectionConfig, InferencePipelineConfig
from ml.inference.engine import EmotionInferenceEngine


@pytest_asyncio.fixture
async def test_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Provide an async engine connected to an isolated test SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db_url = f"sqlite+aiosqlite:///{db_path.replace(chr(92), '/')}"
    engine = create_async_engine(
        db_url,
        connect_args={"check_same_thread": False},
        echo=False,
        future=True,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except Exception:
        pass


@pytest_asyncio.fixture
async def db_session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Provide a function-scoped database session."""
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture(scope="session")
def inference_engine() -> EmotionInferenceEngine:
    """Provide a shared Phase 09 EmotionInferenceEngine with YuNet/PassThrough detectors."""
    config = InferencePipelineConfig(
        face_detection=FaceDetectionConfig(
            detector_type="yunet",
            confidence_threshold=0.30,
        )
    )
    engine = EmotionInferenceEngine(config=config)
    engine.warm_up(num_warmup_passes=1)
    return engine


@pytest.fixture
def app_instance(test_engine: AsyncEngine, inference_engine: EmotionInferenceEngine) -> FastAPI:
    """Create FastAPI application configured with test database and shared inference engine."""
    app = create_application()
    app.state.inference_engine = inference_engine
    app.state.is_ready = True

    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    # Override get_db dependency to yield clean session per request
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    def override_get_engine() -> EmotionInferenceEngine:
        return inference_engine

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_inference_engine] = override_get_engine
    return app


@pytest_asyncio.fixture
async def client(app_instance: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Provide an AsyncClient for interacting with the FastAPI app."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_face_image_bytes() -> bytes:
    """Generate a synthetic face-like RGB image and return PNG bytes."""
    # Create 300x300 canvas with simulated face oval and eyes
    img = np.full((300, 300, 3), 200, dtype=np.uint8)
    # Face oval
    cv2.ellipse(img, (150, 150), (80, 110), 0, 0, 360, (240, 210, 190), -1)
    # Eyes
    cv2.circle(img, (120, 120), 12, (50, 30, 20), -1)
    cv2.circle(img, (180, 120), 12, (50, 30, 20), -1)
    # Nose
    cv2.line(img, (150, 130), (150, 160), (180, 140, 120), 4)
    # Mouth (Smile)
    cv2.ellipse(img, (150, 185), (35, 18), 0, 0, 180, (160, 40, 40), 4)

    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_blank_image_bytes() -> bytes:
    """Generate a blank 200x200 image with no face and return PNG bytes."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_corrupt_image_bytes() -> bytes:
    """Return invalid/corrupted image bytes."""
    return b"CORRUPTED_NOT_AN_IMAGE_DATA_12345"
