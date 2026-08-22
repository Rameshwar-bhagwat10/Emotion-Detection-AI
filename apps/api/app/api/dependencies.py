"""FastAPI dependency injection providers."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ModelNotReadyError
from app.db.session import AsyncSessionLocal
from app.services.prediction_service import PredictionService
from app.services.session_service import SessionService
from ml.inference.engine import EmotionInferenceEngine


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide an asynchronous database session with automatic transaction management."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_request_id(request: Request) -> str:
    """Retrieve the unique request ID assigned by RequestIDMiddleware."""
    return getattr(request.state, "request_id", str(uuid.uuid4()))


def get_inference_engine(request: Request) -> EmotionInferenceEngine:
    """Retrieve the initialized Phase 09 EmotionInferenceEngine from app.state."""
    engine: EmotionInferenceEngine | None = getattr(request.app.state, "inference_engine", None)
    if engine is None:
        raise ModelNotReadyError(
            message="Emotion inference engine is not initialized or still starting up.",
            details={"status": "not_ready"},
        )
    return engine


def get_prediction_service(
    engine: EmotionInferenceEngine = Depends(get_inference_engine),
) -> PredictionService:
    """Provide PredictionService with injected Phase 09 inference engine."""
    return PredictionService(inference_engine=engine)


def get_session_service() -> SessionService:
    """Provide SessionService instance."""
    return SessionService()
