"""FastAPI dependency injection providers."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ModelNotReadyError
from app.db.session import get_session_factory
from app.services.analytics_service import AnalyticsService
from app.services.prediction_service import PredictionService
from app.services.session_service import SessionService
from app.services.video_service import VideoService
from ml.inference.engine import EmotionInferenceEngine


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide an asynchronous database session with automatic transaction management."""
    factory = get_session_factory()
    async with factory() as session:
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
    """Retrieve the initialized Phase 09 EmotionInferenceEngine from app.state with lazy fallback."""
    engine: EmotionInferenceEngine | None = getattr(request.app.state, "inference_engine", None)
    if engine is None:
        try:
            from ml.inference.config import InferencePipelineConfig
            pipeline_config = InferencePipelineConfig()
            engine = EmotionInferenceEngine(config=pipeline_config)
            engine.warm_up(num_warmup_passes=1)
            request.app.state.inference_engine = engine
            request.app.state.is_ready = True
            request.app.state.model_error = None
        except Exception as exc:
            request.app.state.model_error = str(exc)
            raise ModelNotReadyError(
                message=f"Emotion inference engine initialization failed: {exc}",
                details={"status": "not_ready", "error": str(exc)},
            ) from exc
    return engine


def get_prediction_service(
    engine: EmotionInferenceEngine = Depends(get_inference_engine),
) -> PredictionService:
    """Provide PredictionService with injected Phase 09 inference engine."""
    return PredictionService(inference_engine=engine)


def get_session_service() -> SessionService:
    """Provide SessionService instance."""
    return SessionService()


def get_analytics_service() -> AnalyticsService:
    """Provide AnalyticsService instance."""
    return AnalyticsService()


def get_video_service(
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> VideoService:
    """Provide VideoService instance with injected PredictionService and async session factory."""
    return VideoService(
        prediction_service=prediction_service,
        session_factory=get_session_factory(),
    )
