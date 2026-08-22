"""Service layer for analysis session management."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ResourceNotFoundError
from app.db.repositories.analysis_session import SessionRepository
from app.db.repositories.prediction import PredictionRepository
from app.schemas.prediction import (
    BoundingBoxSchema,
    FacePredictionSchema,
    PredictionDetailSchema,
)
from app.schemas.session import (
    SessionCreateRequest,
    SessionListResponse,
    SessionResponse,
)

logger = logging.getLogger(__name__)


class SessionService:
    """Manages analysis sessions and retrieves session prediction histories."""

    def __init__(
        self,
        session_repo: SessionRepository | None = None,
        prediction_repo: PredictionRepository | None = None,
    ) -> None:
        self.session_repo = session_repo or SessionRepository()
        self.prediction_repo = prediction_repo or PredictionRepository()

    async def create_session(
        self,
        db: AsyncSession,
        request: SessionCreateRequest,
    ) -> SessionResponse:
        """Create and persist a new session."""
        session_record = await self.session_repo.create(
            session=db,
            user_id=request.user_id,
            name=request.name,
        )
        await db.commit()
        return SessionResponse.model_validate(session_record)

    async def get_session(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
    ) -> SessionResponse:
        """Fetch session by UUID or raise ResourceNotFoundError."""
        session_record = await self.session_repo.get_by_id(db, session_id)
        if session_record is None:
            raise ResourceNotFoundError(
                message=f"Session with ID '{session_id}' not found",
                details={"session_id": str(session_id)},
            )
        return SessionResponse.model_validate(session_record)

    async def end_session(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
    ) -> SessionResponse:
        """Mark an active session as completed."""
        session_record = await self.session_repo.end_session(db, session_id, status="completed")
        if session_record is None:
            raise ResourceNotFoundError(
                message=f"Session with ID '{session_id}' not found",
                details={"session_id": str(session_id)},
            )
        await db.commit()
        return SessionResponse.model_validate(session_record)

    async def list_sessions(
        self,
        db: AsyncSession,
        user_id: uuid.UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> SessionListResponse:
        """List sessions with pagination."""
        sessions = await self.session_repo.list_sessions(
            db, user_id=user_id, limit=limit, offset=offset
        )
        items = [SessionResponse.model_validate(s) for s in sessions]
        return SessionListResponse(sessions=items, total=len(items))

    async def get_session_predictions(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PredictionDetailSchema]:
        """Fetch all stored predictions for a given session."""
        # Verify session exists
        await self.get_session(db, session_id)

        records = await self.prediction_repo.list_by_session_id(
            db, session_id=session_id, limit=limit, offset=offset, load_faces=True
        )

        results: list[PredictionDetailSchema] = []
        for r in records:
            faces = [
                FacePredictionSchema(
                    face_id=f.face_id,
                    bbox=BoundingBoxSchema(
                        x=f.bbox_x,
                        y=f.bbox_y,
                        width=f.bbox_width,
                        height=f.bbox_height,
                    ),
                    detection_confidence=f.detection_confidence,
                    emotion=f.emotion,
                    confidence=f.confidence,
                    is_uncertain=f.is_uncertain,
                    probabilities=f.probabilities,
                )
                for f in r.faces
            ]
            results.append(
                PredictionDetailSchema(
                    id=r.id,
                    session_id=r.session_id,
                    request_id=r.request_id,
                    model_version=r.model_version,
                    status=r.status,
                    faces_detected=r.faces_detected,
                    processing_time_ms=r.processing_time_ms,
                    image_width=r.image_width,
                    image_height=r.image_height,
                    created_at=r.created_at,
                    faces=faces,
                )
            )

        return results
