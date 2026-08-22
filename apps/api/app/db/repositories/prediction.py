"""Repository for PredictionRecord and DetectedFaceRecord database operations."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.detected_face import DetectedFaceRecord
from app.db.models.prediction import PredictionRecord


class PredictionRepository:
    """Data-access repository for PredictionRecord and DetectedFaceRecord domain models."""

    async def get_by_id(
        self, session: AsyncSession, prediction_id: uuid.UUID, load_faces: bool = True
    ) -> PredictionRecord | None:
        """Fetch a PredictionRecord by UUID."""
        stmt = select(PredictionRecord).where(PredictionRecord.id == prediction_id)
        if load_faces:
            stmt = stmt.options(selectinload(PredictionRecord.faces))
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_request_id(
        self, session: AsyncSession, request_id: str, load_faces: bool = True
    ) -> PredictionRecord | None:
        """Fetch a PredictionRecord by unique request_id."""
        stmt = select(PredictionRecord).where(PredictionRecord.request_id == request_id)
        if load_faces:
            stmt = stmt.options(selectinload(PredictionRecord.faces))
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_session_id(
        self,
        session: AsyncSession,
        session_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
        load_faces: bool = True,
    ) -> list[PredictionRecord]:
        """List all predictions belonging to a specific session."""
        stmt = (
            select(PredictionRecord)
            .where(PredictionRecord.session_id == session_id)
            .order_by(PredictionRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if load_faces:
            stmt = stmt.options(selectinload(PredictionRecord.faces))
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def create_prediction_with_faces(
        self,
        session: AsyncSession,
        prediction_id: uuid.UUID | None,
        request_id: str,
        model_version: str,
        status: str,
        faces_detected: int,
        processing_time_ms: float,
        image_width: int | None,
        image_height: int | None,
        session_id: uuid.UUID | None = None,
        faces_data: list[dict[str, Any]] | None = None,
    ) -> PredictionRecord:
        """Transactionally persist a prediction event along with all detected face records."""
        pred = PredictionRecord(
            id=prediction_id or uuid.uuid4(),
            session_id=session_id,
            request_id=request_id,
            model_version=model_version,
            status=status,
            faces_detected=faces_detected,
            processing_time_ms=processing_time_ms,
            image_width=image_width,
            image_height=image_height,
        )
        session.add(pred)
        await session.flush()

        if faces_data:
            for item in faces_data:
                face_record = DetectedFaceRecord(
                    id=uuid.uuid4(),
                    prediction_id=pred.id,
                    face_id=item["face_id"],
                    bbox_x=item["bbox_x"],
                    bbox_y=item["bbox_y"],
                    bbox_width=item["bbox_width"],
                    bbox_height=item["bbox_height"],
                    detection_confidence=item["detection_confidence"],
                    emotion=item["emotion"],
                    confidence=item["confidence"],
                    is_uncertain=item.get("is_uncertain", False),
                    probabilities=item["probabilities"],
                )
                session.add(face_record)

        await session.flush()
        return pred
