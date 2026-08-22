"""SQLAlchemy domain model for PredictionRecord entity."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.analysis_session import AnalysisSession
    from app.db.models.detected_face import DetectedFaceRecord


class PredictionRecord(Base):
    """Metadata record for an image inference event."""

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    request_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    model_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )  # SUCCESS, NO_FACE_DETECTED, INVALID_IMAGE, ERROR
    faces_detected: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    processing_time_ms: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )
    image_width: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    image_height: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # Relationships
    session: Mapped[AnalysisSession | None] = relationship(
        "AnalysisSession",
        back_populates="predictions",
    )
    faces: Mapped[list[DetectedFaceRecord]] = relationship(
        "DetectedFaceRecord",
        back_populates="prediction",
        cascade="all, delete-orphan",
        order_by="DetectedFaceRecord.face_id.asc()",
    )
