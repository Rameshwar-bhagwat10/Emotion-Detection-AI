"""SQLAlchemy domain model for DetectedFaceRecord entity."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Boolean, CheckConstraint, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.prediction import PredictionRecord


class DetectedFaceRecord(Base):
    """Metadata and emotion classification record for a detected face bounding box."""

    __tablename__ = "face_predictions"
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0", name="chk_face_confidence_range"
        ),
        CheckConstraint(
            "detection_confidence >= 0.0 AND detection_confidence <= 1.0",
            name="chk_face_det_confidence_range",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    face_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    bbox_x: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    bbox_y: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    bbox_width: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    bbox_height: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    detection_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    emotion: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    is_uncertain: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    probabilities: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
    )

    # Relationships
    prediction: Mapped[PredictionRecord] = relationship(
        "PredictionRecord",
        back_populates="faces",
    )
