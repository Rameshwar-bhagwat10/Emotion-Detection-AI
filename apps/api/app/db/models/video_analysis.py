"""SQLAlchemy domain models for Video Analysis, Tracks, Segments, and Predictions."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.analysis_session import AnalysisSession


class VideoAnalysisRecord(Base):
    """Database record representing an uploaded video analysis job and its metadata."""

    __tablename__ = "video_analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    file_path: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="QUEUED",
        nullable=False,
        index=True,
    )  # QUEUED, PROCESSING, COMPLETED, FAILED, CANCELLED
    current_stage: Mapped[str] = mapped_column(
        String(50),
        default="queued",
        nullable=False,
    )  # queued, decoding, processing, generating_timeline, saving, completed, failed, cancelled
    progress_percent: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )
    duration_seconds: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )
    source_fps: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )
    analysis_fps: Mapped[float] = mapped_column(
        Float,
        default=5.0,
        nullable=False,
    )
    width: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    height: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    total_frames: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    frames_analyzed: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    processing_time_seconds: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    session: Mapped[AnalysisSession | None] = relationship(
        "AnalysisSession",
        foreign_keys=[session_id],
    )
    tracks: Mapped[list[VideoTrackRecord]] = relationship(
        "VideoTrackRecord",
        back_populates="video_analysis",
        cascade="all, delete-orphan",
        order_by="VideoTrackRecord.track_id",
    )
    segments: Mapped[list[ExpressionSegmentRecord]] = relationship(
        "ExpressionSegmentRecord",
        back_populates="video_analysis",
        cascade="all, delete-orphan",
        order_by="ExpressionSegmentRecord.start_time",
    )
    predictions: Mapped[list[VideoPredictionRecord]] = relationship(
        "VideoPredictionRecord",
        back_populates="video_analysis",
        cascade="all, delete-orphan",
        order_by="VideoPredictionRecord.timestamp",
    )


class VideoTrackRecord(Base):
    """Metadata for a unique face tracked throughout video playback."""

    __tablename__ = "video_tracks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    video_analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("video_analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    track_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )
    first_seen_sec: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    last_seen_sec: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    total_detections: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    # Relationships
    video_analysis: Mapped[VideoAnalysisRecord] = relationship(
        "VideoAnalysisRecord",
        back_populates="tracks",
    )


class ExpressionSegmentRecord(Base):
    """Chronological temporal segment during which a face exhibited a stable predicted expression."""

    __tablename__ = "expression_segments"
    __table_args__ = (
        Index("idx_video_segments_track_time", "video_analysis_id", "track_id", "start_time"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    video_analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("video_analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    track_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )
    emotion: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    start_time: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        index=True,
    )
    end_time: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    duration: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    average_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    min_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    max_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    prediction_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    # Relationships
    video_analysis: Mapped[VideoAnalysisRecord] = relationship(
        "VideoAnalysisRecord",
        back_populates="segments",
    )


class VideoPredictionRecord(Base):
    """Fine-grained timestamped frame-level prediction for a tracked face."""

    __tablename__ = "video_predictions"
    __table_args__ = (
        Index("idx_video_preds_timestamp", "video_analysis_id", "track_id", "timestamp"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    video_analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("video_analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    track_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )
    timestamp: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        index=True,
    )
    frame_index: Mapped[int] = mapped_column(
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
    raw_emotion: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    raw_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    smoothed_emotion: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    smoothed_confidence: Mapped[float] = mapped_column(
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
    raw_probabilities: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
    )

    # Relationships
    video_analysis: Mapped[VideoAnalysisRecord] = relationship(
        "VideoAnalysisRecord",
        back_populates="predictions",
    )
