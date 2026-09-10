"""Pydantic V2 response and request schemas for Video Analysis."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.prediction import BoundingBoxSchema


class VideoMetadataSchema(BaseModel):
    """Extracted video stream properties and configuration."""

    model_config = ConfigDict(from_attributes=True)

    filename: str = Field(..., description="Uploaded video filename")
    duration_seconds: float = Field(..., description="Total duration of video in seconds")
    source_fps: float = Field(..., description="Source frame rate (FPS) of original video stream")
    analysis_fps: float = Field(..., description="Sampling rate (FPS) evaluated by inference pipeline")
    width: int = Field(..., description="Video frame width in pixels")
    height: int = Field(..., description="Video frame height in pixels")
    total_frames: int = Field(..., description="Total source frame count in container")
    frames_analyzed: int = Field(..., description="Number of sampled frames analyzed")


class VideoTrackSchema(BaseModel):
    """Tracked face spatial lifecycle descriptor."""

    model_config = ConfigDict(from_attributes=True)

    track_id: int = Field(..., description="Persistent face track identifier across video")
    first_seen_sec: float = Field(..., description="Timestamp in seconds when face was first detected")
    last_seen_sec: float = Field(..., description="Timestamp in seconds when face was last detected")
    total_detections: int = Field(..., description="Total frame detections associated with this face track")


class ExpressionSegmentSchema(BaseModel):
    """Continuous temporal interval exhibiting a stable predicted facial expression."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Unique segment UUID")
    track_id: int = Field(..., description="Track ID this segment belongs to")
    emotion: str = Field(..., description="Predicted stable facial expression category")
    start_time: float = Field(..., description="Segment start timestamp in seconds")
    end_time: float = Field(..., description="Segment end timestamp in seconds")
    duration: float = Field(..., description="Segment duration in seconds (end_time - start_time)")
    average_confidence: float = Field(..., description="Mean model confidence across segment frames")
    min_confidence: float = Field(..., description="Minimum model confidence recorded in segment")
    max_confidence: float = Field(..., description="Maximum model confidence recorded in segment")
    prediction_count: int = Field(..., description="Number of sampled predictions within segment")


class ExpressionEventSchema(BaseModel):
    """Instantaneous prediction transition event from one expression to another."""

    track_id: int = Field(..., description="Track ID where transition occurred")
    timestamp: float = Field(..., description="Video timestamp in seconds where transition was confirmed")
    from_emotion: str = Field(..., description="Prior stable facial expression")
    to_emotion: str = Field(..., description="Newly transitioned facial expression")
    confidence: float = Field(..., description="Confidence score at transition point")


class VideoPredictionItemSchema(BaseModel):
    """Fine-grained timestamped frame-level inference item."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Prediction record UUID")
    track_id: int = Field(..., description="Associated face track ID")
    timestamp: float = Field(..., description="Frame timestamp in video seconds")
    frame_index: int = Field(..., description="Video frame index in original container")
    bbox: BoundingBoxSchema = Field(..., description="Detected face bounding box")
    detection_confidence: float = Field(..., description="Face detector confidence score")
    raw_emotion: str = Field(..., description="Unsmoothed model classification")
    raw_confidence: float = Field(..., description="Unsmoothed dominant confidence")
    smoothed_emotion: str = Field(..., description="Temporally smoothed classification")
    smoothed_confidence: float = Field(..., description="Smoothed dominant confidence")
    is_uncertain: bool = Field(..., description="Flag indicating confidence is below threshold")
    probabilities: dict[str, float] = Field(..., description="Smoothed 7-class probability map")
    raw_probabilities: dict[str, float] = Field(..., description="Raw unsmoothed 7-class probability map")


class VideoDistributionItemSchema(BaseModel):
    """Expression frequency and temporal duration share."""

    emotion: str = Field(..., description="Facial expression category")
    count: int = Field(..., description="Total frame prediction count")
    percentage: float = Field(..., description="Share of total frame predictions (%)")
    time_seconds: float = Field(..., description="Total duration of segments for this expression (sec)")
    time_share_percent: float = Field(..., description="Share of total tracked face duration (%)")


class VideoAnalyticsSummarySchema(BaseModel):
    """Video-level aggregated emotion intelligence metrics."""

    duration_seconds: float = Field(..., description="Total video container duration in seconds")
    tracked_faces_count: int = Field(..., description="Number of distinct faces tracked")
    total_predictions_count: int = Field(..., description="Total face predictions generated")
    total_transitions_count: int = Field(..., description="Total confirmed expression transitions")
    dominant_expression: str = Field(..., description="Expression with largest valid time share")
    dominant_expression_time_share: float = Field(..., description="Time share percentage of dominant expression")
    average_confidence: float = Field(..., description="Overall mean confidence across all predictions")
    processing_time_seconds: float = Field(..., description="Total elapsed processing time in seconds")
    processing_ratio: float = Field(..., description="Processing time divided by video duration (e.g. 0.4x realtime)")
    analysis_fps: float = Field(..., description="Actual evaluated sampling rate")
    expression_distribution: list[VideoDistributionItemSchema] = Field(
        default_factory=list, description="Distribution of emotions by count and time share"
    )
    transition_counts: dict[str, int] = Field(
        default_factory=dict, description="Counts of specific prediction transitions (e.g. 'neutral->happy': 4)"
    )


class VideoAnalysisJobResponse(BaseModel):
    """Immediate response after initiating an asynchronous video analysis job."""

    video_id: str = Field(..., description="Unique Video Analysis UUID")
    session_id: str | None = Field(None, description="Linked AnalysisSession UUID if applicable")
    filename: str = Field(..., description="Uploaded video filename")
    status: str = Field(..., description="Current job status (QUEUED, PROCESSING)")
    current_stage: str = Field(..., description="Current processing stage")
    message: str = Field(..., description="Informational message")


class VideoAnalysisStatusResponse(BaseModel):
    """Polling response for tracking background analysis progress."""

    video_id: str = Field(..., description="Video Analysis UUID")
    status: str = Field(..., description="Job status: QUEUED, PROCESSING, COMPLETED, FAILED, CANCELLED")
    current_stage: str = Field(..., description="Current processing stage")
    progress_percent: float = Field(..., description="Progress completion percentage [0.0, 100.0]")
    frames_analyzed: int = Field(..., description="Sampled frames processed so far")
    total_frames: int = Field(..., description="Total frames in video container")
    error_message: str | None = Field(None, description="Error explanation if job failed")


class VideoTimelineResponse(BaseModel):
    """Chronological segments and transition events for timeline visualization."""

    video_id: str = Field(..., description="Video Analysis UUID")
    track_id: int | None = Field(None, description="Filter applied by track ID (None if all tracks)")
    segments: list[ExpressionSegmentSchema] = Field(
        default_factory=list, description="Chronological expression segments"
    )
    events: list[ExpressionEventSchema] = Field(
        default_factory=list, description="Chronological prediction transition events"
    )


class VideoAnalysisDetailResponse(BaseModel):
    """Comprehensive detail payload for a video analysis entity."""

    video_id: str = Field(..., description="Video Analysis UUID")
    session_id: str | None = Field(None, description="Associated session UUID")
    status: str = Field(..., description="Current job status")
    current_stage: str = Field(..., description="Processing stage")
    progress_percent: float = Field(..., description="Progress percentage")
    metadata: VideoMetadataSchema = Field(..., description="Video technical metadata")
    analytics: VideoAnalyticsSummarySchema | None = Field(
        None, description="Video-level aggregated emotion analytics"
    )
    tracks: list[VideoTrackSchema] = Field(
        default_factory=list, description="List of tracked faces"
    )
    created_at: datetime = Field(..., description="Job submission timestamp")
    started_at: datetime | None = Field(None, description="Processing start timestamp")
    completed_at: datetime | None = Field(None, description="Processing completion timestamp")
    error_message: str | None = Field(None, description="Failure description if failed")


class VideoPredictionsListResponse(BaseModel):
    """Paginated frame-level predictions response."""

    video_id: str = Field(..., description="Video Analysis UUID")
    track_id: int | None = Field(None, description="Filtered track ID if applied")
    total: int = Field(..., description="Total matching prediction count")
    page: int = Field(..., description="Current page number (1-based)")
    page_size: int = Field(..., description="Current page size limit")
    predictions: list[VideoPredictionItemSchema] = Field(
        default_factory=list, description="List of timestamped frame predictions"
    )
