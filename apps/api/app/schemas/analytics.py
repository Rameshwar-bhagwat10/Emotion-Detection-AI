"""Pydantic schemas for Phase 13 Analytics, Session Metrics, Timeline, and History."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExpressionCountSchema(BaseModel):
    """Frequency and relative percentage for a single emotion category."""

    emotion: str = Field(description="Normalized emotion label (e.g. happy, neutral)")
    count: int = Field(ge=0, description="Total number of valid predictions in this class")
    percentage: float = Field(
        ge=0.0, le=100.0, description="Proportion of total predictions formatted as percentage"
    )


class ExpressionDistributionSchema(BaseModel):
    """Complete 7-class emotion distribution with dominance metadata."""

    items: list[ExpressionCountSchema] = Field(
        description="Distribution items across all supported emotion classes"
    )
    dominant_emotion: str | None = Field(
        default=None, description="Expression with the highest count, resolved deterministically"
    )
    total_predictions: int = Field(
        ge=0, description="Total face predictions included in the distribution"
    )


class ConfidenceBucketSchema(BaseModel):
    """A histogram bin representing a confidence score interval."""

    range: str = Field(description="Display label, e.g. 60-80%")
    min_val: float = Field(ge=0.0, le=1.0)
    max_val: float = Field(ge=0.0, le=1.0)
    count: int = Field(ge=0, description="Number of predictions falling in this range")
    percentage: float = Field(ge=0.0, le=100.0, description="Percentage of predictions in range")


class ConfidenceAnalyticsSchema(BaseModel):
    """Summary confidence statistics and distribution histogram."""

    average_confidence: float = Field(ge=0.0, le=1.0, description="Mean confidence score")
    min_confidence: float = Field(ge=0.0, le=1.0, description="Minimum confidence recorded")
    max_confidence: float = Field(ge=0.0, le=1.0, description="Maximum confidence recorded")
    distribution: list[ConfidenceBucketSchema] = Field(
        description="5-bin confidence histogram breakdown"
    )
    class_confidences: dict[str, float] = Field(
        default_factory=dict, description="Average confidence score per emotion class"
    )
    low_confidence_count: int = Field(
        ge=0, description="Count of predictions below threshold or marked uncertain"
    )
    high_confidence_count: int = Field(
        ge=0, description="Count of predictions with confidence >= 0.80"
    )


class TimelineBucketSchema(BaseModel):
    """Aggregated prediction statistics for a temporal interval in a session."""

    bucket_index: int = Field(ge=0, description="Sequential bucket index")
    timestamp: str = Field(description="ISO-8601 formatted timestamp of the bucket start")
    relative_seconds: float = Field(
        ge=0.0, description="Seconds elapsed since session start"
    )
    prediction_count: int = Field(ge=0, description="Predictions recorded in this time window")
    emotion_counts: dict[str, int] = Field(
        description="Breakdown of predictions per emotion in this bucket"
    )
    dominant_emotion: str | None = Field(
        default=None, description="Dominant emotion in this bucket"
    )
    average_confidence: float = Field(
        ge=0.0, le=1.0, description="Average confidence for predictions in this bucket"
    )


class TimelineAnalyticsResponse(BaseModel):
    """Complete time-series timeline response for a session."""

    session_id: uuid.UUID = Field(description="Target session identifier")
    bucket_seconds: int = Field(gt=0, description="Time interval width in seconds")
    total_buckets: int = Field(ge=0, description="Total number of intervals generated")
    buckets: list[TimelineBucketSchema] = Field(description="Chronological time buckets")


class DailyTrendItem(BaseModel):
    """Daily aggregated activity item."""

    date: str = Field(description="Date string YYYY-MM-DD")
    predictions_count: int = Field(ge=0, description="Total predictions recorded on this date")
    sessions_count: int = Field(ge=0, description="Distinct sessions active on this date")


class GlobalAnalyticsResponse(BaseModel):
    """Global system-wide analytics overview."""

    total_sessions: int = Field(ge=0, description="Total analysis sessions recorded")
    total_predictions: int = Field(ge=0, description="Total image inference events")
    total_faces: int = Field(ge=0, description="Total detected face predictions")
    total_duration_seconds: float = Field(
        ge=0.0, description="Cumulative detection duration across completed sessions"
    )
    dominant_expression: str | None = Field(
        default=None, description="Global most frequent predicted expression"
    )
    average_confidence: float = Field(
        ge=0.0, le=1.0, description="Global average prediction confidence"
    )
    expression_distribution: ExpressionDistributionSchema = Field(
        description="Global 7-class emotion distribution"
    )
    confidence_analytics: ConfidenceAnalyticsSchema = Field(
        description="Global confidence metrics and histogram"
    )
    model_version_distribution: dict[str, int] = Field(
        default_factory=dict, description="Count of predictions per model version"
    )
    recent_trends: list[DailyTrendItem] = Field(
        default_factory=list, description="Recent 30-day activity trend"
    )


class SessionAnalyticsResponse(BaseModel):
    """Session-level analytics and telemetry metrics."""

    session_id: uuid.UUID = Field(description="Session UUID")
    name: str | None = Field(default=None, description="Friendly session name")
    status: str = Field(description="Session state: active, completed, cancelled")
    started_at: datetime = Field(description="Session start timestamp")
    ended_at: datetime | None = Field(default=None, description="Session end timestamp")
    duration_seconds: float = Field(ge=0.0, description="Session elapsed time in seconds")
    total_predictions: int = Field(ge=0, description="Number of inference events in this session")
    total_faces: int = Field(ge=0, description="Number of face instances detected")
    prediction_rate_per_minute: float = Field(
        ge=0.0, description="Average inference rate (predictions/minute)"
    )
    dominant_expression: str | None = Field(
        default=None, description="Dominant emotion observed during this session"
    )
    expression_distribution: ExpressionDistributionSchema = Field(
        description="Emotion distribution for this session"
    )
    confidence_analytics: ConfidenceAnalyticsSchema = Field(
        description="Session-specific confidence metrics and histogram"
    )
    model_versions: list[str] = Field(
        default_factory=list, description="Model versions used in this session"
    )


class SessionSummaryItem(BaseModel):
    """Enriched session overview card item for session history lists."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(description="Unique session UUID")
    name: str | None = Field(default=None, description="Session name")
    status: str = Field(description="Session state")
    started_at: datetime = Field(description="Session start timestamp")
    ended_at: datetime | None = Field(default=None, description="Session end timestamp")
    duration_seconds: float = Field(ge=0.0, description="Session duration in seconds")
    prediction_count: int = Field(ge=0, description="Total face predictions recorded in session")
    dominant_expression: str | None = Field(
        default=None, description="Dominant predicted emotion"
    )
    average_confidence: float = Field(
        ge=0.0, le=1.0, description="Average prediction confidence"
    )


class SessionAnalyticsListResponse(BaseModel):
    """Paginated list of sessions enriched with analytics summaries."""

    sessions: list[SessionSummaryItem] = Field(description="List of enriched session summaries")
    total: int = Field(ge=0, description="Total count of sessions matching criteria")


class HistoricalPredictionItem(BaseModel):
    """Individual face prediction record for history tables and inspect modal."""

    prediction_id: uuid.UUID = Field(description="Prediction record UUID")
    session_id: uuid.UUID | None = Field(default=None, description="Associated session UUID")
    request_id: str = Field(description="Inference correlation request ID")
    timestamp: datetime = Field(description="Prediction timestamp")
    model_version: str = Field(description="Model version used")
    face_id: int = Field(ge=1, description="Face index within frame")
    emotion: str = Field(description="Predicted emotion class")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")
    is_uncertain: bool = Field(default=False, description="Flag for low-confidence inference")
    bbox: dict[str, int] = Field(description="Face bounding box {x, y, width, height}")
    probabilities: dict[str, float] = Field(
        default_factory=dict, description="Softmax probabilities across all 7 emotion classes"
    )
    processing_time_ms: float = Field(ge=0.0, description="Inference latency in milliseconds")


class PaginatedPredictionsResponse(BaseModel):
    """Paginated historical prediction list."""

    items: list[HistoricalPredictionItem] = Field(description="List of prediction records")
    total: int = Field(ge=0, description="Total records matching filter criteria")
    limit: int = Field(gt=0, description="Page limit")
    offset: int = Field(ge=0, description="Page offset")
