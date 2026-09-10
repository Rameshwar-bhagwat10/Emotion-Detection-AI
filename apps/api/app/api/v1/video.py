"""Video Analysis API endpoints for upload, asynchronous processing, status, timeline, and streaming."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from fastapi.responses import FileResponse

from app.api.dependencies import get_video_service
from app.schemas.video_analysis import (
    VideoAnalysisDetailResponse,
    VideoAnalysisJobResponse,
    VideoAnalysisStatusResponse,
    VideoPredictionsListResponse,
    VideoTimelineResponse,
)
from app.services.video_service import VideoService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video", tags=["Video Analysis"])


@router.post(
    "/upload",
    response_model=VideoAnalysisJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload video and initiate temporal facial expression analysis",
)
@router.post(
    "/analyze",
    response_model=VideoAnalysisJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Alias for /upload",
)
async def upload_video_for_analysis(
    video: UploadFile = File(..., description="Uploaded video binary container (.mp4, .webm, .mov, etc.)"),
    sampling_fps: float = Form(default=5.0, ge=1.0, le=30.0, description="Inference sampling rate (FPS)"),
    session_id: uuid.UUID | None = Form(default=None, description="Optional associated session UUID"),
    video_service: VideoService = Depends(get_video_service),
) -> VideoAnalysisJobResponse:
    """Ingest uploaded video, validate container properties, and start asynchronous temporal analysis."""
    file_bytes = await video.read()
    filename = video.filename or "uploaded_video.mp4"

    return await video_service.submit_analysis_job(
        file_bytes=file_bytes,
        filename=filename,
        sampling_fps=sampling_fps,
        session_id=session_id,
    )


@router.get(
    "/recent",
    summary="List recently analyzed videos",
)
async def list_recent_videos(
    limit: int = Query(default=20, ge=1, le=100, description="Max number of records to return"),
    video_service: VideoService = Depends(get_video_service),
) -> list[dict[str, Any]]:
    """Retrieve recently analyzed videos for historical selection and replay."""
    return await video_service.get_recent_videos(limit=limit)


@router.get(
    "/{video_id}/status",

    response_model=VideoAnalysisStatusResponse,
    summary="Poll processing status and progress for a video analysis job",
)
async def get_video_analysis_status(
    video_id: uuid.UUID,
    video_service: VideoService = Depends(get_video_service),
) -> VideoAnalysisStatusResponse:
    """Retrieve current execution status, stage, and completion percentage."""
    return await video_service.get_status(video_id=video_id)


@router.get(
    "/{video_id}",
    response_model=VideoAnalysisDetailResponse,
    summary="Retrieve detailed video analysis report and aggregated analytics",
)
async def get_video_analysis_detail(
    video_id: uuid.UUID,
    video_service: VideoService = Depends(get_video_service),
) -> Any:
    """Retrieve full video metadata, tracked faces, and comprehensive video-level analytics."""
    detail = await video_service.get_detail(video_id=video_id)
    return detail


@router.get(
    "/{video_id}/timeline",
    response_model=VideoTimelineResponse,
    summary="Retrieve expression segments and transition events for timeline visualization",
)
async def get_video_timeline(
    video_id: uuid.UUID,
    track_id: int | None = Query(None, description="Filter segments by specific face track ID"),
    video_service: VideoService = Depends(get_video_service),
) -> VideoTimelineResponse:
    """Retrieve chronological expression segments and prediction transitions for timeline display."""
    segments, events = await video_service.get_timeline(video_id=video_id, track_id=track_id)
    return VideoTimelineResponse(
        video_id=str(video_id),
        track_id=track_id,
        segments=segments,
        events=events,
    )


@router.get(
    "/{video_id}/predictions",
    response_model=VideoPredictionsListResponse,
    summary="Query paginated frame-level predictions for a video",
)
async def get_video_predictions(
    video_id: uuid.UUID,
    track_id: int | None = Query(None, description="Filter predictions by face track ID"),
    start_time: float | None = Query(None, ge=0.0, description="Start timestamp in seconds"),
    end_time: float | None = Query(None, ge=0.0, description="End timestamp in seconds"),
    page: int = Query(1, ge=1, description="1-based page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Number of items per page"),
    video_service: VideoService = Depends(get_video_service),
) -> VideoPredictionsListResponse:
    """Retrieve paginated frame-level predictions with bounding boxes and 7-class probability distributions."""
    items, total = await video_service.get_predictions(
        video_id=video_id,
        track_id=track_id,
        start_time=start_time,
        end_time=end_time,
        page=page,
        page_size=page_size,
    )
    return VideoPredictionsListResponse(
        video_id=str(video_id),
        track_id=track_id,
        total=total,
        page=page,
        page_size=page_size,
        predictions=items,
    )


@router.post(
    "/{video_id}/cancel",
    summary="Cancel an ongoing video analysis job",
)
async def cancel_video_analysis(
    video_id: uuid.UUID,
    video_service: VideoService = Depends(get_video_service),
) -> dict[str, str]:
    """Request graceful cancellation of a running video analysis."""
    cancelled = await video_service.cancel_job(video_id=video_id)
    return {
        "video_id": str(video_id),
        "status": "cancelled" if cancelled else "already_finished_or_not_found",
    }


@router.get(
    "/{video_id}/stream",
    summary="Stream video file with HTTP Range support for video player seeking",
)
async def stream_video_file(
    video_id: uuid.UUID,
    video_service: VideoService = Depends(get_video_service),
) -> FileResponse:
    """Serve the stored video file with byte-range seeking for browser video players."""
    file_path = await video_service.get_video_file_path(video_id=video_id)
    ext = file_path.suffix.lower()
    media_type = "video/mp4"
    if ext == ".webm":
        media_type = "video/webm"
    elif ext == ".mov":
        media_type = "video/quicktime"
    elif ext == ".mkv":
        media_type = "video/x-matroska"

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name,
    )
