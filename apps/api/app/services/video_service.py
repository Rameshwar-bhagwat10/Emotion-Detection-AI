"""Video Analysis service orchestrating streaming frame sampling, multi-face tracking,
production inference, temporal smoothing, expression segmentation, and database persistence.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.core.exceptions import (
    InvalidImageError,
    PayloadTooLargeError,
    ResourceNotFoundError,
    UnsupportedMediaTypeError,
)
from app.db.repositories.video_analysis import VideoAnalysisRepository
from app.schemas.prediction import BoundingBoxSchema
from app.schemas.video_analysis import (
    ExpressionEventSchema,
    ExpressionSegmentSchema,
    VideoAnalysisJobResponse,
    VideoAnalysisStatusResponse,
    VideoAnalyticsSummarySchema,
    VideoMetadataSchema,
    VideoTrackSchema,
)
from app.services.face_tracker import FaceTracker
from app.services.prediction_service import PredictionService
from app.services.temporal_smoother import TemporalSmoother
from ml.inference.schemas import InferenceStatus

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_VIDEO_DURATION_SECONDS = 600.0  # 10 minutes


class VideoService:
    """Production Video Analysis service for temporal facial expression processing."""

    def __init__(
        self,
        prediction_service: PredictionService,
        session_factory: async_sessionmaker[AsyncSession],
        video_repo: VideoAnalysisRepository | None = None,
        storage_dir: Path | str | None = None,
    ) -> None:
        """Initialize VideoService.

        Args:
            prediction_service: Injected PredictionService holding Champion model engine.
            session_factory: AsyncSession factory for background task database transactions.
            video_repo: Video analysis repository.
            storage_dir: Base directory for storing uploaded video files.
        """
        self.prediction_service = prediction_service
        self.session_factory = session_factory
        self.repo = video_repo or VideoAnalysisRepository()

        if storage_dir is None:
            self.storage_dir = Path(__file__).resolve().parent.parent.parent / "storage" / "videos"
        else:
            self.storage_dir = Path(storage_dir)

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory tracking of active tasks and cancellation flags
        self._active_tasks: dict[uuid.UUID, asyncio.Task[Any]] = {}
        self._cancellation_flags: set[uuid.UUID] = set()

    def validate_and_save_upload(
        self,
        file_bytes: bytes,
        filename: str,
        video_id: uuid.UUID,
    ) -> tuple[Path, dict[str, Any]]:
        """Validate uploaded video container, decode properties, and write to storage.

        Args:
            file_bytes: Raw binary video file bytes.
            filename: Original uploaded filename.
            video_id: Unique UUID assigned to this analysis.

        Returns:
            Tuple of (persisted_file_path, extracted_metadata_dict).
        """
        # 1. Payload size check
        if len(file_bytes) > MAX_VIDEO_SIZE_BYTES:
            raise PayloadTooLargeError(
                message=f"Video file size ({len(file_bytes) / (1024 * 1024):.1f}MB) exceeds limit of 100MB.",
                details={"max_size_mb": 100, "actual_bytes": len(file_bytes)},
            )

        # 2. Extension check
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_VIDEO_EXTENSIONS:
            raise UnsupportedMediaTypeError(
                message=f"Video format '{ext}' is unsupported. Allowed: {sorted(list(SUPPORTED_VIDEO_EXTENSIONS))}",
                details={"allowed_extensions": sorted(list(SUPPORTED_VIDEO_EXTENSIONS)), "received": ext},
            )

        # 3. Write temporarily to verify decodability
        temp_path = self.storage_dir / f"{video_id}{ext}"
        try:
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
        except Exception as exc:
            logger.error(f"Failed to write video bytes to {temp_path}: {exc}")
            raise

        # 4. Decodability and metadata validation via OpenCV
        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            if temp_path.exists():
                temp_path.unlink()
            raise InvalidImageError(
                message="Video file could not be decoded. The stream may be corrupted or use an unsupported codec.",
                details={"filename": filename},
            )

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # Sanity fallbacks
        if source_fps <= 0.0 or source_fps > 120.0:
            source_fps = 30.0

        duration_seconds = total_frames / source_fps if total_frames > 0 else 0.0

        # Read first test frame
        ret, test_frame = cap.read()
        cap.release()

        if not ret or test_frame is None:
            if temp_path.exists():
                temp_path.unlink()
            raise InvalidImageError(
                message="Video container contains no readable frames.",
                details={"filename": filename},
            )

        del test_frame

        if duration_seconds > MAX_VIDEO_DURATION_SECONDS:
            if temp_path.exists():
                temp_path.unlink()
            raise PayloadTooLargeError(
                message=f"Video duration ({duration_seconds:.1f}s) exceeds maximum allowed duration of {MAX_VIDEO_DURATION_SECONDS}s.",
                details={"duration_seconds": duration_seconds, "max_seconds": MAX_VIDEO_DURATION_SECONDS},
            )

        metadata = {
            "filename": filename,
            "duration_seconds": round(duration_seconds, 2),
            "source_fps": round(source_fps, 2),
            "width": width,
            "height": height,
            "total_frames": total_frames,
        }

        return temp_path, metadata

    async def submit_analysis_job(
        self,
        file_bytes: bytes,
        filename: str,
        sampling_fps: float = 5.0,
        session_id: uuid.UUID | None = None,
    ) -> VideoAnalysisJobResponse:
        """Create job record and launch asynchronous video processing."""
        video_id = uuid.uuid4()
        sampling_fps = max(1.0, min(30.0, float(sampling_fps)))

        # Validate video and extract technical properties
        persisted_path, metadata = self.validate_and_save_upload(
            file_bytes=file_bytes,
            filename=filename,
            video_id=video_id,
        )

        async with self.session_factory() as db:
            await self.repo.create(
                session=db,
                video_id=video_id,
                filename=filename,
                file_path=str(persisted_path),
                duration_seconds=metadata["duration_seconds"],
                source_fps=metadata["source_fps"],
                analysis_fps=sampling_fps,
                width=metadata["width"],
                height=metadata["height"],
                total_frames=metadata["total_frames"],
                session_id=session_id,
            )
            await db.commit()

        # Launch background worker task
        task = asyncio.create_task(
            self._process_video_task(
                video_id=video_id,
                file_path=persisted_path,
                sampling_fps=sampling_fps,
                metadata=metadata,
            )
        )
        self._active_tasks[video_id] = task

        return VideoAnalysisJobResponse(
            video_id=str(video_id),
            session_id=str(session_id) if session_id else None,
            filename=filename,
            status="QUEUED",
            current_stage="queued",
            message="Video accepted. Temporal expression analysis has been scheduled.",
        )

    async def cancel_job(self, video_id: uuid.UUID) -> bool:
        """Request cancellation of an active video analysis job."""
        self._cancellation_flags.add(video_id)
        task = self._active_tasks.get(video_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    async def _process_video_task(
        self,
        video_id: uuid.UUID,
        file_path: Path,
        sampling_fps: float,
        metadata: dict[str, Any],
    ) -> None:
        """Background asynchronous execution pipeline for video analysis."""
        t_start = time.perf_counter()
        source_fps = metadata["source_fps"]
        total_frames = metadata["total_frames"]

        # Calculate sampling step
        step_frames = max(1, round(source_fps / sampling_fps))
        effective_analysis_fps = source_fps / step_frames

        logger.info(
            f"[{video_id}] Starting video analysis: {metadata['filename']} "
            f"({metadata['duration_seconds']}s, {source_fps} FPS, sampling step={step_frames} frames -> {effective_analysis_fps:.1f} analysis FPS)"
        )

        async with self.session_factory() as db:
            await self.repo.update_status(
                session=db,
                video_id=video_id,
                status="PROCESSING",
                current_stage="processing",
                started_at=datetime.now(UTC),
            )
            await db.commit()

        # Tracking and smoothing state machines
        tracker = FaceTracker(
            iou_threshold=0.30,
            max_centroid_distance=80.0,
            max_frames_missing=int(effective_analysis_fps * 2.0),
        )
        smoother = TemporalSmoother(
            alpha=0.6,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD,
            probability_margin=0.08,
            persistence_steps=2,
        )

        cap = cv2.VideoCapture(str(file_path))
        frame_idx = 0
        frames_analyzed = 0

        # Accumulated data structures
        raw_predictions_data: list[dict[str, Any]] = []
        track_lifecycle: dict[int, dict[str, Any]] = {}
        # active_segments[track_id] = {"emotion": ..., "start_time": ..., "end_time": ..., "confidences": [...], "count": int}
        active_segments: dict[int, dict[str, Any]] = {}
        completed_segments_data: list[dict[str, Any]] = []

        last_progress_update = time.perf_counter()

        try:
            while cap.isOpened():
                if video_id in self._cancellation_flags:
                    logger.info(f"[{video_id}] Analysis cancelled by user.")
                    cap.release()
                    async with self.session_factory() as db:
                        await self.repo.update_status(
                            session=db,
                            video_id=video_id,
                            status="CANCELLED",
                            current_stage="cancelled",
                        )
                        await db.commit()
                    return

                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break

                if frame_idx % step_frames == 0:
                    timestamp = round(frame_idx / source_fps, 3)

                    # Convert BGR to RGB for model inference
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                    # Production inference reuse (Zero second ML pipeline)
                    infer_result = self.prediction_service.engine.predict_image(image_source=frame_rgb)

                    # Release frame memory immediately
                    del frame_rgb

                    bboxes: list[BoundingBoxSchema] = []
                    if infer_result.faces:
                        bboxes = [
                            BoundingBoxSchema(
                                x=f.bbox.x,
                                y=f.bbox.y,
                                width=f.bbox.width,
                                height=f.bbox.height,
                            )
                            for f in infer_result.faces
                        ]

                    tracked_matches = tracker.update(bboxes)
                    active_track_ids: set[int] = set()

                    if tracked_matches:
                        for match_idx, (track_id, matched_bbox) in enumerate(tracked_matches):
                            active_track_ids.add(track_id)
                            raw_face = infer_result.faces[match_idx]

                            (
                                smoothed_emotion,
                                smoothed_conf,
                                is_uncertain,
                                smoothed_probs,
                            ) = smoother.smooth(
                                face_id=track_id,
                                raw_probabilities=raw_face.probabilities,
                                raw_emotion=raw_face.emotion,
                                raw_confidence=raw_face.confidence,
                            )

                            # Track lifecycle recording
                            if track_id not in track_lifecycle:
                                track_lifecycle[track_id] = {
                                    "track_id": track_id,
                                    "first_seen_sec": timestamp,
                                    "last_seen_sec": timestamp,
                                    "total_detections": 1,
                                }
                            else:
                                track_lifecycle[track_id]["last_seen_sec"] = timestamp
                                track_lifecycle[track_id]["total_detections"] += 1

                            # Frame prediction item
                            pred_item = {
                                "track_id": track_id,
                                "timestamp": timestamp,
                                "frame_index": frame_idx,
                                "bbox_x": matched_bbox.x,
                                "bbox_y": matched_bbox.y,
                                "bbox_width": matched_bbox.width,
                                "bbox_height": matched_bbox.height,
                                "detection_confidence": round(raw_face.detection_confidence, 4),
                                "raw_emotion": raw_face.emotion,
                                "raw_confidence": round(raw_face.confidence, 4),
                                "smoothed_emotion": smoothed_emotion,
                                "smoothed_confidence": round(smoothed_conf, 4),
                                "is_uncertain": is_uncertain,
                                "probabilities": smoothed_probs,
                                "raw_probabilities": raw_face.probabilities,
                            }
                            raw_predictions_data.append(pred_item)

                            # Expression Segment State Machine
                            if track_id not in active_segments:
                                active_segments[track_id] = {
                                    "emotion": smoothed_emotion,
                                    "start_time": timestamp,
                                    "end_time": timestamp,
                                    "confidences": [smoothed_conf],
                                    "prediction_count": 1,
                                }
                            else:
                                curr_seg = active_segments[track_id]
                                if curr_seg["emotion"] == smoothed_emotion:
                                    # Extend active segment
                                    curr_seg["end_time"] = timestamp
                                    curr_seg["confidences"].append(smoothed_conf)
                                    curr_seg["prediction_count"] += 1
                                else:
                                    # Transition: Close prior segment and start new segment
                                    duration = round(curr_seg["end_time"] - curr_seg["start_time"], 3)
                                    # Ensure non-zero duration
                                    min_dur = round(1.0 / effective_analysis_fps, 3)
                                    duration = max(min_dur, duration)
                                    end_t = max(curr_seg["end_time"], curr_seg["start_time"] + duration)

                                    confs = curr_seg["confidences"]
                                    completed_segments_data.append(
                                        {
                                            "track_id": track_id,
                                            "emotion": curr_seg["emotion"],
                                            "start_time": curr_seg["start_time"],
                                            "end_time": end_t,
                                            "duration": duration,
                                            "average_confidence": round(sum(confs) / len(confs), 4),
                                            "min_confidence": round(min(confs), 4),
                                            "max_confidence": round(max(confs), 4),
                                            "prediction_count": curr_seg["prediction_count"],
                                        }
                                    )

                                    # Open new segment
                                    active_segments[track_id] = {
                                        "emotion": smoothed_emotion,
                                        "start_time": timestamp,
                                        "end_time": timestamp,
                                        "confidences": [smoothed_conf],
                                        "prediction_count": 1,
                                    }

                    # Prune missing tracks from smoother
                    smoother.prune_missing(active_track_ids)

                    # Handle tracks lost from active detection (close segment if missing exceeds timeout)
                    inactive_tracks = set(active_segments.keys()) - active_track_ids
                    for inact_id in inactive_tracks:
                        # If track hasn't been seen in > 1.5 seconds, close its active segment
                        last_seen = track_lifecycle.get(inact_id, {}).get("last_seen_sec", 0.0)
                        if timestamp - last_seen > 1.5:
                            curr_seg = active_segments.pop(inact_id)
                            duration = round(curr_seg["end_time"] - curr_seg["start_time"], 3)
                            min_dur = round(1.0 / effective_analysis_fps, 3)
                            duration = max(min_dur, duration)
                            end_t = max(curr_seg["end_time"], curr_seg["start_time"] + duration)
                            confs = curr_seg["confidences"]
                            completed_segments_data.append(
                                {
                                    "track_id": inact_id,
                                    "emotion": curr_seg["emotion"],
                                    "start_time": curr_seg["start_time"],
                                    "end_time": end_t,
                                    "duration": duration,
                                    "average_confidence": round(sum(confs) / len(confs), 4),
                                    "min_confidence": round(min(confs), 4),
                                    "max_confidence": round(max(confs), 4),
                                    "prediction_count": curr_seg["prediction_count"],
                                }
                            )

                    frames_analyzed += 1

                # Discard raw frame buffer immediately
                del frame_bgr
                frame_idx += 1

                # Throttled progress report to database
                now = time.perf_counter()
                if now - last_progress_update >= 1.0 or frame_idx >= total_frames:
                    last_progress_update = now
                    progress = min(95.0, (frame_idx / max(1, total_frames)) * 100.0)
                    async with self.session_factory() as db:
                        await self.repo.update_progress(
                            session=db,
                            video_id=video_id,
                            progress_percent=progress,
                            frames_analyzed=frames_analyzed,
                            current_stage="processing",
                        )
                        await db.commit()

            cap.release()

            # Close any remaining open segments at video conclusion
            for track_id, curr_seg in active_segments.items():
                duration = round(curr_seg["end_time"] - curr_seg["start_time"], 3)
                min_dur = round(1.0 / effective_analysis_fps, 3)
                duration = max(min_dur, duration)
                end_t = max(curr_seg["end_time"], curr_seg["start_time"] + duration)
                confs = curr_seg["confidences"]
                completed_segments_data.append(
                    {
                        "track_id": track_id,
                        "emotion": curr_seg["emotion"],
                        "start_time": curr_seg["start_time"],
                        "end_time": end_t,
                        "duration": duration,
                        "average_confidence": round(sum(confs) / len(confs), 4),
                        "min_confidence": round(min(confs), 4),
                        "max_confidence": round(max(confs), 4),
                        "prediction_count": curr_seg["prediction_count"],
                    }
                )

            # Sort completed segments chronologically
            completed_segments_data.sort(key=lambda s: (s["track_id"], s["start_time"]))

            # Persist results to DB
            async with self.session_factory() as db:
                await self.repo.update_progress(
                    session=db,
                    video_id=video_id,
                    progress_percent=98.0,
                    frames_analyzed=frames_analyzed,
                    current_stage="saving",
                )
                await self.repo.save_results(
                    session=db,
                    video_id=video_id,
                    tracks=list(track_lifecycle.values()),
                    segments=completed_segments_data,
                    predictions=raw_predictions_data,
                )

                t_end = time.perf_counter()
                total_processing_seconds = round(t_end - t_start, 2)

                await self.repo.update_status(
                    session=db,
                    video_id=video_id,
                    status="COMPLETED",
                    current_stage="completed",
                    completed_at=datetime.now(UTC),
                    processing_time_seconds=total_processing_seconds,
                )
                await self.repo.update_progress(
                    session=db,
                    video_id=video_id,
                    progress_percent=100.0,
                    frames_analyzed=frames_analyzed,
                    current_stage="completed",
                )
                await db.commit()

            logger.info(
                f"[{video_id}] Video analysis COMPLETED in {total_processing_seconds}s. "
                f"Generated {len(raw_predictions_data)} predictions, {len(completed_segments_data)} segments across {len(track_lifecycle)} tracks."
            )

        except asyncio.CancelledError:
            logger.info(f"[{video_id}] Video processing task cancelled.")
            if cap.isOpened():
                cap.release()
            async with self.session_factory() as db:
                await self.repo.update_status(
                    session=db,
                    video_id=video_id,
                    status="CANCELLED",
                    current_stage="cancelled",
                )
                await db.commit()
            raise
        except Exception as exc:
            logger.error(f"[{video_id}] Video processing failed: {exc}", exc_info=True)
            if cap.isOpened():
                cap.release()
            async with self.session_factory() as db:
                await self.repo.update_status(
                    session=db,
                    video_id=video_id,
                    status="FAILED",
                    current_stage="failed",
                    error_message=str(exc),
                )
                await db.commit()
        finally:
            self._active_tasks.pop(video_id, None)
            self._cancellation_flags.discard(video_id)

    async def get_status(self, video_id: uuid.UUID) -> VideoAnalysisStatusResponse:
        """Query lightweight progress and execution status."""
        async with self.session_factory() as db:
            record = await self.repo.get_by_id(db, video_id)
            if not record:
                raise ResourceNotFoundError(
                    message=f"Video analysis '{video_id}' not found.",
                    details={"video_id": str(video_id)},
                )
            return VideoAnalysisStatusResponse(
                video_id=str(record.id),
                status=record.status,
                current_stage=record.current_stage,
                progress_percent=record.progress_percent,
                frames_analyzed=record.frames_analyzed,
                total_frames=record.total_frames,
                error_message=record.error_message,
            )

    async def get_detail(self, video_id: uuid.UUID) -> dict[str, Any]:
        """Query complete video metadata, tracks, and aggregated analytics."""
        async with self.session_factory() as db:
            record = await self.repo.get_by_id(db, video_id)
            if not record:
                raise ResourceNotFoundError(
                    message=f"Video analysis '{video_id}' not found.",
                    details={"video_id": str(video_id)},
                )

            analytics = await self.repo.get_analytics_summary(db, video_id)

            tracks = [
                VideoTrackSchema(
                    track_id=t.track_id,
                    first_seen_sec=t.first_seen_sec,
                    last_seen_sec=t.last_seen_sec,
                    total_detections=t.total_detections,
                )
                for t in record.tracks
            ]

            metadata = VideoMetadataSchema(
                filename=record.filename,
                duration_seconds=record.duration_seconds,
                source_fps=record.source_fps,
                analysis_fps=record.analysis_fps,
                width=record.width,
                height=record.height,
                total_frames=record.total_frames,
                frames_analyzed=record.frames_analyzed,
            )

            return {
                "video_id": str(record.id),
                "session_id": str(record.session_id) if record.session_id else None,
                "status": record.status,
                "current_stage": record.current_stage,
                "progress_percent": record.progress_percent,
                "metadata": metadata,
                "analytics": analytics,
                "tracks": tracks,
                "created_at": record.created_at,
                "started_at": record.started_at,
                "completed_at": record.completed_at,
                "error_message": record.error_message,
            }

    async def get_timeline(
        self,
        video_id: uuid.UUID,
        track_id: int | None = None,
    ) -> tuple[list[ExpressionSegmentSchema], list[ExpressionEventSchema]]:
        """Retrieve segments and transition events for timeline visualization."""
        async with self.session_factory() as db:
            record = await self.repo.get_by_id(db, video_id)
            if not record:
                raise ResourceNotFoundError(
                    message=f"Video analysis '{video_id}' not found.",
                    details={"video_id": str(video_id)},
                )
            return await self.repo.get_timeline(db, video_id, track_id=track_id)

    async def get_predictions(
        self,
        video_id: uuid.UUID,
        track_id: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[Any], int]:
        """Query paginated frame-level predictions."""
        async with self.session_factory() as db:
            record = await self.repo.get_by_id(db, video_id)
            if not record:
                raise ResourceNotFoundError(
                    message=f"Video analysis '{video_id}' not found.",
                    details={"video_id": str(video_id)},
                )
            return await self.repo.get_predictions(
                session=db,
                video_id=video_id,
                track_id=track_id,
                start_time=start_time,
                end_time=end_time,
                page=page,
                page_size=page_size,
            )

    async def get_video_file_path(self, video_id: uuid.UUID) -> Path:
        """Resolve file path for streaming video playback."""
        async with self.session_factory() as db:
            record = await self.repo.get_by_id(db, video_id)
            if not record or not record.file_path:
                raise ResourceNotFoundError(
                    message=f"Video '{video_id}' file not found.",
                    details={"video_id": str(video_id)},
                )
            p = Path(record.file_path)
            if not p.exists():
                raise ResourceNotFoundError(
                    message=f"Video file on disk not found for '{video_id}'.",
                    details={"path": str(p)},
                )
            return p

    async def get_recent_videos(self, limit: int = 20) -> list[dict[str, Any]]:
        """Query recently processed video analyses for dashboard and history lists."""
        async with self.session_factory() as db:
            records = await self.repo.list_recent(session=db, limit=limit)
            return [
                {
                    "video_id": str(r.id),
                    "filename": r.filename,
                    "status": r.status,
                    "current_stage": r.current_stage,
                    "progress_percent": r.progress_percent,
                    "duration_seconds": r.duration_seconds,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "frames_analyzed": r.frames_analyzed,
                    "error_message": r.error_message,
                }
                for r in records
            ]

