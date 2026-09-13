"""Real-Time stream processing service orchestrating frame processing, Phase 09 ML inference, temporal smoothing, and session management."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import InferenceError
from app.core.logging import get_logger
from app.db.repositories.analysis_session import SessionRepository
from app.db.repositories.prediction import PredictionRepository
from app.schemas.prediction import BoundingBoxSchema
from app.schemas.realtime import (
    DetectedFaceRealtime,
    RealtimeMetrics,
    RealtimePredictionResponse,
)
from app.services.face_tracker import FaceTracker
from app.services.frame_processor import FrameProcessor
from app.services.prediction_service import PredictionService
from app.services.temporal_smoother import TemporalSmoother
from ml.inference.schemas import InferenceStatus

logger = get_logger(__name__)


class RealTimeService:
    """Orchestrator for real-time video emotion detection sessions."""

    def __init__(
        self,
        prediction_service: PredictionService,
        session_repo: SessionRepository | None = None,
        prediction_repo: PredictionRepository | None = None,
        frame_processor: FrameProcessor | None = None,
        face_tracker: FaceTracker | None = None,
        temporal_smoother: TemporalSmoother | None = None,
    ) -> None:
        self.prediction_service = prediction_service
        self.session_repo = session_repo or SessionRepository()
        self.prediction_repo = prediction_repo or PredictionRepository()
        self.frame_processor = frame_processor or FrameProcessor()
        self.face_tracker = face_tracker or FaceTracker()
        self.temporal_smoother = temporal_smoother or TemporalSmoother(
            alpha=settings.REALTIME_SMOOTHING_ALPHA,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        )

        # Performance & telemetry counters
        self.frames_received: int = 0
        self.frames_processed: int = 0
        self.frames_dropped: int = 0
        self._fps_window_start: float = time.perf_counter()
        self._fps_window_count: int = 0
        self.current_fps: float = 0.0
        self.last_persist_time: float = 0.0

    async def initialize_session(
        self,
        db: AsyncSession,
        session_name: str | None = None,
        user_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        """Create and persist a new real-time video analysis session.

        Args:
            db: Database session.
            session_name: Optional human-readable label.
            user_id: Optional owner user ID.

        Returns:
            UUID of the newly initialized session.
        """
        session_id = uuid.uuid4()
        name = session_name or f"Realtime Session {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}"
        await self.session_repo.create(
            session=db,
            session_id=session_id,
            user_id=user_id,
            name=name,
        )
        await db.commit()
        logger.info(f"Initialized real-time session: {session_id} ('{name}')")
        return session_id

    async def finalize_session(
        self,
        db: AsyncSession,
        session_id: uuid.UUID,
    ) -> None:
        """Complete a real-time analysis session and set end timestamp.

        Args:
            db: Database session.
            session_id: Session ID to finalize.
        """
        try:
            await self.session_repo.end_session(
                session=db,
                session_id=session_id,
                status="completed",
            )
            await db.commit()
            logger.info(f"Finalized real-time session: {session_id}")
        except Exception as exc:
            await db.rollback()
            logger.warning(f"Error finalizing real-time session {session_id}: {exc}")

    def record_dropped_frame(self) -> None:
        """Increment count of frames discarded due to backpressure."""
        self.frames_dropped += 1

    async def process_frame(
        self,
        raw_bytes: bytes,
        frame_id: int,
        client_timestamp: float | None = None,
        db: AsyncSession | None = None,
        session_id: uuid.UUID | None = None,
    ) -> RealtimePredictionResponse:
        """Execute full real-time frame processing pipeline.

        Pipeline stages:
            1. Validate & decode frame bytes in-memory
            2. Invoke Phase 09 ML inference engine via PredictionService
            3. Multi-face spatial tracking (IoU + Centroid)
            4. Temporal emotion probability smoothing (EMA)
            5. Controlled periodic persistence (sampling)
            6. Telemetry & FPS calculation

        Args:
            raw_bytes: Raw JPEG/PNG image bytes.
            frame_id: Monotonically increasing frame sequence ID.
            client_timestamp: Optional client-side capture timestamp.
            db: Optional database session for periodic sampling persistence.
            session_id: Optional associated analysis session ID.

        Returns:
            RealtimePredictionResponse with detected faces, probabilities, and metrics.
        """
        t_start = time.perf_counter()
        self.frames_received += 1

        # Stage 1: In-memory decoding & validation
        _, img_w, img_h = self.frame_processor.decode_frame_bytes(raw_bytes)

        # Stage 2: Phase 09 ML Inference via PredictionService
        t_infer_start = time.perf_counter()
        inference_result = self.prediction_service.predict_frame(raw_bytes)
        t_infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

        if inference_result.status == InferenceStatus.INFERENCE_ERROR:
            raise InferenceError(
                message=inference_result.error_message or "Real-time inference failed.",
                details={"status": inference_result.status.value},
            )

        # Stage 3 & 4: Multi-Face Spatial Tracking & Temporal Smoothing
        detected_faces_realtime: list[DetectedFaceRealtime] = []
        active_face_ids: set[int] = set()

        if inference_result.faces:
            # Extract bounding boxes for tracker
            bboxes = [
                BoundingBoxSchema(
                    x=face.bbox.x,
                    y=face.bbox.y,
                    width=face.bbox.width,
                    height=face.bbox.height,
                )
                for face in inference_result.faces
            ]

            # Update tracker
            tracked_matches = self.face_tracker.update(bboxes)

            # Build smoothed face outputs
            for idx, (track_id, matched_bbox) in enumerate(tracked_matches):
                active_face_ids.add(track_id)
                raw_face = inference_result.faces[idx]

                (
                    smoothed_emotion,
                    smoothed_conf,
                    is_uncertain,
                    smoothed_probs,
                ) = self.temporal_smoother.smooth(
                    face_id=track_id,
                    raw_probabilities=raw_face.probabilities,
                    raw_emotion=raw_face.emotion,
                    raw_confidence=raw_face.confidence,
                )

                face_realtime = DetectedFaceRealtime(
                    face_id=track_id,
                    bbox=matched_bbox,
                    detection_confidence=raw_face.detection_confidence,
                    emotion=smoothed_emotion,
                    confidence=smoothed_conf,
                    is_uncertain=is_uncertain,
                    probabilities=smoothed_probs,
                    raw_emotion=raw_face.emotion,
                    raw_confidence=raw_face.confidence,
                )
                detected_faces_realtime.append(face_realtime)
        else:
            # No face detected: inform tracker of empty frame
            self.face_tracker.update([])

        # Prune inactive faces from smoother
        self.temporal_smoother.prune_missing(active_face_ids)

        # Stage 5: Controlled Periodic Persistence (e.g. 1 sample / second)
        now_mono = time.perf_counter()
        if (
            session_id is not None
            and (now_mono - self.last_persist_time >= settings.REALTIME_PERSISTENCE_INTERVAL_SEC)
        ):
            self.last_persist_time = now_mono
            try:
                faces_data = [
                    {
                        "face_id": f.face_id,
                        "bbox_x": f.bbox.x,
                        "bbox_y": f.bbox.y,
                        "bbox_width": f.bbox.width,
                        "bbox_height": f.bbox.height,
                        "detection_confidence": f.detection_confidence,
                        "emotion": f.emotion,
                        "confidence": f.confidence,
                        "is_uncertain": f.is_uncertain,
                        "probabilities": f.probabilities,
                    }
                    for f in detected_faces_realtime
                ]
                from app.db.session import get_session_factory
                factory = get_session_factory()
                async with factory() as persist_session:
                    await self.prediction_repo.create_prediction_with_faces(
                        session=persist_session,
                        prediction_id=uuid.uuid4(),
                        request_id=f"rt-{session_id}-{frame_id}",
                        model_version=settings.MODEL_VERSION,
                        status=inference_result.status.value,
                        faces_detected=len(detected_faces_realtime),
                        processing_time_ms=t_infer_ms,
                        image_width=img_w,
                        image_height=img_h,
                        session_id=session_id,
                        faces_data=faces_data,
                    )
                    await persist_session.commit()
            except Exception as exc:
                logger.warning(f"Error persisting real-time frame sample: {exc}")

        # Stage 6: Telemetry & FPS updates
        self.frames_processed += 1
        self._fps_window_count += 1
        elapsed_fps_window = now_mono - self._fps_window_start
        if elapsed_fps_window >= 1.0:
            self.current_fps = round(self._fps_window_count / elapsed_fps_window, 1)
            self._fps_window_start = now_mono
            self._fps_window_count = 0

        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        metrics = RealtimeMetrics(
            inference_time_ms=round(t_infer_ms, 2),
            total_processing_time_ms=round(t_total_ms, 2),
            fps=self.current_fps,
            dropped_frames=self.frames_dropped,
            server_timestamp=time.time(),
        )

        return RealtimePredictionResponse(
            type="prediction",
            frame_id=frame_id,
            client_timestamp=client_timestamp,
            session_id=str(session_id) if session_id else None,
            faces_detected=len(detected_faces_realtime),
            faces=detected_faces_realtime,
            metrics=metrics,
        )

    def reset(self) -> None:
        """Reset internal trackers, smoothers, and telemetry counters."""
        self.face_tracker.reset()
        self.temporal_smoother.reset()
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.current_fps = 0.0
        self.last_persist_time = 0.0
