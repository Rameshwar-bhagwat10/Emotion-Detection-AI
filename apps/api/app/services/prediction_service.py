"""Service layer for orchestrating image predictions and persistence."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    DatabaseException,
    InferenceError,
    InvalidImageError,
    PayloadTooLargeError,
    ResourceNotFoundError,
    UnsupportedMediaTypeError,
)
from app.db.repositories.prediction import PredictionRepository
from app.schemas.prediction import (
    BoundingBoxSchema,
    FacePredictionSchema,
    ModelInfoSchema,
    PredictionDetailSchema,
    PredictionResponseSchema,
    PredictionTimingSchema,
)
from ml.inference.engine import EmotionInferenceEngine
from ml.inference.schemas import ImageInferenceResult, InferenceStatus

logger = logging.getLogger(__name__)


class PredictionService:
    """Orchestrates image validation, Phase 09 ML inference, and database persistence."""

    def __init__(
        self,
        inference_engine: EmotionInferenceEngine,
        prediction_repo: PredictionRepository | None = None,
    ) -> None:
        self.engine = inference_engine
        self.repo = prediction_repo or PredictionRepository()

    def predict_frame(
        self,
        image_source: bytes | Any,
    ) -> ImageInferenceResult:
        """Execute synchronous Phase 09 ML inference on a real-time video frame.

        Args:
            image_source: Raw image bytes (JPEG/PNG) or OpenCV BGR NumPy array.

        Returns:
            ImageInferenceResult with detected faces, bounding boxes, and probabilities.
        """
        return self.engine.predict_image(image_source=image_source)

    async def predict_and_persist(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str | None,
        db: AsyncSession | None,
        request_id: str,
        session_id: uuid.UUID | None = None,
    ) -> PredictionResponseSchema:
        """Validate upload, invoke Phase 09 ML engine, persist metadata, and return response schema."""
        # 1. Payload size validation
        max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
        if len(image_bytes) > max_bytes:
            raise PayloadTooLargeError(
                message=f"Image size {len(image_bytes)/(1024*1024):.2f}MB exceeds limit of {settings.MAX_IMAGE_SIZE_MB}MB",
                details={
                    "max_size_mb": settings.MAX_IMAGE_SIZE_MB,
                    "actual_bytes": len(image_bytes),
                },
            )

        # 2. File extension & content type validation
        if filename:
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if ext and ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
                raise UnsupportedMediaTypeError(
                    message=f"File extension '{ext}' is not supported. Allowed: {settings.ALLOWED_IMAGE_EXTENSIONS}",
                    details={
                        "allowed_extensions": settings.ALLOWED_IMAGE_EXTENSIONS,
                        "received_extension": ext,
                    },
                )

        # 3. Phase 09 ML Inference (Single Source of Truth)
        try:
            inference_result: ImageInferenceResult = self.engine.predict_image(
                image_source=image_bytes
            )
        except Exception as exc:
            logger.error(f"[{request_id}] Phase 09 inference call failed: {exc}", exc_info=True)
            raise InferenceError(message=f"ML inference execution failed: {exc}") from exc

        # 4. Check Inference Engine Status
        if inference_result.status in (
            InferenceStatus.INVALID_IMAGE,
            InferenceStatus.INVALID_INPUT,
        ):
            raise InvalidImageError(
                message=inference_result.error_message
                or "Uploaded file is not a valid or readable image.",
                details={"status": inference_result.status.value},
            )
        elif inference_result.status == InferenceStatus.INFERENCE_ERROR:
            raise InferenceError(
                message=inference_result.error_message
                or "Inference pipeline encountered an error.",
                details={"status": inference_result.status.value},
            )

        # 5. Prepare face schemas and persistence payload
        face_schemas: list[FacePredictionSchema] = []
        faces_persist_data: list[dict[str, Any]] = []

        for face in inference_result.faces:
            face_schema = FacePredictionSchema(
                face_id=face.face_id,
                bbox=BoundingBoxSchema(
                    x=face.bbox.x,
                    y=face.bbox.y,
                    width=face.bbox.width,
                    height=face.bbox.height,
                ),
                detection_confidence=face.detection_confidence,
                emotion=face.emotion,
                confidence=face.confidence,
                is_uncertain=face.is_uncertain,
                probabilities=face.probabilities,
            )
            face_schemas.append(face_schema)

            faces_persist_data.append(
                {
                    "face_id": face.face_id,
                    "bbox_x": face.bbox.x,
                    "bbox_y": face.bbox.y,
                    "bbox_width": face.bbox.width,
                    "bbox_height": face.bbox.height,
                    "detection_confidence": face.detection_confidence,
                    "emotion": face.emotion,
                    "confidence": face.confidence,
                    "is_uncertain": face.is_uncertain,
                    "probabilities": face.probabilities,
                }
            )

        # 6. Database Transactional Persistence (if db session provided)
        persisted_id: uuid.UUID | None = None
        model_version = (
            inference_result.model_info.model_name
            if inference_result.model_info
            else settings.MODEL_VERSION
        )
        image_w = inference_result.image.width if inference_result.image else None
        image_h = inference_result.image.height if inference_result.image else None
        proc_time = inference_result.timing.total_ms if inference_result.timing else 0.0

        if db is not None:
            try:
                prediction_record = await self.repo.create_prediction_with_faces(
                    session=db,
                    prediction_id=uuid.uuid4(),
                    request_id=request_id,
                    model_version=model_version,
                    status=inference_result.status.value,
                    faces_detected=inference_result.faces_detected,
                    processing_time_ms=proc_time,
                    image_width=image_w,
                    image_height=image_h,
                    session_id=session_id,
                    faces_data=faces_persist_data,
                )
                persisted_id = prediction_record.id
                await db.commit()
            except Exception as exc:
                await db.rollback()
                logger.error(
                    f"[{request_id}] Failed to persist prediction record: {exc}", exc_info=True
                )
                raise DatabaseException(message=f"Database persistence failed: {exc}") from exc

        # 7. Construct and return final API response
        timing_schema = (
            PredictionTimingSchema(
                image_loading_ms=inference_result.timing.image_loading_ms,
                face_detection_ms=inference_result.timing.face_detection_ms,
                preprocessing_ms=inference_result.timing.preprocessing_ms,
                inference_ms=inference_result.timing.inference_ms,
                postprocessing_ms=inference_result.timing.postprocessing_ms,
                total_ms=inference_result.timing.total_ms,
            )
            if inference_result.timing
            else PredictionTimingSchema()
        )

        model_schema = (
            ModelInfoSchema(
                model_name=inference_result.model_info.model_name,
                architecture=inference_result.model_info.architecture,
                optimization_type=inference_result.model_info.optimization_type,
                device=inference_result.model_info.device,
            )
            if inference_result.model_info
            else ModelInfoSchema(
                model_name=settings.MODEL_VERSION,
                architecture="resnet18",
                optimization_type="pruning_30pct",
                device="cpu",
            )
        )

        return PredictionResponseSchema(
            status=inference_result.status.value,
            request_id=request_id,
            prediction_id=str(persisted_id) if persisted_id else None,
            session_id=str(session_id) if session_id else None,
            faces_detected=inference_result.faces_detected,
            faces=face_schemas,
            timing=timing_schema,
            model_info=model_schema,
            created_at=datetime.now(UTC),
        )

    async def get_prediction_by_id(
        self,
        db: AsyncSession,
        prediction_id: uuid.UUID,
    ) -> PredictionDetailSchema:
        """Retrieve stored prediction and mapped faces by UUID."""
        record = await self.repo.get_by_id(db, prediction_id, load_faces=True)
        if record is None:
            raise ResourceNotFoundError(
                message=f"Prediction with ID '{prediction_id}' not found",
                details={"prediction_id": str(prediction_id)},
            )

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
            for f in record.faces
        ]

        return PredictionDetailSchema(
            id=record.id,
            session_id=record.session_id,
            request_id=record.request_id,
            model_version=record.model_version,
            status=record.status,
            faces_detected=record.faces_detected,
            processing_time_ms=record.processing_time_ms,
            image_width=record.image_width,
            image_height=record.image_height,
            created_at=record.created_at,
            faces=faces,
        )
