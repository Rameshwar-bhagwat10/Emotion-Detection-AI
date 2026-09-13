"""High-level emotion inference engine orchestrating end-to-end computer vision prediction."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from PIL import Image

from ml.inference.config import InferencePipelineConfig, load_inference_config
from ml.inference.face_detector import BaseFaceDetector, FaceDetection, create_face_detector
from ml.inference.image_loader import (
    ImageLoadError,
    ImageNotFoundError,
    InvalidImageDimensionsError,
    UnsupportedImageFormatError,
    load_image,
)
from ml.inference.model_loader import ModelManager
from ml.inference.postprocessing import process_logits
from ml.inference.preprocessor import FacePreprocessor, InvalidCropError, PreprocessingError
from ml.inference.schemas import (
    FacePrediction,
    ImageInferenceResult,
    ImageInfo,
    InferenceStatus,
    InferenceTiming,
    ModelInfo,
)


class EmotionInferenceEngine:
    """Production-grade facial expression emotion inference engine.

    Orchestrates:
        1. Image ingestion & validation
        2. Face detection
        3. Face cropping & boundary padding
        4. Deterministic model preprocessing
        5. Batched Champion model inference
        6. Softmax probability & confidence gating
        7. Structured prediction schemas with latency breakdowns
    """

    def __init__(
        self,
        config: InferencePipelineConfig | None = None,
        config_path: str | Path | None = None,
        auto_load: bool = True,
    ) -> None:
        """Initialize EmotionInferenceEngine.

        Args:
            config: Optional pre-constructed InferencePipelineConfig.
            config_path: Optional path to YAML configuration.
            auto_load: Whether to eagerly load the model and warm up upon instantiation.
        """
        if config is not None:
            self.config = config
        else:
            self.config = load_inference_config(config_path)

        self.model_manager = ModelManager(config=self.config.model)
        self.detector: BaseFaceDetector = create_face_detector(self.config.face_detection)
        self.preprocessor = FacePreprocessor(self.config.model)

        self.model: torch.nn.Module | None = None
        self.metadata: dict[str, Any] = {}
        self.device = self.model_manager.device

        if auto_load:
            self.load()

    def load(self) -> None:
        """Eagerly load model into memory and execute warm-up passes."""
        self.model, self.metadata = self.model_manager.load_champion()
        self.device = self.model_manager.device
        if self.config.warmup_iterations > 0:
            self.model_manager.warmup(self.config.warmup_iterations)

    def warmup(self, num_warmup_passes: int = 3) -> None:
        """Execute non-gradient warm-up passes for both emotion model and face detector."""
        self.model_manager.warmup(num_warmup_passes)
        # Prime face detector buffers
        try:
            dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
            self.detector.detect(dummy_img)
        except Exception as e:
            logging.debug(f"Detector warmup notice: {e}")

    def warm_up(self, num_warmup_passes: int = 3) -> None:
        """Alias for warmup."""
        self.warmup(num_warmup_passes)

    def _get_model_info(self) -> ModelInfo:
        """Construct ModelInfo descriptor."""
        return ModelInfo(
            model_name=str(self.metadata.get("model_name", "champion-pruning-30")),
            architecture=str(
                self.metadata.get("base_model", self.config.model.expected_architecture)
            ),
            optimization_type=str(self.metadata.get("optimization_technique", "pruning_30pct")),
            device=str(self.device),
        )

    def _execute_forward_chunks(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in batch chunks."""
        assert self.model is not None
        with torch.inference_mode():
            if batch_tensor.shape[0] <= self.config.batch_size:
                logits: torch.Tensor = self.model(batch_tensor)
            else:
                chunks: list[torch.Tensor] = []
                for b_start in range(0, batch_tensor.shape[0], self.config.batch_size):
                    chunk = batch_tensor[b_start : b_start + self.config.batch_size]
                    chunks.append(self.model(chunk))
                logits = torch.cat(chunks, dim=0)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            return logits

    def _extract_crops(
        self, image_rgb: np.ndarray, detections: list[FaceDetection]
    ) -> tuple[list[np.ndarray], list[FaceDetection]]:
        """Extract valid face crops for detected bounding boxes."""
        valid_crops: list[np.ndarray] = []
        valid_detections: list[FaceDetection] = []
        for det in detections:
            try:
                crop = self.preprocessor.crop_face(image_rgb, det.bbox)
                valid_crops.append(crop)
                valid_detections.append(det)
            except (InvalidCropError, PreprocessingError) as e:
                logging.warning(f"Skipping invalid face crop {det.face_id}: {e}")
        return valid_crops, valid_detections

    def predict_image(
        self,
        image_source: str | Path | bytes | BinaryIO | Image.Image | np.ndarray,
    ) -> ImageInferenceResult:
        """Run end-to-end emotion prediction on an image input.

        Args:
            image_source: Image file path, raw byte stream, PIL Image, or NumPy array.

        Returns:
            Structured ImageInferenceResult containing all detected faces, emotions, and timing.
        """
        if self.model is None:
            self.load()

        model_info = self._get_model_info()
        timing = InferenceTiming()
        t_start_total = time.perf_counter()

        # 1. Image Ingestion
        t0 = time.perf_counter()
        try:
            image_rgb, img_fmt = load_image(
                image_source, max_dimension=self.config.max_image_dimension
            )
        except ImageNotFoundError as e:
            return ImageInferenceResult(
                status=InferenceStatus.INVALID_IMAGE,
                model_info=model_info,
                error_message=f"Image not found: {e}",
            )
        except (UnsupportedImageFormatError, InvalidImageDimensionsError, ImageLoadError) as e:
            return ImageInferenceResult(
                status=InferenceStatus.INVALID_IMAGE,
                model_info=model_info,
                error_message=f"Invalid image: {e}",
            )
        except Exception as e:
            return ImageInferenceResult(
                status=InferenceStatus.INFERENCE_ERROR,
                model_info=model_info,
                error_message=f"Unexpected error loading image: {e}",
            )

        timing.image_loading_ms = (time.perf_counter() - t0) * 1000.0
        h, w = image_rgb.shape[:2]
        image_info = ImageInfo(
            width=w,
            height=h,
            channels=image_rgb.shape[2] if image_rgb.ndim == 3 else 1,
            format=img_fmt,
        )

        # 2. Face Detection
        t1 = time.perf_counter()
        detections = self.detector.detect(image_rgb)
        timing.face_detection_ms = (time.perf_counter() - t1) * 1000.0

        if not detections:
            timing.total_ms = (time.perf_counter() - t_start_total) * 1000.0
            return ImageInferenceResult(
                status=InferenceStatus.NO_FACE_DETECTED,
                image=image_info,
                faces_detected=0,
                faces=[],
                timing=timing,
                model_info=model_info,
            )

        # 3. Cropping & Preprocessing
        t2 = time.perf_counter()
        valid_crops, valid_detections = self._extract_crops(image_rgb, detections)
        if not valid_crops:
            timing.total_ms = (time.perf_counter() - t_start_total) * 1000.0
            return ImageInferenceResult(
                status=InferenceStatus.NO_FACE_DETECTED,
                image=image_info,
                faces_detected=0,
                faces=[],
                timing=timing,
                model_info=model_info,
                error_message="No valid face bounding boxes.",
            )

        batch_tensor = self.preprocessor.preprocess_batch(valid_crops, device=self.device)
        timing.preprocessing_ms = (time.perf_counter() - t2) * 1000.0

        # 4. Neural Network Inference
        t3 = time.perf_counter()
        logits = self._execute_forward_chunks(batch_tensor)
        timing.inference_ms = (time.perf_counter() - t3) * 1000.0

        # 5. Postprocessing
        t4 = time.perf_counter()
        face_predictions: list[FacePrediction] = process_logits(
            logits=logits,
            detections=valid_detections,
            classes=self.config.classes,
            confidence_threshold=self.config.confidence_threshold,
            uncertain_label=self.config.uncertain_label,
            logit_adjustment_tau=getattr(self.config.model, "logit_adjustment_tau", 0.30),
            class_biases=self.metadata.get("calibration_biases", None) if self.metadata else None,
        )
        timing.postprocessing_ms = (time.perf_counter() - t4) * 1000.0
        timing.total_ms = (time.perf_counter() - t_start_total) * 1000.0

        return ImageInferenceResult(
            status=InferenceStatus.SUCCESS,
            image=image_info,
            faces_detected=len(face_predictions),
            faces=face_predictions,
            timing=timing,
            model_info=model_info,
        )

    def predict_batch(
        self,
        image_sources: Sequence[str | Path | bytes | BinaryIO | Image.Image | np.ndarray],
    ) -> list[ImageInferenceResult]:
        """Predict emotions across a list of images."""
        return [self.predict_image(src) for src in image_sources]
