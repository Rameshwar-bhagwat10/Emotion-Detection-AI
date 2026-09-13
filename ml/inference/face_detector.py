"""Face detection subsystem with bounding-box processing and multiple detector backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ml.inference.config import FaceDetectionConfig

BUNDLED_YUNET_PATH = Path(__file__).resolve().parent / "assets" / "face_detection_yunet.onnx"
BUNDLED_HAAR_PATH = (
    Path(__file__).resolve().parent / "assets" / "haarcascade_frontalface_default.xml"
)


@dataclass
class FaceBoundingBox:
    """Represents a 2D bounding box with coordinate validation, padding, and clipping."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate and cast coordinates."""
        self.x = int(self.x)
        self.y = int(self.y)
        self.width = int(self.width)
        self.height = int(self.height)

    @property
    def x1(self) -> int:
        """Left coordinate."""
        return self.x

    @property
    def y1(self) -> int:
        """Top coordinate."""
        return self.y

    @property
    def x2(self) -> int:
        """Right coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom coordinate."""
        return self.y + self.height

    @property
    def area(self) -> int:
        """Area in pixels."""
        return max(0, self.width) * max(0, self.height)

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Return (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    def pad_and_clip(
        self,
        padding_fraction: float,
        img_width: int,
        img_height: int,
    ) -> FaceBoundingBox:
        """Expand bounding box symmetrically by a padding fraction and clip to image borders.

        Args:
            padding_fraction: Fraction by which to expand box width and height (e.g. 0.15 for 15%).
            img_width: Total image width in pixels.
            img_height: Total image height in pixels.

        Returns:
            Padded and safely clipped FaceBoundingBox.
        """
        if padding_fraction <= 0.0:
            x1 = max(0, min(self.x, img_width - 1))
            y1 = max(0, min(self.y, img_height - 1))
            x2 = max(x1 + 1, min(self.x + self.width, img_width))
            y2 = max(y1 + 1, min(self.y + self.height, img_height))
            return FaceBoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

        pad_w = int(self.width * padding_fraction / 2.0)
        pad_h = int(self.height * padding_fraction / 2.0)

        x1 = max(0, self.x - pad_w)
        y1 = max(0, self.y - pad_h)
        x2 = min(img_width, self.x + self.width + pad_w)
        y2 = min(img_height, self.y + self.height + pad_h)

        new_w = max(1, x2 - x1)
        new_h = max(1, y2 - y1)

        return FaceBoundingBox(x=x1, y=y1, width=new_w, height=new_h)


@dataclass
class FaceDetection:
    """Detection result for a single face."""

    face_id: int
    bbox: FaceBoundingBox
    confidence: float


class BaseFaceDetector(ABC):
    """Abstract base class for all face detection backends."""

    @abstractmethod
    def detect(self, image_rgb: np.ndarray) -> list[FaceDetection]:
        """Detect faces in an RGB image.

        Args:
            image_rgb: NumPy array of shape [H, W, 3] uint8.

        Returns:
            List of detected faces with deterministic face IDs (1, 2, ...).
        """
        pass


class YuNetFaceDetector(BaseFaceDetector):
    """Face detector utilizing OpenCV DNN YuNet ONNX model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: FaceDetectionConfig | None = None,
    ) -> None:
        """Initialize YuNet detector."""
        self.config = config or FaceDetectionConfig()

        target_path = Path(model_path) if model_path else BUNDLED_YUNET_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"YuNet ONNX model not found at {target_path}")

        self.model_path = target_path
        self._detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (320, 320),
            score_threshold=self.config.confidence_threshold,
            nms_threshold=self.config.nms_threshold,
            top_k=self.config.max_faces,
        )

    def detect(self, image_rgb: np.ndarray) -> list[FaceDetection]:
        """Detect faces using YuNet."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image of shape [H, W, 3], got {image_rgb.shape}")

        h, w = image_rgb.shape[:2]
        self._detector.setInputSize((w, h))
        if hasattr(self._detector, "setScoreThreshold"):
            self._detector.setScoreThreshold(self.config.confidence_threshold)

        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _, faces = self._detector.detect(bgr)

        detections: list[FaceDetection] = []
        if faces is not None:
            raw_detections = []
            for face in faces:
                fx, fy, fw, fh = map(int, face[0:4])
                conf = float(face[14])
                if conf >= self.config.confidence_threshold:
                    bbox = FaceBoundingBox(x=fx, y=fy, width=fw, height=fh)
                    padded = bbox.pad_and_clip(self.config.face_padding, img_width=w, img_height=h)
                    raw_detections.append((padded, conf))

            # Sort by area descending (largest faces first)
            raw_detections.sort(key=lambda item: item[0].area, reverse=True)

            for idx, (p_bbox, conf) in enumerate(raw_detections[: self.config.max_faces]):
                detections.append(FaceDetection(face_id=idx + 1, bbox=p_bbox, confidence=conf))

        # Robust Fallback: If YuNet detected 0 faces, try Haar cascade so non-frontal / challenging faces are not skipped
        if len(detections) == 0 and hasattr(cv2, "CascadeClassifier"):
            try:
                if not hasattr(self, "_fallback_haar") or self._fallback_haar is None:
                    self._fallback_haar = HaarCascadeFaceDetector(self.config)
                haar_detections = self._fallback_haar.detect(image_rgb)
                if haar_detections:
                    return haar_detections
            except Exception as exc:
                logging.debug(f"Haar cascade fallback notice: {exc}")

        return detections


class HaarCascadeFaceDetector(BaseFaceDetector):
    """Face detector using OpenCV Haar Cascade or fallback to YuNet if CascadeClassifier is unavailable."""

    def __init__(self, config: FaceDetectionConfig | None = None) -> None:
        """Initialize Haar Cascade detector with fallback support."""
        self.config = config or FaceDetectionConfig()
        self._fallback_yunet: YuNetFaceDetector | None = None
        self.detector: Any = None

        if not hasattr(cv2, "CascadeClassifier"):
            logging.info(
                "OpenCV CascadeClassifier unavailable in current cv2 build. Using YuNet backend."
            )
            self._fallback_yunet = YuNetFaceDetector(config=self.config)
            return

        # Locate XML
        cv2_data = getattr(cv2, "data", None)
        cv2_path = (
            Path(cv2_data.haarcascades) / "haarcascade_frontalface_default.xml"
            if cv2_data and hasattr(cv2_data, "haarcascades")
            else Path("nonexistent")
        )
        if BUNDLED_HAAR_PATH.exists():
            cascade_path = str(BUNDLED_HAAR_PATH)
        elif cv2_path.exists():
            cascade_path = str(cv2_path)
        else:
            cascade_path = ""

        if cascade_path:
            self.detector = cv2.CascadeClassifier(cascade_path)
            if self.detector.empty():
                self.detector = None

        if self.detector is None:
            self._fallback_yunet = YuNetFaceDetector(config=self.config)

    def detect(self, image_rgb: np.ndarray) -> list[FaceDetection]:
        """Detect faces using Haar cascade or fallback."""
        if self._fallback_yunet is not None or self.detector is None:
            if self._fallback_yunet is None:
                self._fallback_yunet = YuNetFaceDetector(config=self.config)
            return self._fallback_yunet.detect(image_rgb)

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image of shape [H, W, 3], got {image_rgb.shape}")

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        min_w, min_h = self.config.min_face_size

        rects = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.config.haar_scale_factor,
            minNeighbors=self.config.haar_min_neighbors,
            minSize=(min_w, min_h),
        )

        detections: list[FaceDetection] = []
        if len(rects) == 0:
            return detections

        raw_detections = []
        for x, y, fw, fh in rects:
            bbox = FaceBoundingBox(x=x, y=y, width=fw, height=fh)
            padded = bbox.pad_and_clip(self.config.face_padding, img_width=w, img_height=h)
            raw_detections.append((padded, 0.90))

        raw_detections.sort(key=lambda item: item[0].area, reverse=True)

        for idx, (p_bbox, conf) in enumerate(raw_detections[: self.config.max_faces]):
            detections.append(FaceDetection(face_id=idx + 1, bbox=p_bbox, confidence=conf))

        return detections


class PassThroughFaceDetector(BaseFaceDetector):
    """Pass-through detector treating the entire image (or centered crop) as one face."""

    def __init__(self, config: FaceDetectionConfig | None = None) -> None:
        """Initialize PassThroughFaceDetector."""
        self.config = config or FaceDetectionConfig()

    def detect(self, image_rgb: np.ndarray) -> list[FaceDetection]:
        """Return the entire image as one detected face."""
        h, w = image_rgb.shape[:2]
        bbox = FaceBoundingBox(x=0, y=0, width=w, height=h)
        return [FaceDetection(face_id=1, bbox=bbox, confidence=1.0)]


def create_face_detector(
    config: FaceDetectionConfig | None = None,
    detector_type: str | None = None,
) -> BaseFaceDetector:
    """Factory creating configured face detector instance."""
    cfg = config or FaceDetectionConfig()
    dtype = detector_type or cfg.detector_type

    if dtype == "yunet":
        return YuNetFaceDetector(config=cfg)
    elif dtype == "haar":
        return HaarCascadeFaceDetector(cfg)
    elif dtype == "passthrough":
        return PassThroughFaceDetector(cfg)
    else:
        raise ValueError(f"Unknown face detector type: {dtype}")
