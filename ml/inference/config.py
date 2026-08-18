"""Configuration schemas and loader for the Phase 09 Inference Pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DeviceConfig:
    """Device selection configuration."""

    strategy: str = "auto"  # "auto" | "cpu" | "cuda"
    allow_fallback_to_cpu: bool = True

    def validate(self) -> None:
        """Validate device settings."""
        if self.strategy not in ("auto", "cpu", "cuda"):
            raise ValueError(
                f"Invalid device strategy: {self.strategy}. Must be 'auto', 'cpu', or 'cuda'."
            )


@dataclass
class ModelInferenceConfig:
    """Model checkpoint, format, and expected tensor dimensions configuration."""

    champion_dir: str = "artifacts/optimized/champion"
    metadata_file: str = "artifacts/optimized/champion/metadata.json"
    weights_file: str = "artifacts/optimized/champion/model.pt"
    onnx_file: str = "artifacts/optimized/champion/model.onnx"
    model_format: str = "pytorch"  # "pytorch" | "onnx"
    expected_architecture: str = "resnet18"
    input_size: list[int] = field(default_factory=lambda: [48, 48])
    input_channels: int = 1
    mean: list[float] = field(default_factory=lambda: [0.507743])
    std: list[float] = field(default_factory=lambda: [0.255009])

    def validate(self) -> None:
        """Validate model configuration values."""
        if self.model_format not in ("pytorch", "onnx"):
            raise ValueError(
                f"Unsupported model_format '{self.model_format}'. Must be 'pytorch' or 'onnx'."
            )
        if len(self.input_size) != 2 or any(d <= 0 for d in self.input_size):
            raise ValueError(f"input_size must be positive [H, W], got {self.input_size}")
        if self.input_channels not in (1, 3):
            raise ValueError(f"input_channels must be 1 or 3, got {self.input_channels}")
        if any(s <= 0 for s in self.std):
            raise ValueError(f"Standard deviation values must be strictly positive, got {self.std}")


@dataclass
class FaceDetectionConfig:
    """Face detection subsystem configuration."""

    detector_type: str = "yunet"  # "yunet" | "haar" | "passthrough"
    haar_scale_factor: float = 1.1
    haar_min_neighbors: int = 5
    min_face_size: list[int] = field(default_factory=lambda: [20, 20])
    confidence_threshold: float = 0.50
    nms_threshold: float = 0.30
    face_padding: float = 0.15  # Expand bounding box by 15% before cropping
    max_faces: int = 20

    def validate(self) -> None:
        """Validate face detector settings."""
        if self.detector_type not in ("haar", "yunet", "passthrough"):
            raise ValueError(f"Unsupported detector_type: {self.detector_type}")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {self.confidence_threshold}"
            )
        if not (0.0 <= self.nms_threshold <= 1.0):
            raise ValueError(f"nms_threshold must be in [0.0, 1.0], got {self.nms_threshold}")
        if self.face_padding < 0.0:
            raise ValueError(f"face_padding must be >= 0.0, got {self.face_padding}")
        if self.max_faces <= 0:
            raise ValueError(f"max_faces must be > 0, got {self.max_faces}")
        if self.haar_scale_factor <= 1.0:
            raise ValueError(f"haar_scale_factor must be > 1.0, got {self.haar_scale_factor}")
        if self.haar_min_neighbors < 0:
            raise ValueError(f"haar_min_neighbors must be >= 0, got {self.haar_min_neighbors}")


@dataclass
class InferencePipelineConfig:
    """Top-level configuration for the emotion inference pipeline."""

    version: str = "1.0.0"
    device: DeviceConfig = field(default_factory=DeviceConfig)
    model: ModelInferenceConfig = field(default_factory=ModelInferenceConfig)
    face_detection: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    batch_size: int = 16
    warmup_iterations: int = 3
    confidence_threshold: float = 0.40
    uncertain_label: str = "uncertain"
    enable_timing: bool = True
    max_image_dimension: int = 4096
    classes: list[str] = field(
        default_factory=lambda: [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        ]
    )

    def validate(self) -> None:
        """Validate entire inference pipeline configuration."""
        self.device.validate()
        self.model.validate()
        self.face_detection.validate()

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.warmup_iterations < 0:
            raise ValueError(f"warmup_iterations must be >= 0, got {self.warmup_iterations}")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {self.confidence_threshold}"
            )
        if self.max_image_dimension <= 0:
            raise ValueError(f"max_image_dimension must be > 0, got {self.max_image_dimension}")
        if len(self.classes) != 7:
            raise ValueError(f"Expected 7 classes, got {len(self.classes)}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferencePipelineConfig:
        """Build InferencePipelineConfig from dictionary."""
        dev_data = data.get("device", {})
        device_cfg = DeviceConfig(
            strategy=dev_data.get("strategy", "auto"),
            allow_fallback_to_cpu=dev_data.get("allow_fallback_to_cpu", True),
        )

        mod_data = data.get("model", {})
        norm_data = mod_data.get("normalization", {})
        model_cfg = ModelInferenceConfig(
            champion_dir=mod_data.get("champion_dir", "artifacts/optimized/champion"),
            metadata_file=mod_data.get(
                "metadata_file", "artifacts/optimized/champion/metadata.json"
            ),
            weights_file=mod_data.get("weights_file", "artifacts/optimized/champion/model.pt"),
            onnx_file=mod_data.get("onnx_file", "artifacts/optimized/champion/model.onnx"),
            model_format=mod_data.get("model_format", "pytorch"),
            expected_architecture=mod_data.get("expected_architecture", "resnet18"),
            input_size=mod_data.get("input_size", [48, 48]),
            input_channels=mod_data.get("input_channels", 1),
            mean=norm_data.get("mean", [0.507743]),
            std=norm_data.get("std", [0.255009]),
        )

        fd_data = data.get("face_detection", {})
        fd_cfg = FaceDetectionConfig(
            detector_type=fd_data.get("detector_type", "haar"),
            haar_scale_factor=float(fd_data.get("haar_scale_factor", 1.1)),
            haar_min_neighbors=int(fd_data.get("haar_min_neighbors", 5)),
            min_face_size=fd_data.get("min_face_size", [20, 20]),
            confidence_threshold=float(fd_data.get("confidence_threshold", 0.50)),
            nms_threshold=float(fd_data.get("nms_threshold", 0.30)),
            face_padding=float(fd_data.get("face_padding", 0.15)),
            max_faces=int(fd_data.get("max_faces", 20)),
        )

        inf_data = data.get("inference", {})

        cfg = cls(
            version=str(data.get("version", "1.0.0")),
            device=device_cfg,
            model=model_cfg,
            face_detection=fd_cfg,
            batch_size=int(inf_data.get("batch_size", 16)),
            warmup_iterations=int(inf_data.get("warmup_iterations", 3)),
            confidence_threshold=float(inf_data.get("confidence_threshold", 0.40)),
            uncertain_label=str(inf_data.get("uncertain_label", "uncertain")),
            enable_timing=bool(inf_data.get("enable_timing", True)),
            max_image_dimension=int(inf_data.get("max_image_dimension", 4096)),
            classes=data.get(
                "classes",
                ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
            ),
        )
        cfg.validate()
        return cfg


def load_inference_config(path: str | Path | None = None) -> InferencePipelineConfig:
    """Load and validate inference configuration from YAML file or defaults."""
    target_path = (
        Path(path)
        if path
        else Path(__file__).resolve().parent.parent / "configs" / "inference.yaml"
    )
    if not target_path.exists():
        cfg = InferencePipelineConfig()
        cfg.validate()
        return cfg

    with open(target_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return InferencePipelineConfig.from_dict(data)
