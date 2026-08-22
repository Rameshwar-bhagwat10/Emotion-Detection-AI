"""In-memory real-time video frame validation, decoding, and preprocessing."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from app.core.config import settings
from app.core.exceptions import InvalidImageError, PayloadTooLargeError
from app.core.logging import get_logger

logger = get_logger(__name__)


class FrameProcessor:
    """Handles in-memory validation and decoding of binary / base64 video frames."""

    def __init__(
        self,
        max_frame_size_mb: int | None = None,
    ) -> None:
        """Initialize FrameProcessor."""
        self.max_bytes = (max_frame_size_mb or settings.REALTIME_MAX_FRAME_SIZE_MB) * 1024 * 1024

    def decode_frame_bytes(self, raw_bytes: bytes) -> tuple[np.ndarray, int, int]:
        """Decode raw binary JPEG/PNG bytes into a BGR OpenCV NumPy array.

        Args:
            raw_bytes: Raw binary image payload.

        Returns:
            Tuple of `(image_bgr_array, width, height)`.

        Raises:
            PayloadTooLargeError: If raw bytes exceed configured threshold.
            InvalidImageError: If bytes cannot be decoded into a valid 3-channel image.
        """
        if not raw_bytes:
            raise InvalidImageError("Empty frame buffer received.")

        if len(raw_bytes) > self.max_bytes:
            raise PayloadTooLargeError(
                f"Frame size {len(raw_bytes)} bytes exceeds maximum {self.max_bytes} bytes."
            )

        try:
            np_arr = np.frombuffer(raw_bytes, np.uint8)
            image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image_bgr is None or image_bgr.size == 0:
                raise InvalidImageError("Failed to decode frame bytes into an image array.")

            height, width = image_bgr.shape[:2]
            if width < 16 or height < 16:
                raise InvalidImageError(f"Frame dimensions too small: {width}x{height}.")

            return image_bgr, width, height

        except (PayloadTooLargeError, InvalidImageError):
            raise
        except Exception as exc:
            logger.warning(f"Frame decoding failure: {exc}")
            raise InvalidImageError(f"Corrupted or unsupported frame format: {exc}") from exc

    def decode_base64_frame(self, base64_str: str) -> tuple[bytes, np.ndarray, int, int]:
        """Decode a base64-encoded image string into raw bytes and a BGR image array.

        Args:
            base64_str: Base64 string, optionally prefixed with 'data:image/...;base64,'.

        Returns:
            Tuple of `(raw_bytes, image_bgr_array, width, height)`.
        """
        if not base64_str:
            raise InvalidImageError("Empty base64 frame string received.")

        # Strip data URL prefix if present
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]

        try:
            raw_bytes = base64.b64decode(base64_str)
        except Exception as exc:
            raise InvalidImageError(f"Invalid base64 encoding: {exc}") from exc

        image_bgr, width, height = self.decode_frame_bytes(raw_bytes)
        return raw_bytes, image_bgr, width, height
