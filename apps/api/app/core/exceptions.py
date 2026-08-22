"""Centralized exception definitions and FastAPI error handlers."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AppException(Exception):
    """Base application exception."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class InvalidImageError(AppException):
    """Raised when an uploaded image is invalid, corrupted, or has bad dimensions."""

    def __init__(
        self, message: str = "Invalid image upload", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            code="INVALID_IMAGE",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class UnsupportedMediaTypeError(AppException):
    """Raised when an uploaded file format is not supported."""

    def __init__(
        self, message: str = "Unsupported media type", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            code="UNSUPPORTED_MEDIA_TYPE",
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            details=details,
        )


class PayloadTooLargeError(AppException):
    """Raised when an uploaded file exceeds the allowed size limit."""

    def __init__(
        self,
        message: str = "Payload exceeds maximum allowed size",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="PAYLOAD_TOO_LARGE",
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            details=details,
        )


class ResourceNotFoundError(AppException):
    """Raised when a requested resource (prediction, session, etc.) is not found."""

    def __init__(
        self, message: str = "Resource not found", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            code="NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class ModelNotReadyError(AppException):
    """Raised when the ML inference engine is not ready to process predictions."""

    def __init__(
        self,
        message: str = "ML inference engine is not ready",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code="MODEL_NOT_READY",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
        )


class InferenceError(AppException):
    """Raised when neural network or computer vision inference fails."""

    def __init__(
        self, message: str = "Inference execution failed", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            code="INFERENCE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class DatabaseException(AppException):
    """Raised when a database query or transaction fails."""

    def __init__(
        self, message: str = "Database operation failed", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            code="DATABASE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


# Backward compatibility aliases
NotFoundException = ResourceNotFoundError
ValidationException = InvalidImageError
ServiceUnavailableException = ModelNotReadyError


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers with the FastAPI application."""

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(
            f"[{request_id}] AppException ({exc.code}): {exc.message} on path {request.url.path}"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": request_id,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"[{request_id}] Validation error on {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid request parameters or payload",
                    "request_id": request_id,
                    "details": {"errors": exc.errors()},
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        code_map = {
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            413: "PAYLOAD_TOO_LARGE",
            415: "UNSUPPORTED_MEDIA_TYPE",
            500: "INTERNAL_SERVER_ERROR",
            503: "SERVICE_UNAVAILABLE",
        }
        code = code_map.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": code,
                    "message": str(exc.detail),
                    "request_id": request_id,
                }
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            f"[{request_id}] Unhandled server exception on {request.url.path}: {exc}", exc_info=True
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected internal server error occurred.",
                    "request_id": request_id,
                }
            },
        )
