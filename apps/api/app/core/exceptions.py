"""Centralized exception definitions and FastAPI error handlers."""

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class AppException(Exception):
    """Base application exception."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class NotFoundException(AppException):
    """Resource not found exception."""

    def __init__(self, message: str = "Resource not found", details: dict[str, Any] | None = None):
        super().__init__(message=message, status_code=status.HTTP_404_NOT_FOUND, details=details)


class ValidationException(AppException):
    """Data validation exception."""

    def __init__(self, message: str = "Validation error", details: dict[str, Any] | None = None):
        super().__init__(
            message=message, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, details=details
        )


class DatabaseException(AppException):
    """Database operation exception."""

    def __init__(
        self, message: str = "Database operation failed", details: dict[str, Any] | None = None
    ):
        super().__init__(
            message=message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, details=details
        )


class ServiceUnavailableException(AppException):
    """Downstream or external service unavailable."""

    def __init__(self, message: str = "Service unavailable", details: dict[str, Any] | None = None):
        super().__init__(
            message=message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, details=details
        )


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers with the FastAPI application."""

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.message,
                    "type": exc.__class__.__name__,
                    "details": exc.details,
                    "path": str(request.url.path),
                }
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "message": "An unexpected server error occurred.",
                    "type": "InternalServerError",
                    "path": str(request.url.path),
                }
            },
        )
