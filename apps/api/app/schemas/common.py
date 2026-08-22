"""Common Pydantic schemas for health, readiness, and error responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Structured error payload details."""

    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error description")
    request_id: str = Field(description="Unique correlation request ID")
    details: dict[str, Any] | None = Field(default=None, description="Optional extra error details")


class ErrorResponse(BaseModel):
    """Standardized top-level API error response."""

    error: ErrorDetail


class DependencyHealth(BaseModel):
    """Health status of an individual subsystem/dependency."""

    status: str = Field(description="Health status: healthy, unhealthy, unavailable")
    details: dict[str, Any] | None = Field(default=None, description="Optional diagnostic details")


class HealthResponse(BaseModel):
    """Response model for lightweight health probe."""

    status: str = Field(description="Overall service status: ok, degraded, unhealthy")
    service: str = Field(description="Application service name")
    version: str = Field(description="Application version string")
    environment: str = Field(description="Runtime environment (development, production, etc.)")
    dependencies: dict[str, Any] = Field(
        default_factory=dict, description="Subsystem health statuses"
    )


class ReadinessResponse(BaseModel):
    """Response model for deep readiness probe."""

    status: str = Field(description="Readiness status: ready or not_ready")
    service: str = Field(description="Application service name")
    version: str = Field(description="Application version string")
    model: str = Field(description="ML model readiness status: ready or not_ready")
    database: str = Field(description="Database connectivity status: ready or not_ready")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional readiness metadata"
    )
