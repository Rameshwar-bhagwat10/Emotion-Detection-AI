"""Unit and API tests for Health, Readiness, and Liveness probes."""

from __future__ import annotations

import pytest
from fastapi import status
from httpx import AsyncClient

from app.core.config import settings


@pytest.mark.asyncio
async def test_health_probe_returns_200(client: AsyncClient) -> None:
    """Verify GET /api/v1/health returns HTTP 200 with structured status payload."""
    response = await client.get("/api/v1/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "status" in data
    assert data["status"] in ["ok", "degraded"]
    assert data["service"] == settings.APP_NAME
    assert "version" in data
    assert "dependencies" in data
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers


@pytest.mark.asyncio
async def test_liveness_probe_returns_200(client: AsyncClient) -> None:
    """Verify GET /api/v1/health/live returns fast HTTP 200."""
    response = await client.get("/api/v1/health/live")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == settings.APP_NAME


@pytest.mark.asyncio
async def test_readiness_probe_returns_structured_payload(client: AsyncClient) -> None:
    """Verify GET /api/v1/health/ready returns deep readiness verification."""
    response = await client.get("/api/v1/health/ready")
    assert response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]

    data = response.json()
    assert "status" in data
    assert "model" in data
    assert "database" in data
    assert data["model"] == "ready"
    assert "details" in data


@pytest.mark.asyncio
async def test_root_endpoint_metadata(client: AsyncClient) -> None:
    """Verify root / endpoint returns API navigation metadata."""
    response = await client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "message" in data
    assert "docs" in data
    assert "health" in data
