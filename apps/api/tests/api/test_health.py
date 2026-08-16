"""Tests for API health endpoints."""

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import app


@pytest.mark.asyncio
async def test_health_endpoint_returns_200():
    """Verify that GET /api/v1/health returns HTTP 200 with structured status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "status" in data
    assert data["status"] in ["ok", "degraded"]
    assert data["service"] == settings.APP_NAME
    assert "version" in data
    assert "environment" in data
    assert "dependencies" in data
    assert "database" in data["dependencies"]
    assert "redis" in data["dependencies"]


@pytest.mark.asyncio
async def test_liveness_endpoint_returns_200():
    """Verify that GET /api/v1/health/live returns HTTP 200."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health/live")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == settings.APP_NAME


@pytest.mark.asyncio
async def test_root_endpoint_returns_200():
    """Verify root endpoint metadata."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "message" in data
    assert "health" in data
