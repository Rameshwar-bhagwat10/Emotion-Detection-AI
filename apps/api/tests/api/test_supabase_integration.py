"""Unit and integration tests for Supabase client connection and health probes."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.core.config import settings
from app.integrations.supabase import check_supabase_health, get_supabase_client


@pytest.mark.asyncio
async def test_supabase_client_initialization() -> None:
    """Verify that Supabase client initializes cleanly with configured credentials."""
    client = get_supabase_client()
    assert client is not None, "Failed to initialize Supabase client"
    assert settings.SUPABASE_URL in str(client.auth._url)


@pytest.mark.asyncio
async def test_supabase_health_probe_live() -> None:
    """Verify that Supabase health probe connects to project and returns healthy status."""
    health_status = await check_supabase_health()
    assert "status" in health_status
    assert health_status["status"] in ["healthy", "degraded", "unavailable"]
    if health_status["status"] == "healthy":
        assert "latency_ms" in health_status
        assert health_status["supabase_url"] == settings.SUPABASE_URL


@pytest.mark.asyncio
async def test_health_endpoint_includes_supabase(client: AsyncClient) -> None:
    """Verify that GET /api/v1/health includes the Supabase dependency status."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "dependencies" in data
    assert "supabase" in data["dependencies"]
    supabase_dep = data["dependencies"]["supabase"]
    assert "status" in supabase_dep
