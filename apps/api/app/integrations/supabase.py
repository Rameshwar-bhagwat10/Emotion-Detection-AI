"""Supabase integration provider for database, auth, and cloud storage."""

from __future__ import annotations

import time
from typing import Any

import httpx
from supabase import Client, create_client

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_supabase_client: Client | None = None


def get_supabase_client() -> Client | None:
    """Retrieve or initialize the singleton Supabase client.

    Returns:
        Configured Supabase Client instance, or None if credentials are not configured.
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    if not settings.SUPABASE_URL or not settings.SUPABASE_ANON_KEY:
        logger.warning("Supabase URL or Anon Key not configured in settings.")
        return None

    try:
        _supabase_client = create_client(
            supabase_url=settings.SUPABASE_URL,
            supabase_key=settings.SUPABASE_ANON_KEY,
        )
        logger.info(f"Initialized Supabase client for project: {settings.SUPABASE_URL}")
        return _supabase_client
    except Exception as exc:
        logger.error(f"Failed to initialize Supabase client: {exc}", exc_info=True)
        return None


async def check_supabase_health() -> dict[str, Any]:
    """Verify Supabase service reachability and auth/REST API health.

    Returns:
        Dictionary indicating status, latency, and endpoint details.
    """
    if not settings.SUPABASE_URL or not settings.SUPABASE_ANON_KEY:
        return {
            "status": "unconfigured",
            "message": "Supabase credentials not configured",
        }

    start_time = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=5.0) as http_client:
            headers = {
                "apikey": settings.SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
            }
            response = await http_client.get(
                f"{settings.SUPABASE_URL}/auth/v1/settings",
                headers=headers,
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        if response.status_code == 200:
            return {
                "status": "healthy",
                "supabase_url": settings.SUPABASE_URL,
                "latency_ms": latency_ms,
            }
        else:
            return {
                "status": "degraded",
                "supabase_url": settings.SUPABASE_URL,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.warning(f"Supabase health probe failed: {exc}")
        return {
            "status": "unavailable",
            "supabase_url": settings.SUPABASE_URL,
            "latency_ms": latency_ms,
            "error": str(exc),
        }
