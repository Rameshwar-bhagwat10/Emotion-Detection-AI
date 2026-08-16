"""Redis client connection and health check integration."""

from typing import Any

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_redis_client: aioredis.Redis | None = None


def get_redis_client() -> aioredis.Redis:
    """Obtain or initialize the global async Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=3,
        )
    return _redis_client


async def close_redis_client() -> None:
    """Close the global Redis client connection pool."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None


async def check_redis_health() -> dict[str, Any]:
    """Verify Redis availability via ping.

    Returns:
        Dictionary indicating status or error message.
    """
    try:
        client = get_redis_client()
        pong = await client.ping()
        if pong:
            return {
                "status": "healthy",
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
            }
        return {
            "status": "unhealthy",
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
        }
    except Exception as exc:
        logger.warning(f"Redis health check failed: {str(exc)}")
        return {
            "status": "unavailable",
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "error": "Could not connect to Redis instance",
        }
