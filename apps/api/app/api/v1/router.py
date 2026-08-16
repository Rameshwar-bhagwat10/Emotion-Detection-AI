"""Central API v1 Router aggregating all v1 sub-routers."""

from fastapi import APIRouter

from app.api.v1 import health

api_v1_router = APIRouter()

# Register core v1 endpoints
api_v1_router.include_router(health.router)
