"""API endpoints for analysis session lifecycle management."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db, get_session_service
from app.schemas.prediction import PredictionDetailSchema
from app.schemas.session import (
    SessionCreateRequest,
    SessionListResponse,
    SessionResponse,
)
from app.services.session_service import SessionService

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post(
    "",
    summary="Create Analysis Session",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    response_description="Newly created analysis session",
)
async def create_session(
    payload: SessionCreateRequest,
    db: AsyncSession = Depends(get_db),
    service: SessionService = Depends(get_session_service),
) -> SessionResponse:
    """Create a new session record for grouping image predictions."""
    return await service.create_session(db=db, request=payload)


@router.get(
    "",
    summary="List Analysis Sessions",
    response_model=SessionListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_sessions(
    user_id: uuid.UUID | None = Query(default=None, description="Filter by user UUID"),
    limit: int = Query(default=50, ge=1, le=100, description="Max items to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    db: AsyncSession = Depends(get_db),
    service: SessionService = Depends(get_session_service),
) -> SessionListResponse:
    """List sessions ordered chronologically descending."""
    return await service.list_sessions(db=db, user_id=user_id, limit=limit, offset=offset)


@router.get(
    "/{session_id}",
    summary="Get Session Details",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
)
async def get_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    service: SessionService = Depends(get_session_service),
) -> SessionResponse:
    """Retrieve session details by UUID."""
    return await service.get_session(db=db, session_id=session_id)


@router.post(
    "/{session_id}/end",
    summary="End Analysis Session",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
)
async def end_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    service: SessionService = Depends(get_session_service),
) -> SessionResponse:
    """Mark an active session as completed."""
    return await service.end_session(db=db, session_id=session_id)


@router.get(
    "/{session_id}/predictions",
    summary="Get Session Prediction History",
    response_model=list[PredictionDetailSchema],
    status_code=status.HTTP_200_OK,
)
async def get_session_predictions(
    session_id: uuid.UUID,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    service: SessionService = Depends(get_session_service),
) -> list[PredictionDetailSchema]:
    """Retrieve all predictions stored under a session."""
    result: list[PredictionDetailSchema] = await service.get_session_predictions(
        db=db, session_id=session_id, limit=limit, offset=offset
    )
    return result
