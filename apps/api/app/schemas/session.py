"""Pydantic schemas for session lifecycle management."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SessionCreateRequest(BaseModel):
    """Payload for creating a new analysis session."""

    name: str | None = Field(
        default=None, max_length=255, description="Optional friendly session name"
    )
    user_id: uuid.UUID | None = Field(default=None, description="Optional associated user UUID")


class SessionResponse(BaseModel):
    """Standard analysis session response model."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(description="Unique session UUID")
    user_id: uuid.UUID | None = Field(default=None, description="Associated user UUID")
    name: str | None = Field(default=None, description="Session name")
    status: str = Field(description="Session state: active, completed, cancelled")
    started_at: datetime = Field(description="Session start timestamp")
    ended_at: datetime | None = Field(default=None, description="Session end timestamp")
    created_at: datetime = Field(description="Record creation timestamp")


class SessionListResponse(BaseModel):
    """Paginated list of sessions."""

    sessions: list[SessionResponse] = Field(description="List of session objects")
    total: int = Field(ge=0, description="Total number of sessions matching query")
