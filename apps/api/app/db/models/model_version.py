"""SQLAlchemy domain model for ModelVersion registry entity."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import JSON, Boolean, Float, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ModelVersion(Base):
    """Registry metadata entity for tracking deployed ML models."""

    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )
    architecture: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    accuracy: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    macro_f1: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    is_champion: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )
