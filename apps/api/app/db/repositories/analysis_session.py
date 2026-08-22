"""Repository for AnalysisSession database operations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.analysis_session import AnalysisSession


class SessionRepository:
    """Data-access repository for AnalysisSession domain model."""

    async def get_by_id(
        self, session: AsyncSession, session_id: uuid.UUID, load_predictions: bool = False
    ) -> AnalysisSession | None:
        """Fetch an AnalysisSession by UUID."""
        stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        if load_predictions:
            stmt = stmt.options(selectinload(AnalysisSession.predictions))
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(
        self,
        session: AsyncSession,
        user_id: uuid.UUID | None = None,
        name: str | None = None,
        session_id: uuid.UUID | None = None,
    ) -> AnalysisSession:
        """Create and persist a new AnalysisSession entity."""
        analysis_session = AnalysisSession(
            id=session_id or uuid.uuid4(),
            user_id=user_id,
            name=name,
            status="active",
            started_at=datetime.now(UTC),
        )
        session.add(analysis_session)
        await session.flush()
        return analysis_session

    async def end_session(
        self, session: AsyncSession, session_id: uuid.UUID, status: str = "completed"
    ) -> AnalysisSession | None:
        """Mark an active session as ended/completed/cancelled."""
        analysis_session = await self.get_by_id(session, session_id)
        if analysis_session is None:
            return None
        analysis_session.status = status
        analysis_session.ended_at = datetime.now(UTC)
        await session.flush()
        return analysis_session

    async def list_sessions(
        self,
        session: AsyncSession,
        user_id: uuid.UUID | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AnalysisSession]:
        """List sessions ordered by creation date descending."""
        stmt = (
            select(AnalysisSession)
            .order_by(AnalysisSession.started_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if user_id is not None:
            stmt = stmt.where(AnalysisSession.user_id == user_id)
        result = await session.execute(stmt)
        return list(result.scalars().all())
