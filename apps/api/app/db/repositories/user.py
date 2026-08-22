"""Repository for User database operations."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.user import User


class UserRepository:
    """Data-access repository for User domain model."""

    async def get_by_id(self, session: AsyncSession, user_id: uuid.UUID) -> User | None:
        """Fetch a User by UUID."""
        stmt = select(User).where(User.id == user_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_email(self, session: AsyncSession, email: str) -> User | None:
        """Fetch a User by unique email address."""
        stmt = select(User).where(User.email == email)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(
        self,
        session: AsyncSession,
        email: str | None = None,
        full_name: str | None = None,
        user_id: uuid.UUID | None = None,
    ) -> User:
        """Create and persist a new User entity."""
        user = User(
            id=user_id or uuid.uuid4(),
            email=email,
            full_name=full_name,
            is_active=True,
        )
        session.add(user)
        await session.flush()
        return user
