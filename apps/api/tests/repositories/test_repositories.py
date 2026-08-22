"""Unit and integration tests for SQLAlchemy 2.x repositories."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.analysis_session import SessionRepository
from app.db.repositories.prediction import PredictionRepository
from app.db.repositories.user import UserRepository


@pytest.mark.asyncio
async def test_user_repository_crud(db_session: AsyncSession) -> None:
    """Verify UserRepository creation and lookup."""
    repo = UserRepository()
    email = f"user_{uuid.uuid4().hex[:8]}@example.com"

    # Create user
    user = await repo.create(db_session, email=email, full_name="Test User")
    await db_session.commit()

    assert user.id is not None
    assert user.email == email
    assert user.full_name == "Test User"
    assert user.is_active is True

    # Lookup by ID
    fetched_by_id = await repo.get_by_id(db_session, user.id)
    assert fetched_by_id is not None
    assert fetched_by_id.email == email

    # Lookup by Email
    fetched_by_email = await repo.get_by_email(db_session, email)
    assert fetched_by_email is not None
    assert fetched_by_email.id == user.id


@pytest.mark.asyncio
async def test_session_repository_lifecycle(db_session: AsyncSession) -> None:
    """Verify SessionRepository create, list, and state transition."""
    user_repo = UserRepository()
    session_repo = SessionRepository()

    user = await user_repo.create(db_session, email="session_user@example.com")
    await db_session.commit()

    # Create session
    session_record = await session_repo.create(db_session, user_id=user.id, name="Active Session")
    await db_session.commit()

    assert session_record.status == "active"
    assert session_record.user_id == user.id
    assert session_record.ended_at is None

    # End session
    ended_session = await session_repo.end_session(
        db_session, session_record.id, status="completed"
    )
    await db_session.commit()

    assert ended_session is not None
    assert ended_session.status == "completed"
    assert ended_session.ended_at is not None

    # List sessions
    sessions = await session_repo.list_sessions(db_session, user_id=user.id)
    assert len(sessions) == 1
    assert sessions[0].id == session_record.id


@pytest.mark.asyncio
async def test_prediction_repository_persistence_and_faces(db_session: AsyncSession) -> None:
    """Verify PredictionRepository transactional persistence with multiple faces."""
    pred_repo = PredictionRepository()
    session_repo = SessionRepository()

    session_record = await session_repo.create(db_session, name="Pred Session")
    await db_session.commit()

    faces_data = [
        {
            "face_id": 1,
            "bbox_x": 10,
            "bbox_y": 20,
            "bbox_width": 50,
            "bbox_height": 50,
            "detection_confidence": 0.95,
            "emotion": "happy",
            "confidence": 0.92,
            "is_uncertain": False,
            "probabilities": {
                "angry": 0.01,
                "disgust": 0.01,
                "fear": 0.01,
                "happy": 0.92,
                "sad": 0.02,
                "surprise": 0.02,
                "neutral": 0.01,
            },
        },
        {
            "face_id": 2,
            "bbox_x": 100,
            "bbox_y": 120,
            "bbox_width": 48,
            "bbox_height": 48,
            "detection_confidence": 0.88,
            "emotion": "neutral",
            "confidence": 0.65,
            "is_uncertain": False,
            "probabilities": {
                "angry": 0.05,
                "disgust": 0.02,
                "fear": 0.03,
                "happy": 0.10,
                "sad": 0.10,
                "surprise": 0.05,
                "neutral": 0.65,
            },
        },
    ]

    # Create prediction
    pred = await pred_repo.create_prediction_with_faces(
        session=db_session,
        prediction_id=uuid.uuid4(),
        request_id="req-test-12345",
        model_version="champion-pruning-30",
        status="success",
        faces_detected=2,
        processing_time_ms=12.5,
        image_width=640,
        image_height=480,
        session_id=session_record.id,
        faces_data=faces_data,
    )
    await db_session.commit()

    assert pred.id is not None
    assert pred.faces_detected == 2

    # Fetch by ID with faces
    fetched = await pred_repo.get_by_id(db_session, pred.id, load_faces=True)
    assert fetched is not None
    assert len(fetched.faces) == 2
    assert fetched.faces[0].emotion == "happy"
    assert fetched.faces[1].emotion == "neutral"

    # Fetch by request ID
    by_req = await pred_repo.get_by_request_id(db_session, "req-test-12345")
    assert by_req is not None
    assert by_req.id == pred.id
