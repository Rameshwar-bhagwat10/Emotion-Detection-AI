"""API tests for the Session lifecycle endpoints."""

from __future__ import annotations

import uuid

import pytest
from fastapi import status
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_session_lifecycle(client: AsyncClient, sample_face_image_bytes: bytes) -> None:
    """Verify session creation, prediction attachment, retrieval, and completion."""
    # 1. Create a new session
    create_payload = {"name": "Test Interview Session"}
    create_res = await client.post("/api/v1/sessions", json=create_payload)
    assert create_res.status_code == status.HTTP_201_CREATED
    session_data = create_res.json()
    session_id = session_data["id"]
    assert session_data["name"] == "Test Interview Session"
    assert session_data["status"] == "active"
    assert session_data["ended_at"] is None

    # 2. Get session details
    get_res = await client.get(f"/api/v1/sessions/{session_id}")
    assert get_res.status_code == status.HTTP_200_OK
    assert get_res.json()["id"] == session_id

    # 3. Post a prediction attached to this session
    files = {"image": ("face.png", sample_face_image_bytes, "image/png")}
    pred_res = await client.post(
        "/api/v1/predictions", files=files, data={"session_id": session_id}
    )
    assert pred_res.status_code == status.HTTP_200_OK
    assert pred_res.json()["session_id"] == session_id

    # 4. Get session predictions history
    history_res = await client.get(f"/api/v1/sessions/{session_id}/predictions")
    assert history_res.status_code == status.HTTP_200_OK
    predictions = history_res.json()
    assert isinstance(predictions, list)
    assert len(predictions) >= 1
    assert predictions[0]["session_id"] == session_id

    # 5. End the session
    end_res = await client.post(f"/api/v1/sessions/{session_id}/end")
    assert end_res.status_code == status.HTTP_200_OK
    ended_session = end_res.json()
    assert ended_session["status"] == "completed"
    assert ended_session["ended_at"] is not None

    # 6. List sessions
    list_res = await client.get("/api/v1/sessions?limit=10")
    assert list_res.status_code == status.HTTP_200_OK
    list_data = list_res.json()
    assert "sessions" in list_data
    assert list_data["total"] >= 1


@pytest.mark.asyncio
async def test_session_not_found(client: AsyncClient) -> None:
    """Verify GET and POST on unknown session ID returns HTTP 404."""
    random_id = str(uuid.uuid4())
    get_res = await client.get(f"/api/v1/sessions/{random_id}")
    assert get_res.status_code == status.HTTP_404_NOT_FOUND

    end_res = await client.post(f"/api/v1/sessions/{random_id}/end")
    assert end_res.status_code == status.HTTP_404_NOT_FOUND
