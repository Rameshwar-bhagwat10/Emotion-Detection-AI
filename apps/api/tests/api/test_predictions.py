"""API and functional tests for the Prediction endpoint."""

from __future__ import annotations

import uuid

import pytest
from fastapi import status
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_prediction_with_valid_image(
    client: AsyncClient, sample_face_image_bytes: bytes
) -> None:
    """Verify POST /api/v1/predictions successfully runs inference and returns structured response."""
    files = {"image": ("test_face.png", sample_face_image_bytes, "image/png")}
    response = await client.post("/api/v1/predictions", files=files)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert "status" in data
    assert data["status"].upper() in ["SUCCESS", "NO_FACE_DETECTED"]
    assert "request_id" in data
    assert "faces_detected" in data
    assert "faces" in data
    assert "timing" in data
    assert "model_info" in data
    assert data["model_info"]["model_name"] == "champion-pruning-30"

    # If faces detected, check probabilities and bounding boxes
    if data["faces_detected"] > 0:
        face = data["faces"][0]
        assert "face_id" in face
        assert "bbox" in face
        assert "emotion" in face
        assert "confidence" in face
        assert 0.0 <= face["confidence"] <= 1.0
        assert "probabilities" in face
        probs = face["probabilities"]
        assert len(probs) == 7
        assert abs(sum(probs.values()) - 1.0) < 1e-4


@pytest.mark.asyncio
async def test_prediction_with_blank_no_face_image(
    client: AsyncClient, sample_blank_image_bytes: bytes
) -> None:
    """Verify POST /api/v1/predictions with zero faces returns no_face_detected without crashing."""
    files = {"image": ("blank.png", sample_blank_image_bytes, "image/png")}
    response = await client.post("/api/v1/predictions", files=files)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"].upper() == "NO_FACE_DETECTED"
    assert data["faces_detected"] == 0
    assert data["faces"] == []


@pytest.mark.asyncio
async def test_prediction_with_corrupted_image(
    client: AsyncClient, sample_corrupt_image_bytes: bytes
) -> None:
    """Verify POST /api/v1/predictions rejects corrupted bytes with HTTP 400."""
    files = {"image": ("corrupted.png", sample_corrupt_image_bytes, "image/png")}
    response = await client.post("/api/v1/predictions", files=files)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "INVALID_IMAGE"


@pytest.mark.asyncio
async def test_prediction_with_unsupported_format(client: AsyncClient) -> None:
    """Verify POST /api/v1/predictions rejects unsupported file extensions with HTTP 415."""
    files = {"image": ("document.pdf", b"%PDF-1.4 fake pdf data", "application/pdf")}
    response = await client.post("/api/v1/predictions", files=files)

    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


@pytest.mark.asyncio
async def test_prediction_with_oversized_image(client: AsyncClient) -> None:
    """Verify POST /api/v1/predictions rejects payloads exceeding size limit with HTTP 413."""
    oversized_bytes = b"0" * (11 * 1024 * 1024)  # 11MB exceeds default 10MB limit
    files = {"image": ("large.png", oversized_bytes, "image/png")}
    response = await client.post("/api/v1/predictions", files=files)

    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "PAYLOAD_TOO_LARGE"


@pytest.mark.asyncio
async def test_get_prediction_by_id_and_not_found(
    client: AsyncClient, sample_face_image_bytes: bytes
) -> None:
    """Verify GET /api/v1/predictions/{id} retrieves stored prediction and returns 404 for unknown."""
    # 1. Create prediction
    files = {"image": ("face.png", sample_face_image_bytes, "image/png")}
    post_res = await client.post("/api/v1/predictions", files=files)
    assert post_res.status_code == status.HTTP_200_OK
    pred_data = post_res.json()
    pred_id = pred_data.get("prediction_id")

    if pred_id:
        # 2. Retrieve prediction
        get_res = await client.get(f"/api/v1/predictions/{pred_id}")
        assert get_res.status_code == status.HTTP_200_OK
        detail = get_res.json()
        assert detail["id"] == pred_id
        assert detail["status"] == pred_data["status"]
        assert detail["model_version"] == "champion-pruning-30"

    # 3. Test 404 for random UUID
    random_uuid = str(uuid.uuid4())
    not_found_res = await client.get(f"/api/v1/predictions/{random_uuid}")
    assert not_found_res.status_code == status.HTTP_404_NOT_FOUND
    err = not_found_res.json()
    assert err["error"]["code"] == "NOT_FOUND"
