"""End-to-end integration tests for complete FastAPI -> Phase 09 Inference -> DB flow."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import status
from httpx import AsyncClient

from app.db.repositories.prediction import PredictionRepository


@pytest.mark.asyncio
async def test_full_end_to_end_prediction_flow(
    client: AsyncClient, sample_face_image_bytes: bytes
) -> None:
    """Execute complete workflow: Session Creation -> Image Upload -> ML Inference -> DB Persistence -> Retrieval."""
    # 1. Start an analysis session
    session_res = await client.post("/api/v1/sessions", json={"name": "E2E Integration Session"})
    assert session_res.status_code == status.HTTP_201_CREATED
    session_id = session_res.json()["id"]

    # 2. Upload image for facial emotion prediction
    files = {"image": ("e2e_face.png", sample_face_image_bytes, "image/png")}
    data = {"session_id": session_id}
    predict_res = await client.post("/api/v1/predictions", files=files, data=data)
    assert predict_res.status_code == status.HTTP_200_OK
    pred_data = predict_res.json()

    assert pred_data["session_id"] == session_id
    assert "prediction_id" in pred_data
    assert "request_id" in pred_data
    assert pred_data["model_info"]["model_name"] == "champion-pruning-30"
    assert pred_data["timing"]["total_ms"] > 0.0

    prediction_id = pred_data["prediction_id"]

    # 3. Retrieve prediction details by ID
    get_pred_res = await client.get(f"/api/v1/predictions/{prediction_id}")
    assert get_pred_res.status_code == status.HTTP_200_OK
    pred_detail = get_pred_res.json()
    assert pred_detail["id"] == prediction_id
    assert pred_detail["session_id"] == session_id

    # 4. Retrieve session predictions history
    history_res = await client.get(f"/api/v1/sessions/{session_id}/predictions")
    assert history_res.status_code == status.HTTP_200_OK
    history = history_res.json()
    assert len(history) == 1
    assert history[0]["id"] == prediction_id

    # 5. Complete session
    end_res = await client.post(f"/api/v1/sessions/{session_id}/end")
    assert end_res.status_code == status.HTTP_200_OK
    assert end_res.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_transaction_rollback_on_persistence_error(
    client: AsyncClient, sample_face_image_bytes: bytes
) -> None:
    """Verify that if database persistence fails, the transaction is safely rolled back and 500 returned."""
    files = {"image": ("face.png", sample_face_image_bytes, "image/png")}

    with patch.object(
        PredictionRepository,
        "create_prediction_with_faces",
        side_effect=RuntimeError("Simulated DB connection failure"),
    ):
        response = await client.post("/api/v1/predictions", files=files)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "DATABASE_ERROR"
