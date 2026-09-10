"""End-to-end integration test suite for Phase 12 Application Integration.

Validates:
1. Service Health probe (/api/v1/health)
2. Readiness verification (/api/v1/health/ready)
3. Session lifecycle (create, get, list, end)
4. Full image inference flow (/api/v1/predictions) with real model & face detector
5. Multi-class probability distribution & bounding boxes
6. Database session correlation
7. Invalid input rejection
8. WebSocket handshake (/api/v1/realtime/emotion)
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
import sys

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.websockets import WebSocketDisconnect

# Ensure apps/api and root are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
API_DIR = ROOT_DIR / "apps" / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app, lifespan


@pytest.fixture(scope="module")
def event_loop():
    """Create a module-scoped event loop for async integration testing."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def client():
    """Provide an AsyncClient running inside the FastAPI application lifespan."""
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            yield ac


@pytest.mark.asyncio
async def test_health_probe(client: AsyncClient):
    """Test GET /api/v1/health returns HTTP 200 and structured status."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "service" in data
    assert "dependencies" in data
    assert "database" in data["dependencies"]


@pytest.mark.asyncio
async def test_readiness_probe(client: AsyncClient):
    """Test GET /api/v1/health/ready verifies ML engine and database readiness."""
    response = await client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model"] == "ready"
    assert data["database"] == "ready"
    assert "details" in data
    assert "model_version" in data["details"]


@pytest.mark.asyncio
async def test_session_lifecycle(client: AsyncClient):
    """Test session creation, retrieval, listing, and finalization."""
    # 1. Create session
    create_res = await client.post(
        "/api/v1/sessions",
        json={"name": "Integration Test Session"},
    )
    assert create_res.status_code == 201
    session_data = create_res.json()
    session_id = session_data["id"]
    assert session_data["name"] == "Integration Test Session"
    assert session_data["status"] == "active"

    # 2. Get session details
    get_res = await client.get(f"/api/v1/sessions/{session_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == session_id

    # 3. List sessions
    list_res = await client.get("/api/v1/sessions?limit=10")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1
    session_ids = [s["id"] for s in list_data["sessions"]]
    assert session_id in session_ids

    # 4. End session
    end_res = await client.post(f"/api/v1/sessions/{session_id}/end")
    assert end_res.status_code == 200
    assert end_res.json()["status"] == "completed"
    assert end_res.json()["ended_at"] is not None


@pytest.mark.asyncio
async def test_image_prediction_flow(client: AsyncClient):
    """Test image upload inference with real face photograph and session association."""
    # Create a session to associate with prediction
    sess_res = await client.post("/api/v1/sessions", json={"name": "Prediction Flow Test"})
    assert sess_res.status_code == 201
    session_id = sess_res.json()["id"]

    # Load actual test image from reports directory
    test_img_path = ROOT_DIR / "reports" / "random_and_internet_eval" / "internet_1_happy_smile_orig.jpg"
    assert test_img_path.exists(), f"Sample test image not found at {test_img_path}"

    with open(test_img_path, "rb") as f:
        img_bytes = f.read()

    # Call POST /api/v1/predictions
    files = {"image": ("test_face.jpg", img_bytes, "image/jpeg")}
    data = {"session_id": session_id}

    pred_res = await client.post("/api/v1/predictions", files=files, data=data)
    assert pred_res.status_code == 200
    pred_data = pred_res.json()

    # Validate schema integrity
    assert pred_data["status"].lower() == "success"
    assert pred_data["faces_detected"] >= 1
    assert len(pred_data["faces"]) >= 1

    first_face = pred_data["faces"][0]
    assert first_face["face_id"] >= 1
    assert "bbox" in first_face
    assert first_face["bbox"]["width"] > 0
    assert first_face["bbox"]["height"] > 0
    assert 0.0 <= first_face["detection_confidence"] <= 1.0

    # Emotion prediction and softmax distribution
    assert first_face["emotion"] in (
        "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "uncertain"
    )
    assert 0.0 <= first_face["confidence"] <= 1.0
    assert isinstance(first_face["is_uncertain"], bool)

    probs = first_face["probabilities"]
    assert len(probs) == 7
    assert "happy" in probs
    assert "neutral" in probs
    assert "surprise" in probs

    # Sum of probabilities should be approximately 1.0
    prob_sum = sum(probs.values())
    assert 0.95 <= prob_sum <= 1.05

    # Telemetry and model info
    assert "timing" in pred_data
    assert pred_data["timing"]["total_ms"] > 0
    assert "model_info" in pred_data
    assert pred_data["model_info"]["architecture"] == "resnet18"


@pytest.mark.asyncio
async def test_invalid_image_upload(client: AsyncClient):
    """Test that invalid file payloads return appropriate HTTP error responses."""
    fake_bytes = b"NOT_A_VALID_IMAGE_FILE_CONTENT"
    files = {"image": ("corrupt.jpg", fake_bytes, "image/jpeg")}

    res = await client.post("/api/v1/predictions", files=files)
    # Should return 400 Bad Request or 422 Unprocessable Entity
    assert res.status_code in (400, 422)


@pytest.mark.asyncio
async def test_session_predictions_history(client: AsyncClient):
    """Test querying prediction history associated with a session."""
    # 1. Create session
    sess_res = await client.post("/api/v1/sessions", json={"name": "History Test"})
    session_id = sess_res.json()["id"]

    # 2. Add prediction to session
    test_img_path = ROOT_DIR / "reports" / "random_and_internet_eval" / "internet_4_neutral_orig.jpg"
    with open(test_img_path, "rb") as f:
        img_bytes = f.read()

    files = {"image": ("neutral_face.jpg", img_bytes, "image/jpeg")}
    data = {"session_id": session_id}
    pred_res = await client.post("/api/v1/predictions", files=files, data=data)
    assert pred_res.status_code == 200

    # 3. Retrieve history via GET /api/v1/sessions/{id}/predictions
    hist_res = await client.get(f"/api/v1/sessions/{session_id}/predictions")
    assert hist_res.status_code == 200
    history = hist_res.json()
    assert len(history) >= 1
    assert history[0]["faces_detected"] >= 1


def test_websocket_realtime_stream():
    """Test WebSocket connection, server handshake, and status payload."""
    from starlette.testclient import TestClient

    with TestClient(app) as test_client:
        with test_client.websocket_connect("/api/v1/realtime/emotion") as ws:
            handshake = ws.receive_json()
            assert handshake["type"] == "status"
            assert handshake["status"] == "connected"
            assert "session_id" in handshake
            assert handshake["session_id"] is not None
            assert "details" in handshake
            assert "model_version" in handshake["details"]
            assert "target_fps" in handshake["details"]

