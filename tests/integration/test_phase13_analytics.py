"""Integration test suite for Phase 13 Analytics, Session History & Explainability APIs.

Validates:
1. GET /api/v1/analytics/overview (Global statistics, 7-class distribution, confidence bins, trends)
2. GET /api/v1/analytics/sessions (Enriched session history list with dominant emotion and avg confidence)
3. GET /api/v1/analytics/sessions/{id} (Session-isolated metrics and expression distribution)
4. GET /api/v1/analytics/sessions/{id}/timeline (Time-bucketed aggregation for charts)
5. GET /api/v1/analytics/sessions/{id}/confidence (Confidence histogram and uncertainty metrics)
6. GET /api/v1/history/predictions (Filterable and paginated prediction log)
7. GET /api/v1/history/predictions/{id} (Detailed prediction inspection with top-k probabilities)
8. Mathematical correctness (sum(counts) == total, sum(pct) ~= 100%)
9. Error handling (non-existent session returns HTTP 404)
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
import sys
import uuid

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

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


def create_synthetic_face_image() -> bytes:
    """Create a synthetic RGB image with skin-toned oval approximating a face."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)
    for y in range(300):
        for x in range(300):
            if ((x - 150) / 70) ** 2 + ((y - 150) / 90) ** 2 <= 1.0:
                img[y, x] = (160, 190, 230)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_global_analytics_overview(client: AsyncClient):
    """Test GET /api/v1/analytics/overview returns valid structure and non-negative counts."""
    response = await client.get("/api/v1/analytics/overview")
    assert response.status_code == 200
    data = response.json()

    assert "total_sessions" in data
    assert "total_predictions" in data
    assert "total_faces" in data
    assert "total_duration_seconds" in data
    assert "expression_distribution" in data
    assert "confidence_analytics" in data

    dist = data["expression_distribution"]
    assert "items" in dist
    assert len(dist["items"]) >= 7  # All supported emotion classes
    expected_emotions = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}
    returned_emotions = {item["emotion"] for item in dist["items"]}
    assert expected_emotions.issubset(returned_emotions)

    # Confidence distribution histogram
    conf = data["confidence_analytics"]
    assert "distribution" in conf
    assert len(conf["distribution"]) == 5  # 5 bins: 0-20, 20-40, 40-60, 60-80, 80-100


@pytest.mark.asyncio
async def test_session_lifecycle_and_analytics(client: AsyncClient):
    """Test full cycle: create session -> seed predictions -> get session analytics & timeline."""
    # 1. Create a session
    sess_res = await client.post("/api/v1/sessions", json={"name": "Phase 13 Analytics Test Session"})
    assert sess_res.status_code == 201
    sess_data = sess_res.json()
    session_id = sess_data["id"]

    # 2. Seed a prediction under this session
    img_bytes = create_synthetic_face_image()
    files = {"image": ("test_face.jpg", img_bytes, "image/jpeg")}
    data = {"session_id": session_id}
    pred_res = await client.post("/api/v1/predictions", files=files, data=data)
    assert pred_res.status_code == 200

    # 3. End session
    end_res = await client.post(f"/api/v1/sessions/{session_id}/end")
    assert end_res.status_code == 200

    # 4. Fetch session-level analytics
    analytics_res = await client.get(f"/api/v1/analytics/sessions/{session_id}")
    assert analytics_res.status_code == 200
    analytics = analytics_res.json()

    assert analytics["session_id"] == session_id
    assert analytics["status"] == "completed"
    assert analytics["total_predictions"] >= 1
    assert "expression_distribution" in analytics
    assert "confidence_analytics" in analytics

    # Mathematical consistency: sum of counts equals total faces
    total_faces = analytics["total_faces"]
    items = analytics["expression_distribution"]["items"]
    sum_counts = sum(item["count"] for item in items)
    assert sum_counts == total_faces

    if total_faces > 0:
        sum_pct = sum(item["percentage"] for item in items)
        assert 99.0 <= sum_pct <= 101.0  # subject to 1-decimal rounding

    # 5. Fetch session timeline
    timeline_res = await client.get(f"/api/v1/analytics/sessions/{session_id}/timeline")
    assert timeline_res.status_code == 200
    timeline = timeline_res.json()
    assert timeline["session_id"] == session_id
    assert "buckets" in timeline
    assert isinstance(timeline["buckets"], list)

    # 6. Fetch session confidence
    conf_res = await client.get(f"/api/v1/analytics/sessions/{session_id}/confidence")
    assert conf_res.status_code == 200
    conf_data = conf_res.json()
    assert "distribution" in conf_data
    assert len(conf_data["distribution"]) == 5


@pytest.mark.asyncio
async def test_session_analytics_list(client: AsyncClient):
    """Test GET /api/v1/analytics/sessions returns enriched session cards."""
    response = await client.get("/api/v1/analytics/sessions?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert "sessions" in data
    assert "total" in data
    assert isinstance(data["sessions"], list)

    if len(data["sessions"]) > 0:
        first = data["sessions"][0]
        assert "id" in first
        assert "prediction_count" in first
        assert "duration_seconds" in first
        assert "average_confidence" in first


@pytest.mark.asyncio
async def test_prediction_history_filters(client: AsyncClient):
    """Test GET /api/v1/history/predictions with pagination and sorting."""
    # List predictions
    response = await client.get("/api/v1/history/predictions?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "limit" in data
    assert "offset" in data
    assert data["limit"] == 10

    if data["total"] > 0:
        first_item = data["items"][0]
        assert "prediction_id" in first_item
        assert "emotion" in first_item
        assert "confidence" in first_item
        assert "bbox" in first_item
        assert "probabilities" in first_item
        assert "model_version" in first_item

        pred_id = first_item["prediction_id"]

        # Inspect prediction detail
        inspect_res = await client.get(f"/api/v1/history/predictions/{pred_id}")
        assert inspect_res.status_code == 200
        inspect_data = inspect_res.json()
        assert inspect_data["id"] == pred_id
        assert len(inspect_data["faces"]) >= 1


@pytest.mark.asyncio
async def test_nonexistent_session_404(client: AsyncClient):
    """Test querying analytics for a random non-existent UUID returns HTTP 404."""
    random_uuid = uuid.uuid4()
    response = await client.get(f"/api/v1/analytics/sessions/{random_uuid}")
    assert response.status_code == 404
    data = response.json()
    assert data["error"]["code"] == "NOT_FOUND"
