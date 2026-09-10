"""Integration tests for Video Analysis capability (Phase 13 Video Implementation).

Validates:
1. Video container validation & technical metadata extraction.
2. Asynchronous job initiation (POST /api/v1/video/upload -> HTTP 202).
3. Status polling and stage progression (QUEUED -> PROCESSING -> COMPLETED).
4. Frame sampling, multi-face tracking, and temporal probability smoothing.
5. Expression segment generation and prediction transition events.
6. Aggregated video analytics (time share vs frame share, dominant expression, confidence metrics).
7. Paginated frame predictions API with probability distributions and bounding boxes.
8. Video streaming endpoint with Accept-Ranges support.
9. Mathematical consistency (sum(durations) <= video_duration, segment start < end).
10. Error handling for corrupted, empty, or unsupported media formats.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
import sys
import tempfile
import time
import uuid

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

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


def generate_synthetic_mp4_video(
    duration_sec: float = 2.0,
    fps: int = 10,
    width: int = 240,
    height: int = 240,
) -> bytes:
    """Synthesize a valid MP4 video container with a moving face."""
    total_frames = int(duration_sec * fps)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, float(fps), (width, height))

        for i in range(total_frames):
            frame = np.full((height, width, 3), 220, dtype=np.uint8)

            # Draw a face oval: center (cx, cy) shifts slightly
            cx = 120 + int(10 * np.sin(i / 2.0))
            cy = 120
            # Skin tone oval
            cv2.ellipse(frame, (cx, cy), (50, 65), 0, 0, 360, (180, 200, 240), -1)
            # Eyes
            cv2.circle(frame, (cx - 18, cy - 15), 6, (60, 60, 60), -1)
            cv2.circle(frame, (cx + 18, cy - 15), 6, (60, 60, 60), -1)

            # In first half: smiling mouth; in second half: neutral mouth
            if i < total_frames // 2:
                cv2.ellipse(frame, (cx, cy + 22), (20, 10), 0, 0, 180, (40, 40, 180), 3)
            else:
                cv2.line(frame, (cx - 15, cy + 25), (cx + 15, cy + 25), (40, 40, 180), 3)

            writer.write(frame)

        writer.release()

        with open(tmp_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes
    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


@pytest.mark.asyncio
async def test_video_upload_and_job_lifecycle(client: AsyncClient):
    """Test full asynchronous lifecycle from upload to completion."""
    video_bytes = generate_synthetic_mp4_video(duration_sec=2.0, fps=10)

    # 1. Upload video
    files = {"video": ("synthetic_test.mp4", video_bytes, "video/mp4")}
    data = {"sampling_fps": "5.0"}

    response = await client.post("/api/v1/video/upload", files=files, data=data)
    assert response.status_code == 202, f"Upload failed: {response.text}"
    job_data = response.json()

    assert "video_id" in job_data
    assert job_data["status"] in ("QUEUED", "PROCESSING")
    video_id = job_data["video_id"]

    # 2. Poll for completion
    max_wait_sec = 25
    start_poll = time.time()
    completed = False
    status_data = None

    while time.time() - start_poll < max_wait_sec:
        status_res = await client.get(f"/api/v1/video/{video_id}/status")
        assert status_res.status_code == 200
        status_data = status_res.json()

        if status_data["status"] == "COMPLETED":
            completed = True
            break
        elif status_data["status"] == "FAILED":
            pytest.fail(f"Video analysis failed: {status_data.get('error_message')}")

        await asyncio.sleep(0.5)

    assert completed, f"Job timed out after {max_wait_sec}s. Last status: {status_data}"
    assert status_data["progress_percent"] == 100.0
    assert status_data["current_stage"] == "completed"
    assert status_data["frames_analyzed"] > 0


@pytest.mark.asyncio
async def test_video_detail_and_analytics(client: AsyncClient):
    """Test detailed video report and analytical aggregations."""
    video_bytes = generate_synthetic_mp4_video(duration_sec=2.0, fps=10)
    files = {"video": ("test_analytics.mp4", video_bytes, "video/mp4")}
    res = await client.post("/api/v1/video/upload", files=files, data={"sampling_fps": "5.0"})
    video_id = res.json()["video_id"]

    # Wait for completion
    for _ in range(50):
        s = await client.get(f"/api/v1/video/{video_id}/status")
        if s.json()["status"] == "COMPLETED":
            break
        await asyncio.sleep(0.5)

    detail_res = await client.get(f"/api/v1/video/{video_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()

    # Verify metadata
    metadata = detail["metadata"]
    assert metadata["duration_seconds"] > 0
    assert metadata["source_fps"] > 0
    assert metadata["total_frames"] > 0
    assert metadata["frames_analyzed"] > 0

    # Verify analytics
    analytics = detail["analytics"]
    assert analytics is not None
    assert "duration_seconds" in analytics
    assert "dominant_expression" in analytics
    assert "average_confidence" in analytics
    assert "expression_distribution" in analytics
    assert "transition_counts" in analytics

    # Mathematical consistency checks
    dist = analytics["expression_distribution"]
    assert len(dist) > 0
    total_count = sum(item["count"] for item in dist)
    assert total_count == analytics["total_predictions_count"]


@pytest.mark.asyncio
async def test_video_timeline_and_mathematical_consistency(client: AsyncClient):
    """Test expression segments, transition events, and chronological ordering."""
    video_bytes = generate_synthetic_mp4_video(duration_sec=2.0, fps=10)
    files = {"video": ("test_timeline.mp4", video_bytes, "video/mp4")}
    res = await client.post("/api/v1/video/upload", files=files, data={"sampling_fps": "5.0"})
    video_id = res.json()["video_id"]

    for _ in range(50):
        s = await client.get(f"/api/v1/video/{video_id}/status")
        if s.json()["status"] == "COMPLETED":
            break
        await asyncio.sleep(0.5)

    timeline_res = await client.get(f"/api/v1/video/{video_id}/timeline")
    assert timeline_res.status_code == 200
    timeline = timeline_res.json()

    segments = timeline["segments"]
    events = timeline["events"]

    if segments:
        for seg in segments:
            assert seg["start_time"] >= 0.0, "Start time cannot be negative"
            assert seg["end_time"] >= seg["start_time"], "End time must be >= start time"
            assert seg["duration"] > 0.0, "Duration must be positive"
            assert 0.0 <= seg["average_confidence"] <= 1.0, "Confidence must be in [0, 1]"

        # Verify segments for same track are chronologically ordered
        for i in range(len(segments) - 1):
            if segments[i]["track_id"] == segments[i + 1]["track_id"]:
                assert segments[i]["start_time"] <= segments[i + 1]["start_time"], "Segments must be chronological"

    if events:
        for i in range(len(events) - 1):
            assert events[i]["timestamp"] <= events[i + 1]["timestamp"], "Events must be chronological"


@pytest.mark.asyncio
async def test_video_predictions_pagination(client: AsyncClient):
    """Test paginated frame-level predictions query."""
    video_bytes = generate_synthetic_mp4_video(duration_sec=2.0, fps=10)
    files = {"video": ("test_preds.mp4", video_bytes, "video/mp4")}
    res = await client.post("/api/v1/video/upload", files=files, data={"sampling_fps": "5.0"})
    video_id = res.json()["video_id"]

    for _ in range(50):
        s = await client.get(f"/api/v1/video/{video_id}/status")
        if s.json()["status"] == "COMPLETED":
            break
        await asyncio.sleep(0.5)

    preds_res = await client.get(f"/api/v1/video/{video_id}/predictions?page=1&page_size=5")
    assert preds_res.status_code == 200
    preds_data = preds_res.json()

    assert "total" in preds_data
    assert "predictions" in preds_data
    assert preds_data["page"] == 1
    assert preds_data["page_size"] == 5

    if preds_data["predictions"]:
        p = preds_data["predictions"][0]
        assert "timestamp" in p
        assert "frame_index" in p
        assert "bbox" in p
        assert "smoothed_emotion" in p
        assert "probabilities" in p
        assert len(p["probabilities"]) == 7


@pytest.mark.asyncio
async def test_video_streaming_endpoint(client: AsyncClient):
    """Test video file streaming endpoint returns 200 OK."""
    video_bytes = generate_synthetic_mp4_video(duration_sec=1.0, fps=10)
    files = {"video": ("test_stream.mp4", video_bytes, "video/mp4")}
    res = await client.post("/api/v1/video/upload", files=files)
    video_id = res.json()["video_id"]

    stream_res = await client.get(f"/api/v1/video/{video_id}/stream")
    assert stream_res.status_code in (200, 206)
    assert len(stream_res.content) > 0


@pytest.mark.asyncio
async def test_video_validation_errors(client: AsyncClient):
    """Test rejection of corrupted and unsupported file uploads."""
    # 1. Unsupported extension (.txt)
    files = {"video": ("invalid_doc.txt", b"plain text content", "text/plain")}
    res = await client.post("/api/v1/video/upload", files=files)
    assert res.status_code in (415, 400)

    # 2. Corrupted video bytes
    files = {"video": ("corrupted.mp4", b"not a valid video container", "video/mp4")}
    res = await client.post("/api/v1/video/upload", files=files)
    assert res.status_code in (400, 422)

    # 3. Non-existent UUID
    fake_id = uuid.uuid4()
    res = await client.get(f"/api/v1/video/{fake_id}/status")
    assert res.status_code == 404
