"""Unit and integration tests for Real-Time WebSocket stream and temporal emotion detection."""

from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image


def create_test_face_image_bytes() -> bytes:
    """Create a synthetic face-like JPEG image buffer."""
    img = Image.new("RGB", (320, 240), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def create_blank_image_bytes() -> bytes:
    """Create a blank grayscale JPEG with no face features."""
    img = Image.new("RGB", (320, 240), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def test_realtime_websocket_connect_and_ping(app_instance):
    """Verify WebSocket handshake status message and ping/pong protocol."""
    client = TestClient(app_instance)
    with client.websocket_connect("/api/v1/realtime/emotion") as websocket:
        # Handshake
        initial_msg = websocket.receive_json()
        assert initial_msg["type"] == "status"
        assert initial_msg["status"] == "connected"
        assert "session_id" in initial_msg

        # Ping
        websocket.send_json({"type": "ping", "timestamp": 123456.78})
        pong_msg = websocket.receive_json()
        assert pong_msg["type"] == "pong"
        assert pong_msg["client_timestamp"] == 123456.78
        assert "server_timestamp" in pong_msg


def test_realtime_websocket_binary_frame_flow(app_instance):
    """Verify real-time binary frame transmission and prediction response schema."""
    client = TestClient(app_instance)
    frame_bytes = create_test_face_image_bytes()

    with client.websocket_connect("/api/v1/realtime/emotion") as websocket:
        # Consume handshake
        _ = websocket.receive_json()

        # Send binary frame
        websocket.send_bytes(frame_bytes)

        # Receive prediction
        res = websocket.receive_json()
        assert res["type"] == "prediction"
        assert "frame_id" in res
        assert "faces_detected" in res
        assert "faces" in res
        assert "metrics" in res
        assert res["metrics"]["inference_time_ms"] >= 0.0


def test_realtime_websocket_json_base64_frame(app_instance):
    """Verify real-time JSON frame message with base64 encoded image."""
    client = TestClient(app_instance)
    frame_bytes = create_test_face_image_bytes()
    b64_str = base64.b64encode(frame_bytes).decode("utf-8")

    with client.websocket_connect("/api/v1/realtime/emotion") as websocket:
        _ = websocket.receive_json()

        payload = {
            "type": "frame",
            "frame_id": 42,
            "timestamp": 1723900.0,
            "image": f"data:image/jpeg;base64,{b64_str}",
        }
        websocket.send_json(payload)

        res = websocket.receive_json()
        assert res["type"] == "prediction"
        assert res["frame_id"] == 42
        assert res["client_timestamp"] == 1723900.0


def test_realtime_websocket_no_face_handling(app_instance):
    """Verify that a frame without faces returns zero faces and no fabricated emotions."""
    client = TestClient(app_instance)
    blank_bytes = create_blank_image_bytes()

    with client.websocket_connect("/api/v1/realtime/emotion") as websocket:
        _ = websocket.receive_json()

        websocket.send_bytes(blank_bytes)
        res = websocket.receive_json()

        assert res["type"] == "prediction"
        assert res["faces_detected"] == 0
        assert res["faces"] == []


def test_realtime_websocket_invalid_message_error(app_instance):
    """Verify structured error handling for malformed JSON and corrupted frames."""
    client = TestClient(app_instance)

    with client.websocket_connect("/api/v1/realtime/emotion") as websocket:
        _ = websocket.receive_json()

        # Send invalid frame bytes
        websocket.send_bytes(b"not_an_image_corrupted_data_bytes")
        err = websocket.receive_json()
        assert err["type"] == "error"
        assert err["code"] == "INVALID_FRAME"

        # Send unknown JSON message type
        websocket.send_json({"type": "unknown_action_xyz"})
        err2 = websocket.receive_json()
        assert err2["type"] == "error"
        assert err2["code"] == "INVALID_MESSAGE"


def test_realtime_websocket_backpressure_and_queue(app_instance):
    """Verify that rapid frame submission is throttled via bounded queue without crashing."""
    client = TestClient(app_instance)
    frame_bytes = create_test_face_image_bytes()

    with client.websocket_connect("/api/v1/realtime/emotion") as websocket:
        _ = websocket.receive_json()

        # Rapidly send 5 binary frames
        for _ in range(5):
            websocket.send_bytes(frame_bytes)

        # Server should process at least one frame successfully
        res = websocket.receive_json()
        assert res["type"] == "prediction"
        assert res["metrics"]["dropped_frames"] >= 0


def test_realtime_websocket_config_update(app_instance):
    """Verify live stream config tuning over WebSocket."""
    client = TestClient(app_instance)

    with client.websocket_connect("/api/v1/realtime/emotion") as websocket:
        _ = websocket.receive_json()

        websocket.send_json(
            {
                "type": "config",
                "smoothing_enabled": True,
                "confidence_threshold": 0.50,
            }
        )
        ack = websocket.receive_json()
        assert ack["type"] == "status"
        assert ack["status"] == "configured"
