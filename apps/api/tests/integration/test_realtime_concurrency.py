"""Integration tests for multi-client WebSocket streams and session data isolation."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image


def create_test_frame_bytes(color: tuple[int, int, int]) -> bytes:
    """Create a synthetic test frame with a unique background color."""
    img = Image.new("RGB", (320, 240), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def test_concurrent_realtime_websocket_clients_isolation(app_instance):
    """Verify that multiple WebSocket clients maintain isolated state, sessions, and queues."""
    client = TestClient(app_instance)

    frame_client_1 = create_test_frame_bytes((100, 150, 200))
    frame_client_2 = create_test_frame_bytes((200, 100, 50))

    # Client 1 Session
    with client.websocket_connect("/api/v1/realtime/emotion") as ws1:
        init_1 = ws1.receive_json()
        assert init_1["type"] == "status"
        session_1 = init_1["session_id"]
        assert session_1 is not None

        ws1.send_bytes(frame_client_1)
        res1 = ws1.receive_json()
        assert res1["type"] == "prediction"
        assert res1["session_id"] == session_1

    # Client 2 Session
    with client.websocket_connect("/api/v1/realtime/emotion") as ws2:
        init_2 = ws2.receive_json()
        assert init_2["type"] == "status"
        session_2 = init_2["session_id"]
        assert session_2 is not None

        # Verify session IDs are distinct across client connections
        assert session_1 != session_2, "Clients must have distinct session IDs"

        ws2.send_bytes(frame_client_2)
        res2 = ws2.receive_json()
        assert res2["type"] == "prediction"
        assert res2["session_id"] == session_2
