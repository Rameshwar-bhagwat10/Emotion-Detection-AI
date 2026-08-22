# Real-Time Video & WebSocket Troubleshooting Guide

This guide documents common issues encountered during real-time webcam emotion detection streaming and their solutions.

---

## 1. Camera Access Issues

### `NotAllowedError: Permission denied`
- **Cause**: Browser camera permissions were denied or revoked by the user.
- **Resolution**:
  1. Click the lock or camera icon in the browser URL bar.
  2. Change permissions from **Block** to **Allow**.
  3. Refresh the page or click **Retry Camera Access**.

### `NotFoundError: DevicesNotFoundError`
- **Cause**: No webcam hardware was found on the system.
- **Resolution**: Ensure your physical webcam or virtual camera is connected and recognized by the operating system.

---

## 2. WebSocket Connection Issues

### `WebSocket connection failed: 1013 Try Again Later`
- **Cause**: Phase 09 Inference Engine is still loading weights or performing initial warmup passes.
- **Resolution**: Check server health at `GET /api/v1/health/ready` until `status: "ready"`.

### WebSocket Drops / Network Loss
- **Behavior**: The `useRealtimeEmotion` hook automatically initiates exponential backoff reconnects while the camera is active.
- **Resolution**: Verify the FastAPI server is running on port 8000 and CORS / WebSocket origins are allowed in `.env`.

---

## 3. High Latency or Jitter

### Bounding Boxes Lagging Behind Face
- **Cause**: Ingestion FPS higher than client rendering or network bandwidth limit.
- **Resolution**:
  1. Lower target FPS in the controls (e.g. from 20 FPS to 10 FPS).
  2. Reduce JPEG quality from 0.8 to 0.7.
  3. Ensure server is using hardware acceleration if available.
