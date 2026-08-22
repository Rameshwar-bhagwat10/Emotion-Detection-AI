# Real-Time WebSocket Protocol Reference

**Endpoint**: `WS /api/v1/realtime/emotion`

---

## 1. Connection Lifecycle

```
Client                                  Server
  │                                       │
  │─── HTTP GET /api/v1/realtime/emotion ─▶│ (WebSocket Upgrade)
  │◀── HTTP 101 Switching Protocols ──────│
  │                                       │
  │◀── RealtimeStatusMessage (connected) ─│
  │                                       │
  │─── Binary Frame (JPEG) / JSON Frame ─▶│
  │◀── RealtimePredictionResponse ────────│
  │                                       │
  │─── Ping / Config Message ────────────▶│
  │◀── Pong / Status Message ─────────────│
  │                                       │
  │─── WebSocket Close (1000) ───────────▶│
  │◀── Clean Teardown & Session End ──────│
```

---

## 2. Client-to-Server Messages

### 2.1 Binary Image Frame (Preferred)
- **Transport**: Binary WebSocket frame (`ArrayBuffer`).
- **Format**: Standard JPEG/PNG image binary buffer.
- **Maximum Size**: 5 MB (`REALTIME_MAX_FRAME_SIZE_MB`).

### 2.2 JSON Image Frame (Fallback)
```json
{
  "type": "frame",
  "frame_id": 104,
  "timestamp": 1723981234.56,
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
}
```

### 2.3 Ping Message
```json
{
  "type": "ping",
  "timestamp": 1723981234.56
}
```

### 2.4 Stream Configuration Update
```json
{
  "type": "config",
  "target_fps": 15,
  "smoothing_enabled": true,
  "confidence_threshold": 0.45
}
```

---

## 3. Server-to-Client Messages

### 3.1 Real-Time Prediction Response (`type: "prediction"`)
```json
{
  "type": "prediction",
  "frame_id": 104,
  "client_timestamp": 1723981234.56,
  "session_id": "6a67fef2-4f18-4a57-b087-9bb3a62eaec1",
  "faces_detected": 1,
  "faces": [
    {
      "face_id": 1,
      "bbox": {
        "x": 140,
        "y": 80,
        "width": 180,
        "height": 180
      },
      "detection_confidence": 0.98,
      "emotion": "happy",
      "confidence": 0.94,
      "is_uncertain": false,
      "probabilities": {
        "angry": 0.01,
        "disgust": 0.00,
        "fear": 0.01,
        "happy": 0.94,
        "sad": 0.01,
        "surprise": 0.02,
        "neutral": 0.01
      },
      "raw_emotion": "happy",
      "raw_confidence": 0.96
    }
  ],
  "metrics": {
    "inference_time_ms": 4.25,
    "total_processing_time_ms": 7.82,
    "fps": 10.2,
    "dropped_frames": 0,
    "server_timestamp": 1723981234.57
  }
}
```

### 3.2 Status Message (`type: "status"`)
```json
{
  "type": "status",
  "status": "connected",
  "session_id": "6a67fef2-4f18-4a57-b087-9bb3a62eaec1",
  "message": "WebSocket connection established. Inference pipeline active.",
  "details": {
    "model_version": "champion-pruning-30",
    "target_fps": 10,
    "smoothing_alpha": 0.6,
    "max_frame_size_mb": 5
  }
}
```

### 3.3 Error Message (`type: "error"`)
```json
{
  "type": "error",
  "code": "INVALID_FRAME",
  "message": "Frame size exceeds maximum 5MB.",
  "frame_id": 104
}
```

Standard Error Codes:
- `INVALID_FRAME`: Corrupted, unreadable, or oversized frame data.
- `INVALID_MESSAGE`: Malformed JSON or unknown message type.
- `MODEL_NOT_READY`: Phase 09 inference engine is uninitialized.
- `INFERENCE_ERROR`: Internal model inference error.
- `SERVER_BUSY`: Server overloaded (excessive backlog).
