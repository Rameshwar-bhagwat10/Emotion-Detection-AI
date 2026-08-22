# REST API Reference & Specification

This document details the HTTP endpoints exposed by the **Emotion Detection AI API (v1)**.

---

## Base URLs & OpenAPI Docs

- **Base Route**: `http://localhost:8000/api/v1`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

---

## 1. Health & Readiness Endpoints

### `GET /api/v1/health`
**Description**: Service liveness and basic subsystem health status.

**Response (HTTP 200)**:
```json
{
  "status": "ok",
  "service": "emotion-detection-api",
  "version": "1.0.0",
  "environment": "development",
  "dependencies": {
    "database": {
      "status": "healthy",
      "latency_ms": 1.25
    },
    "redis": {
      "status": "healthy",
      "host": "localhost",
      "port": 6379
    }
  }
}
```

### `GET /api/v1/health/live`
**Description**: Lightweight container liveness probe.

**Response (HTTP 200)**:
```json
{
  "status": "ok",
  "service": "emotion-detection-api"
}
```

### `GET /api/v1/health/ready`
**Description**: Deep readiness probe verifying database connectivity and Phase 09 ML inference engine availability.

**Response (HTTP 200 / 503)**:
```json
{
  "status": "ready",
  "service": "emotion-detection-api",
  "version": "1.0.0",
  "model": "ready",
  "database": "ready",
  "details": {
    "model_version": "champion-pruning-30",
    "device": "auto",
    "database": {
      "status": "healthy",
      "latency_ms": 1.10
    }
  }
}
```

---

## 2. Prediction Endpoints

### `POST /api/v1/predictions`
**Description**: Uploads an image file to detect faces, classify facial emotions, and persist analysis records.

**Request**:
- `Content-Type`: `multipart/form-data`
- `image`: File binary (`image/jpeg`, `image/png`, `image/webp`). Max size: 10 MB.
- `session_id` *(optional)*: UUID of an existing session.

**Response (HTTP 200)**:
```json
{
  "status": "SUCCESS",
  "request_id": "96c0bb14-993d-4c31-893c-cf572d4ae993",
  "prediction_id": "787889db-4927-4638-b7f7-333e6f9872e4",
  "session_id": "52f6fbf7-074d-453d-8e4d-7bc51ff10626",
  "faces_detected": 1,
  "faces": [
    {
      "face_id": 1,
      "bbox": {
        "x": 120,
        "y": 80,
        "width": 180,
        "height": 180
      },
      "detection_confidence": 0.96,
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
      }
    }
  ],
  "timing": {
    "image_loading_ms": 0.12,
    "face_detection_ms": 8.54,
    "preprocessing_ms": 0.40,
    "inference_ms": 4.31,
    "postprocessing_ms": 0.15,
    "total_ms": 13.52
  },
  "model_info": {
    "model_name": "champion-pruning-30",
    "architecture": "resnet18",
    "optimization_type": "pruning_30pct",
    "device": "cpu"
  },
  "created_at": "2026-08-18T16:20:00.000000Z"
}
```

### `GET /api/v1/predictions/{prediction_id}`
**Description**: Retrieves stored prediction record and mapped face analysis details by UUID.

**Response (HTTP 200)**:
```json
{
  "id": "787889db-4927-4638-b7f7-333e6f9872e4",
  "session_id": "52f6fbf7-074d-453d-8e4d-7bc51ff10626",
  "request_id": "96c0bb14-993d-4c31-893c-cf572d4ae993",
  "model_version": "champion-pruning-30",
  "status": "SUCCESS",
  "faces_detected": 1,
  "processing_time_ms": 13.52,
  "image_width": 640,
  "image_height": 480,
  "created_at": "2026-08-18T16:20:00.000000Z",
  "faces": [
    {
      "face_id": 1,
      "bbox": {
        "x": 120,
        "y": 80,
        "width": 180,
        "height": 180
      },
      "detection_confidence": 0.96,
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
      }
    }
  ]
}
```

---

## 3. Session Endpoints

### `POST /api/v1/sessions`
**Description**: Initializes a new analysis session.

**Request (JSON)**:
```json
{
  "name": "Live Interview Analysis",
  "user_id": null
}
```

**Response (HTTP 201)**:
```json
{
  "id": "52f6fbf7-074d-453d-8e4d-7bc51ff10626",
  "user_id": null,
  "name": "Live Interview Analysis",
  "status": "active",
  "started_at": "2026-08-18T16:15:00.000000Z",
  "ended_at": null,
  "created_at": "2026-08-18T16:15:00.000000Z"
}
```

### `GET /api/v1/sessions/{session_id}`
**Description**: Fetches session details by UUID.

### `POST /api/v1/sessions/{session_id}/end`
**Description**: Marks an active session as completed and sets the `ended_at` timestamp.

### `GET /api/v1/sessions/{session_id}/predictions`
**Description**: Retrieves all prediction records attached to a session.

---

## 4. Error Code Summary

| HTTP Code | Error Code | Description |
| :--- | :--- | :--- |
| **400** | `INVALID_IMAGE` | Corrupted, unreadable, or invalid image bytes |
| **404** | `NOT_FOUND` | Resource (prediction/session ID) does not exist |
| **413** | `PAYLOAD_TOO_LARGE` | Uploaded image exceeds `MAX_IMAGE_SIZE_MB` (10 MB) |
| **415** | `UNSUPPORTED_MEDIA_TYPE` | File extension or MIME type not in allowed list |
| **422** | `VALIDATION_ERROR` | Schema validation error in request query or payload |
| **500** | `DATABASE_ERROR` | Database transaction or persistence failure |
| **500** | `INFERENCE_ERROR` | Internal machine learning pipeline failure |
| **503** | `MODEL_NOT_READY` | Inference engine is initializing or unavailable |
