# Backend Architecture & Service Layer

This document describes the architectural principles, component responsibilities, lifecycle management, and integration patterns of the **Emotion Detection AI Backend (Phase 10)**.

---

## 1. System Overview & Layered Architecture

The backend is built with **FastAPI**, **SQLAlchemy 2.x**, and **PostgreSQL**, strictly encapsulating the **Phase 09 ML Inference Engine** without recreating or leaking machine learning logic into HTTP routers or database repositories.

```
                    ┌─────────────────────────┐
                    │      HTTP Client        │
                    │ (Next.js / Curl / Test) │
                    └────────────┬────────────┘
                                 │
                         HTTP / REST API
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      FastAPI App        │
                    │  - Request ID (UUID4)   │
                    │  - CORS Middleware      │
                    │  - Exception Handlers   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   API Routers (/api/v1) │
                    │  - /health & /ready     │
                    │  - /predictions         │
                    │  - /sessions            │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      Service Layer      │
                    │  - PredictionService    │
                    │  - SessionService       │
                    └──────┬────────────┬─────┘
                           │            │
             Inference Call│            │DB Transaction
                           ▼            ▼
           ┌──────────────────────┐  ┌───────────────────────┐
           │ Phase 09 ML Engine   │  │   Repository Layer    │
           │ - YuNet Detection    │  │ - PredictionRepository│
           │ - 15% Face Padding   │  │ - SessionRepository   │
           │ - ResNet-18 Champion │  │ - UserRepository      │
           │ - Softmax & Gating   │  └──────────┬────────────┘
           └──────────────────────┘             │
                                                ▼
                                     ┌───────────────────────┐
                                     │  PostgreSQL Database  │
                                     │ (SQLAlchemy 2.x Async)│
                                     └───────────────────────┘
```

---

## 2. Component Responsibilities

### A. API Layer (`apps/api/app/api/`)
- **HTTP Routing**: Dispatches requests to versioned route handlers (`/api/v1`).
- **Validation**: Enforces Pydantic request/response model constraints and multipart upload headers.
- **Dependency Injection**: Injects database sessions (`get_db`), request IDs (`get_request_id`), and singleton inference engines (`get_inference_engine`).
- **Zero ML Logic**: No face cropping, model loading, tensor operations, or raw SQL queries exist in route functions.

### B. Service Layer (`apps/api/app/services/`)
- **`PredictionService`**:
  - Validates payload file size against `MAX_IMAGE_SIZE_MB` (default: 10 MB).
  - Validates image extension against `ALLOWED_IMAGE_EXTENSIONS` (`.jpg`, `.jpeg`, `.png`, `.webp`).
  - Calls Phase 09 `EmotionInferenceEngine.predict_image(image_source=image_bytes)`.
  - Coordinates database transactions to persist `PredictionRecord` and `DetectedFaceRecord` entities.
  - Formats output into `PredictionResponseSchema`.
- **`SessionService`**:
  - Manages session lifecycle (`active` $\to$ `completed` / `cancelled`).
  - Retrieves paginated predictions and session history.

### C. Repository Layer (`apps/api/app/db/repositories/`)
- **`PredictionRepository`**: Handles atomic transactional persistence of prediction events and face bounding box/probability records.
- **`SessionRepository`**: Manages session querying, creation, state transitions, and eager loading of prediction relations.
- **`UserRepository`**: Foundation for user lookup and future authentication integration.

### D. Phase 09 Inference Engine (`ml/inference/`)
- Single source of truth for computer vision and deep learning inference.
- Manages device resolution (`auto`, `cpu`, `cuda`), YuNet face detection, deterministic preprocessing ($48 \times 48$, $\mu = 0.507743, \sigma = 0.255009$), and the Phase 08 Optimized Champion (`champion-pruning-30`).

---

## 3. Application Lifecycle & Model Warm-Up

FastAPI modern `lifespan` manages startup and teardown:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Load Phase 09 Champion & Face Detector
    pipeline_config = InferencePipelineConfig()
    engine = EmotionInferenceEngine(config=pipeline_config)
    engine.warm_up(num_warmup_passes=3)
    app.state.inference_engine = engine
    app.state.is_ready = True
    
    yield
    
    # 2. Teardown: Close Redis & Dispose DB Engine
    await close_redis_client()
    await engine.dispose()
```

- **Singleton Model Lifecycle**: The model is loaded into memory **once** at startup and reused across all concurrent requests. Zero per-request reloads.
- **Warm-Up Execution**: Pre-allocates execution buffers and runs 3 non-gradient forward passes before opening HTTP traffic.

---

## 4. Error Handling & Security

- **Centralized Handlers**: Standardized error response envelope:
  ```json
  {
    "error": {
      "code": "INVALID_IMAGE",
      "message": "The uploaded file is not a valid image.",
      "request_id": "c1f7a08b-287b-4a5c-9c90-951c5188f58b",
      "details": {}
    }
  }
  ```
- **Information Masking**: Internal tracebacks, database passwords, file system paths, and weights files are never leaked to API clients.
- **Request ID Tracking**: Every request generates or propagates an `X-Request-ID` header and records request execution duration in `X-Process-Time-Ms`.
