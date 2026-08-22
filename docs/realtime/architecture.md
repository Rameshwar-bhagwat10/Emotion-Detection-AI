# Real-Time Video & Webcam Emotion Detection Architecture

This document specifies the end-to-end architecture for **Phase 11 — Real-Time Webcam & Video Emotion Detection** of the AI-Based Facial Expression Emotion Detection & Analytics System.

---

## 1. System Topology

```mermaid
flowchart TD
    subgraph Browser ["Client (Browser / React / Next.js)"]
        Cam[Webcam API: getUserMedia]
        Video[HTMLVideoElement]
        OffCanvas[Offscreen Canvas: Sampling & Resize]
        Encoder[JPEG Blob Encoder]
        WS_Client[useRealtimeEmotion WebSocket Client]
        Overlay[Canvas Face Overlay: Bounding Boxes & Badges]
        HUD[Telemetry & Emotion Distribution HUD]

        Cam --> Video
        Video --> OffCanvas
        OffCanvas --> Encoder
        Encoder --> WS_Client
        WS_Client --> Overlay
        WS_Client --> HUD
    end

    subgraph Backend ["FastAPI Backend (Phase 10 & 11)"]
        WS_Endpoint["WS /api/v1/realtime/emotion"]
        Queue["Bounded Frame Queue (Latest-Frame-Wins)"]
        Worker["Async Frame Processing Worker"]
        RT_Service[RealTimeService]
        Frame_Proc[FrameProcessor]
        Tracker[FaceTracker: IoU & Centroid]
        Smoother[TemporalSmoother: EMA]
        Pred_Service[PredictionService]

        WS_Endpoint --> Queue
        Queue --> Worker
        Worker --> RT_Service
        RT_Service --> Frame_Proc
        RT_Service --> Pred_Service
        RT_Service --> Tracker
        RT_Service --> Smoother
    end

    subgraph ML_Engine ["Phase 09 ML Inference Engine (Single Source of Truth)"]
        Engine[EmotionInferenceEngine Singleton]
        Detector[Face Detector: YuNet / Haar]
        Champion[Champion Model: champion-pruning-30 ResNet-18]

        Pred_Service --> Engine
        Engine --> Detector
        Engine --> Champion
    end

    subgraph DB ["Database & Persistence (Supabase / PostgreSQL)"]
        SessionRepo[SessionRepository]
        PredRepo[PredictionRepository]

        RT_Service -.->|Periodic Sampling (1/s)| PredRepo
        RT_Service -.->|Session Lifecycle| SessionRepo
    end

    WS_Client <-->|Binary / JSON WebSocket Stream| WS_Endpoint
```

---

## 2. Key Components

### 2.1 Browser Frame Capture & Transport (`useRealtimeEmotion`)
- Uses standard browser `navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, frameRate: 30 } })`.
- Frame sampling loop captures video snapshots at a configurable target processing rate (default `10 FPS`).
- In-memory JPEG encoding (`quality = 0.8`) minimizes payload size without perceptible loss in facial emotion clarity.
- Transmits raw binary frames via WebSocket `arraybuffer` to eliminate Base64 encoding overhead (reducing payload size by ~33%).

### 2.2 WebSocket Endpoint (`/api/v1/realtime/emotion`)
- High-performance asynchronous endpoint supporting concurrent client connections.
- Handshake initializes/associates a database session and returns stream parameters.
- Protected by bounded queue (`maxsize = 2`) with `latest-frame-wins` drop policy: prevents server buffer bloat and guarantees ultra-low latency.
- Stale-frame expiration: drops frames queued longer than `REALTIME_STALE_FRAME_MS` (350 ms).

### 2.3 RealTimeService (`RealTimeService`)
- Direct pipeline coordinator between WebSocket transport, Phase 10 `PredictionService`, and Phase 09 `EmotionInferenceEngine`.
- **Zero duplicate ML code**: delegates face detection and tensor classification to Phase 09.
- Performs multi-face tracking and exponential moving average (EMA) temporal smoothing.

### 2.4 Face Tracking & Temporal Smoothing
- **`FaceTracker`**: Spatial IoU and centroid distance matching assigns stable session-local integer IDs (`1`, `2`, ...) to faces across frames. Does **not** perform facial recognition or biometric identification.
- **`TemporalSmoother`**: Applies Exponential Moving Average ($P_{\text{smooth}, t} = \alpha P_{\text{raw}, t} + (1 - \alpha) P_{\text{smooth}, t-1}$ with $\alpha = 0.6$) across 7 emotion probabilities to eliminate frame-to-frame classification jitter.

### 2.5 Controlled Persistence Policy
- To prevent database exhaustion at 30 FPS, real-time predictions are sampled and persisted at configurable intervals (`REALTIME_PERSISTENCE_INTERVAL_SEC = 1.0`).
- Raw video frames and images are **never** persisted to disk or database.
