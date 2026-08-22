"""WebSocket router for real-time video stream facial emotion detection."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.core.config import settings
from app.core.exceptions import InvalidImageError, PayloadTooLargeError
from app.core.logging import get_logger
from app.schemas.realtime import (
    RealtimeErrorMessage,
    RealtimePredictionResponse,
    RealtimeStatusMessage,
)
from app.services.prediction_service import PredictionService
from app.services.realtime_service import RealTimeService
from ml.inference.engine import EmotionInferenceEngine

logger = get_logger(__name__)

router = APIRouter(tags=["Realtime"])


async def _handle_json_frame(
    payload: dict[str, Any],
    client_frame_counter: int,
    receive_mono_time: float,
    frame_queue: asyncio.Queue[tuple[bytes, int, float | None, float]],
    realtime_service: RealTimeService,
    websocket: WebSocket,
) -> int:
    """Handle incoming JSON frame message."""
    base64_img = payload.get("image")
    raw_frame_id = payload.get("frame_id", client_frame_counter + 1)
    frame_id = int(raw_frame_id)
    new_counter = max(client_frame_counter, frame_id)
    client_ts_raw = payload.get("timestamp")
    client_ts: float | None = float(client_ts_raw) if client_ts_raw is not None else None

    if not base64_img or not isinstance(base64_img, str):
        err = RealtimeErrorMessage(
            code="INVALID_FRAME",
            message="Missing 'image' field in frame message.",
            frame_id=frame_id,
        )
        await websocket.send_text(err.model_dump_json())
        return int(new_counter)

    try:
        raw_bytes, _, _, _ = realtime_service.frame_processor.decode_base64_frame(base64_img)
    except Exception as exc:
        err = RealtimeErrorMessage(
            code="INVALID_FRAME",
            message=f"Failed to decode base64 frame: {exc}",
            frame_id=frame_id,
        )
        await websocket.send_text(err.model_dump_json())
        return int(new_counter)

    if frame_queue.full():
        try:
            _ = frame_queue.get_nowait()
            frame_queue.task_done()
            realtime_service.record_dropped_frame()
        except asyncio.QueueEmpty:
            pass

    await frame_queue.put((raw_bytes, frame_id, client_ts, receive_mono_time))
    return int(new_counter)


async def _handle_config_message(
    payload: dict[str, Any],
    active_session_id: uuid.UUID | None,
    realtime_service: RealTimeService,
    websocket: WebSocket,
) -> None:
    """Handle live stream configuration update message."""
    if "smoothing_enabled" in payload:
        enabled = payload["smoothing_enabled"]
        realtime_service.temporal_smoother.alpha = 0.6 if enabled else 1.0
    if "confidence_threshold" in payload:
        realtime_service.temporal_smoother.confidence_threshold = float(
            payload["confidence_threshold"]
        )
    ack = RealtimeStatusMessage(
        status="configured",
        session_id=str(active_session_id) if active_session_id else None,
        message="Stream configuration updated successfully.",
    )
    await websocket.send_text(ack.model_dump_json())


async def _run_frame_worker(
    stop_event: asyncio.Event,
    frame_queue: asyncio.Queue[tuple[bytes, int, float | None, float]],
    realtime_service: RealTimeService,
    db: AsyncSession,
    active_session_id: uuid.UUID | None,
    websocket: WebSocket,
) -> None:
    """Worker task that processes queued frames and sends predictions."""
    while not stop_event.is_set():
        try:
            try:
                raw_bytes, f_id, client_ts, enqueued_time = await asyncio.wait_for(
                    frame_queue.get(), timeout=0.1
                )
            except TimeoutError:
                continue

            frame_age_ms = (time.perf_counter() - enqueued_time) * 1000.0
            if frame_age_ms > settings.REALTIME_STALE_FRAME_MS:
                realtime_service.record_dropped_frame()
                frame_queue.task_done()
                continue

            response: RealtimePredictionResponse = await realtime_service.process_frame(
                raw_bytes=raw_bytes,
                frame_id=f_id,
                client_timestamp=client_ts,
                db=db,
                session_id=active_session_id,
            )
            await websocket.send_text(response.model_dump_json())
            frame_queue.task_done()
        except asyncio.CancelledError:
            break
        except (InvalidImageError, PayloadTooLargeError) as exc:
            err = RealtimeErrorMessage(code="INVALID_FRAME", message=str(exc))
            await websocket.send_text(err.model_dump_json())
            frame_queue.task_done()
        except Exception as exc:
            logger.error(f"Error in frame worker: {exc}", exc_info=True)
            err = RealtimeErrorMessage(code="INFERENCE_ERROR", message=f"Pipeline error: {exc}")
            try:
                await websocket.send_text(err.model_dump_json())
            except Exception:
                pass
            frame_queue.task_done()


@router.websocket("/realtime/emotion")
async def realtime_emotion_stream(  # noqa: C901
    websocket: WebSocket,
    session_id: str | None = Query(default=None, description="Optional existing session UUID"),
    session_name: str | None = Query(
        default=None, description="Optional label for auto-created session"
    ),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Real-time bidirectional WebSocket endpoint for webcam & video emotion detection."""
    await websocket.accept()

    # 1. Initialize Inference Engine & Services
    engine: EmotionInferenceEngine | None = getattr(websocket.app.state, "inference_engine", None)
    if engine is None:
        error_msg = RealtimeErrorMessage(
            code="MODEL_NOT_READY",
            message="Phase 09 Inference Engine is not initialized or still warming up.",
        )
        await websocket.send_text(error_msg.model_dump_json())
        await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    prediction_service = PredictionService(inference_engine=engine)
    realtime_service = RealTimeService(prediction_service=prediction_service)

    # 2. Database Session Initialization
    active_session_id: uuid.UUID | None = None
    if session_id:
        try:
            active_session_id = uuid.UUID(session_id)
        except ValueError:
            logger.warning(f"Invalid session UUID received: {session_id}")

    try:
        if active_session_id is None:
            active_session_id = await realtime_service.initialize_session(
                db=db,
                session_name=session_name or "Real-Time Webcam Session",
            )
    except Exception as exc:
        logger.warning(f"Could not initialize database session for WebSocket: {exc}")

    # 3. Send Connected Handshake Message
    status_msg = RealtimeStatusMessage(
        status="connected",
        session_id=str(active_session_id) if active_session_id else None,
        message="WebSocket connection established. Inference pipeline active.",
        details={
            "model_version": settings.MODEL_VERSION,
            "target_fps": settings.REALTIME_TARGET_FPS,
            "smoothing_alpha": settings.REALTIME_SMOOTHING_ALPHA,
            "max_frame_size_mb": settings.REALTIME_MAX_FRAME_SIZE_MB,
        },
    )
    await websocket.send_text(status_msg.model_dump_json())

    # 4. Single-Item Bounded Queue with Backpressure
    frame_queue: asyncio.Queue[tuple[bytes, int, float | None, float]] = asyncio.Queue(
        maxsize=settings.REALTIME_QUEUE_MAX_SIZE
    )
    stop_event = asyncio.Event()
    client_frame_counter = 0

    # 5. Background Frame Processing Worker Task
    worker_task = asyncio.create_task(
        _run_frame_worker(
            stop_event=stop_event,
            frame_queue=frame_queue,
            realtime_service=realtime_service,
            db=db,
            active_session_id=active_session_id,
            websocket=websocket,
        )
    )

    # 6. WebSocket Message Receiving Loop
    try:
        while True:
            message = await websocket.receive()
            receive_mono_time = time.perf_counter()

            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                client_frame_counter += 1
                f_id = client_frame_counter
                c_ts = time.time()

                if frame_queue.full():
                    try:
                        _ = frame_queue.get_nowait()
                        frame_queue.task_done()
                        realtime_service.record_dropped_frame()
                    except asyncio.QueueEmpty:
                        pass
                await frame_queue.put((raw_bytes, f_id, c_ts, receive_mono_time))

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    err = RealtimeErrorMessage(
                        code="INVALID_MESSAGE",
                        message="Payload could not be parsed as valid JSON.",
                    )
                    await websocket.send_text(err.model_dump_json())
                    continue

                msg_type = payload.get("type", "frame")
                if msg_type == "frame":
                    client_frame_counter = await _handle_json_frame(
                        payload,
                        client_frame_counter,
                        receive_mono_time,
                        frame_queue,
                        realtime_service,
                        websocket,
                    )
                elif msg_type == "ping":
                    pong = {
                        "type": "pong",
                        "client_timestamp": payload.get("timestamp"),
                        "server_timestamp": time.time(),
                    }
                    await websocket.send_text(json.dumps(pong))
                elif msg_type == "config":
                    await _handle_config_message(
                        payload, active_session_id, realtime_service, websocket
                    )
                else:
                    err = RealtimeErrorMessage(
                        code="INVALID_MESSAGE",
                        message=f"Unknown message type: '{msg_type}'.",
                    )
                    await websocket.send_text(err.model_dump_json())

    except (WebSocketDisconnect, ConnectionResetError):
        logger.info(f"WebSocket client disconnected for session: {active_session_id}")
    except Exception as exc:
        logger.warning(f"Unexpected WebSocket error: {exc}")
    finally:
        stop_event.set()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

        if active_session_id is not None:
            try:
                await realtime_service.finalize_session(db=db, session_id=active_session_id)
            except Exception as exc:
                logger.warning(f"Error finalizing session during teardown: {exc}")

        realtime_service.reset()
        logger.info(f"Cleaned up real-time stream resources for session {active_session_id}.")
