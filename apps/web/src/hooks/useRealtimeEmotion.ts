"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  CameraPermissionState,
  DetectedFaceRealtime,
  RealtimePredictionResponse,
  RealtimeStreamConfig,
  ServerRealtimeMessage,
  WebSocketConnectionState,
} from "@/types/realtime";
import { createSession, endSession } from "@/lib/api/endpoints";
import { env } from "@/config/environment";
import {
  getOrCreateUserId,
  saveUserSession,
  updateUserSession,
} from "@/lib/storage/user-storage";

export type SessionLifecycleState =
  | "NOT_STARTED"
  | "STARTING"
  | "ACTIVE"
  | "STOPPING"
  | "COMPLETED"
  | "ERROR";

const DEFAULT_CONFIG: RealtimeStreamConfig = {
  targetFps: 10,
  jpegQuality: 0.8,
  processingWidth: 640,
  processingHeight: 480,
  smoothingEnabled: true,
  confidenceThreshold: 0.4,
  isMirrored: true,
};

export function useRealtimeEmotion(initialConfig: Partial<RealtimeStreamConfig> = {}) {
  const [config, setConfig] = useState<RealtimeStreamConfig>({
    ...DEFAULT_CONFIG,
    ...initialConfig,
  });

  // State Machines
  const [cameraState, setCameraState] = useState<CameraPermissionState>("idle");
  const [connectionState, setConnectionState] = useState<WebSocketConnectionState>("idle");
  const [sessionState, setSessionState] = useState<SessionLifecycleState>("NOT_STARTED");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);

  // Predictions & Session Data
  const [latestResponse, setLatestResponse] = useState<RealtimePredictionResponse | null>(null);
  const [predictions, setPredictions] = useState<DetectedFaceRealtime[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionDuration, setSessionDuration] = useState<number>(0);
  const [totalPredictions, setTotalPredictions] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  // Telemetry Metrics
  const [metrics, setMetrics] = useState({
    cameraFps: 0,
    inferenceFps: 0,
    latencyMs: 0,
    droppedFrames: 0,
  });

  // Dynamic frame dimensions for exact bounding box alignment
  const [processedFrameDims, setProcessedFrameDims] = useState<{ width: number; height: number }>({
    width: 640,
    height: 480,
  });
  const processedDimsRef = useRef<{ width: number; height: number }>({ width: 640, height: 480 });

  // DOM & Hardware Refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  // Timers and counters
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const sessionTimerRef = useRef<NodeJS.Timeout | null>(null);
  const cameraFpsTrackerRef = useRef<{ count: number; lastTime: number }>({ count: 0, lastTime: 0 });
  const frameIdCounterRef = useRef<number>(0);
  const isStreamingRef = useRef<boolean>(false);
  const activeSessionIdRef = useRef<string | null>(null);

  isStreamingRef.current = isStreaming;
  activeSessionIdRef.current = sessionId;

  // Offscreen canvas setup
  useEffect(() => {
    if (typeof window !== "undefined" && !offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement("canvas");
    }
  }, []);

  /**
   * Capture and transmit single video frame over binary WebSocket.
   */
  const captureAndSendFrame = useCallback(() => {
    const video = videoRef.current;
    const socket = socketRef.current;
    const offscreen = offscreenCanvasRef.current;

    if (!video || !socket || !offscreen || socket.readyState !== WebSocket.OPEN) {
      return;
    }

    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return;
    }

    const vWidth = video.videoWidth || 640;
    const vHeight = video.videoHeight || 480;
    const maxDim = 640;
    let targetW = maxDim;
    let targetH = Math.round(maxDim * (vHeight / vWidth));
    if (vHeight > vWidth) {
      targetH = maxDim;
      targetW = Math.round(maxDim * (vWidth / vHeight));
    }

    // Ensure dimensions are even integers for encoder stability
    targetW = targetW % 2 === 0 ? targetW : targetW - 1;
    targetH = targetH % 2 === 0 ? targetH : targetH - 1;

    offscreen.width = targetW;
    offscreen.height = targetH;

    const ctx = offscreen.getContext("2d");
    if (!ctx) return;

    // Draw current video frame to offscreen canvas without distortion
    ctx.drawImage(video, 0, 0, targetW, targetH);

    if (
      processedDimsRef.current.width !== targetW ||
      processedDimsRef.current.height !== targetH
    ) {
      processedDimsRef.current = { width: targetW, height: targetH };
      setProcessedFrameDims({ width: targetW, height: targetH });
    }

    // Track camera capture FPS
    const now = performance.now();
    const tracker = cameraFpsTrackerRef.current;
    tracker.count += 1;
    if (now - tracker.lastTime >= 1000) {
      setMetrics((prev) => ({
        ...prev,
        cameraFps: Math.round((tracker.count * 1000) / (now - tracker.lastTime)),
      }));
      tracker.count = 0;
      tracker.lastTime = now;
    }

    // Binary transmission of JPEG frame
    frameIdCounterRef.current += 1;
    offscreen.toBlob(
      (blob) => {
        if (!blob || socket.readyState !== WebSocket.OPEN) return;
        blob.arrayBuffer().then((buffer) => {
          if (socket.readyState === WebSocket.OPEN) {
            socket.send(buffer);
          }
        });
      },
      "image/jpeg",
      config.jpegQuality
    );
  }, [config.jpegQuality]);

  /**
   * Connect to WebSocket backend endpoint with session correlation.
   */
  const connectWebSocket = useCallback((activeSessionUUID?: string): Promise<WebSocket> => {
    return new Promise((resolve, reject) => {
      const baseWsUrl = process.env.NEXT_PUBLIC_WS_URL || env.wsUrl;

      const url = new URL(baseWsUrl);
      if (activeSessionUUID) {
        url.searchParams.set("session_id", activeSessionUUID);
      }
      url.searchParams.set("session_name", "Webcam Detection Session");

      setConnectionState("connecting");
      const socket = new WebSocket(url.toString());
      socket.binaryType = "arraybuffer";

      socket.onopen = () => {
        setConnectionState("connected");
        setError(null);
        resolve(socket);
      };

      socket.onmessage = (event) => {
        try {
          const data: ServerRealtimeMessage = JSON.parse(event.data);

          if (data.type === "prediction") {
            const now = Date.now();
            const latency = data.client_timestamp
              ? Math.max(0, now - data.client_timestamp)
              : data.metrics.total_processing_time_ms;

            setLatestResponse(data);
            setPredictions(data.faces);
            setTotalPredictions((prev) => prev + 1);

            setMetrics((prev) => ({
              ...prev,
              inferenceFps: data.metrics.fps,
              latencyMs: Math.round(latency),
              droppedFrames: data.metrics.dropped_frames,
            }));
          } else if (data.type === "status") {
            if (data.session_id) {
              setSessionId(data.session_id);
              activeSessionIdRef.current = data.session_id;
              saveUserSession({
                id: data.session_id,
                name: `Webcam Session ${new Date().toLocaleTimeString()}`,
                status: "active",
                started_at: new Date().toISOString(),
              });
            }
          } else if (data.type === "error") {
            setError(`${data.code}: ${data.message}`);
          }
        } catch (err) {
          console.warn("Failed to parse WebSocket message:", err);
        }
      };

      socket.onerror = (event) => {
        setConnectionState("error");
        setError("WebSocket connection error. Verify FastAPI server is running.");
        reject(event);
      };

      socket.onclose = () => {
        setConnectionState("disconnected");
      };

      socketRef.current = socket;
    });
  }, []);

  /**
   * Start camera feed and initialize real-time analysis stream.
   */
  const start = useCallback(async () => {
    setError(null);
    setCameraState("requesting");
    setSessionState("STARTING");

    try {
      // 1. Request Hardware Camera Permission
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: config.processingWidth },
          height: { ideal: config.processingHeight },
          frameRate: { ideal: 30 },
        },
        audio: false,
      });

      mediaStreamRef.current = stream;
      setCameraState("granted");

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // 2. Initialize or connect backend session with user isolation
      let initialSessionId: string | undefined;
      try {
        const userId = getOrCreateUserId();
        const sessionName = `Webcam Session ${new Date().toLocaleTimeString()}`;
        const sessionRecord = await createSession(sessionName, userId);
        initialSessionId = sessionRecord.id;
        setSessionId(sessionRecord.id);
        activeSessionIdRef.current = sessionRecord.id;
        saveUserSession({
          id: sessionRecord.id,
          name: sessionRecord.name || sessionName,
          status: sessionRecord.status || "active",
          started_at: sessionRecord.started_at || new Date().toISOString(),
        });
      } catch (sessErr) {
        console.warn("Backend session creation warning (continuing with auto-session):", sessErr);
      }

      // 3. Connect to WebSocket
      await connectWebSocket(initialSessionId);

      // 4. Start Frame Sampling Loop
      setIsStreaming(true);
      setSessionState("ACTIVE");
      setSessionDuration(0);
      setTotalPredictions(0);
      cameraFpsTrackerRef.current = { count: 0, lastTime: performance.now() };

      const intervalMs = Math.round(1000 / config.targetFps);
      frameIntervalRef.current = setInterval(() => {
        captureAndSendFrame();
      }, intervalMs);

      // 5. Start Session Duration Timer
      sessionTimerRef.current = setInterval(() => {
        setSessionDuration((prev) => prev + 1);
      }, 1000);
    } catch (err: unknown) {
      const mediaError = err as Error;
      if (mediaError.name === "NotAllowedError" || mediaError.name === "PermissionDeniedError") {
        setCameraState("denied");
        setError("Camera access denied. Please allow camera permissions in your browser settings.");
      } else if (
        mediaError.name === "NotFoundError" ||
        mediaError.name === "DevicesNotFoundError"
      ) {
        setCameraState("unavailable");
        setError("No camera device was detected on this system.");
      } else {
        setCameraState("error");
        setError(`Failed to initialize camera: ${mediaError.message || "Unknown error"}`);
      }
      setIsStreaming(false);
      setSessionState("ERROR");
    }
  }, [config, connectWebSocket, captureAndSendFrame]);

  /**
   * Stop camera feed, release hardware tracks, finalize session, and clean up.
   */
  const stop = useCallback(async () => {
    setSessionState("STOPPING");
    setIsStreaming(false);

    // 1. Clear frame sampling interval & session timer
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (sessionTimerRef.current) {
      clearInterval(sessionTimerRef.current);
      sessionTimerRef.current = null;
    }

    // 2. Stop all MediaStream hardware tracks
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // 3. Finalize backend session and local storage record if ID is known
    const currentSessId = activeSessionIdRef.current;
    if (currentSessId) {
      updateUserSession(currentSessId, {
        status: "completed",
        ended_at: new Date().toISOString(),
      });
      try {
        await endSession(currentSessId);
      } catch (endErr) {
        console.warn("Session finalize notice:", endErr);
      }
    }

    // 4. Close WebSocket cleanly
    if (socketRef.current) {
      socketRef.current.close(1000, "User stopped stream");
      socketRef.current = null;
    }

    setCameraState("idle");
    setConnectionState("disconnected");
    setSessionState("COMPLETED");
    setPredictions([]);
    setLatestResponse(null);
    setMetrics({ cameraFps: 0, inferenceFps: 0, latencyMs: 0, droppedFrames: 0 });
  }, []);

  // Guarantee resource cleanup on unmount
  useEffect(() => {
    return () => {
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
      if (sessionTimerRef.current) clearInterval(sessionTimerRef.current);
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (socketRef.current) {
        socketRef.current.close(1000, "Component unmounted");
      }
    };
  }, []);

  return {
    videoRef,
    canvasRef,
    cameraState,
    connectionState,
    sessionState,
    isStreaming,
    predictions,
    latestResponse,
    sessionId,
    sessionDuration,
    totalPredictions,
    metrics,
    error,
    config,
    setConfig,
    processedFrameDims,
    start,
    stop,
  };
}
