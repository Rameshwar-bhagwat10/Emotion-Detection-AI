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

  const [cameraState, setCameraState] = useState<CameraPermissionState>("idle");
  const [connectionState, setConnectionState] = useState<WebSocketConnectionState>("idle");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [latestResponse, setLatestResponse] = useState<RealtimePredictionResponse | null>(null);
  const [predictions, setPredictions] = useState<DetectedFaceRealtime[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Performance Telemetry State
  const [metrics, setMetrics] = useState({
    cameraFps: 0,
    inferenceFps: 0,
    latencyMs: 0,
    droppedFrames: 0,
  });

  // DOM & Media Refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  // Internal Loop Refs
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const cameraFpsTrackerRef = useRef<{ count: number; lastTime: number }>({ count: 0, lastTime: 0 });
  const frameIdCounterRef = useRef<number>(0);
  const isStreamingRef = useRef<boolean>(false);

  // Synchronize ref with state
  isStreamingRef.current = isStreaming;

  // Initialize offscreen canvas once
  useEffect(() => {
    if (typeof window !== "undefined" && !offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement("canvas");
    }
  }, []);

  /**
   * Capture and transmit a single video frame over WebSocket.
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

    const { processingWidth, processingHeight, jpegQuality } = config;
    offscreen.width = processingWidth;
    offscreen.height = processingHeight;

    const ctx = offscreen.getContext("2d");
    if (!ctx) return;

    // Draw video frame to offscreen canvas
    ctx.drawImage(video, 0, 0, processingWidth, processingHeight);

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

    // Capture as Blob and transmit binary frame
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
      jpegQuality
    );
  }, [config]);

  /**
   * Connect to WebSocket backend endpoint.
   */
  const connectWebSocket = useCallback((): Promise<WebSocket> => {
    return new Promise((resolve, reject) => {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const host = process.env.NEXT_PUBLIC_WS_URL || `${protocol}//${window.location.hostname}:8000/api/v1/realtime/emotion`;

      setConnectionState("connecting");
      const socket = new WebSocket(host);
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
            const latency = data.client_timestamp ? Math.max(0, now - data.client_timestamp) : data.metrics.total_processing_time_ms;

            setLatestResponse(data);
            setPredictions(data.faces);
            setMetrics((prev) => ({
              ...prev,
              inferenceFps: data.metrics.fps,
              latencyMs: Math.round(latency),
              droppedFrames: data.metrics.dropped_frames,
            }));
          } else if (data.type === "status") {
            // Status update
          } else if (data.type === "error") {
            setError(`${data.code}: ${data.message}`);
          }
        } catch (err) {
          console.warn("Failed to parse WebSocket message:", err);
        }
      };

      socket.onerror = (event) => {
        setConnectionState("error");
        setError("WebSocket connection error.");
        reject(event);
      };

      socket.onclose = () => {
        setConnectionState("disconnected");
        if (isStreamingRef.current) {
          // Attempt graceful reconnect if stream is still meant to be active
          setConnectionState("reconnecting");
        }
      };

      socketRef.current = socket;
    });
  }, []);

  /**
   * Start camera feed and begin real-time analysis stream.
   */
  const start = useCallback(async () => {
    setError(null);
    setCameraState("requesting");

    try {
      // 1. Request Browser Camera Permission
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

      // 2. Connect to WebSocket
      await connectWebSocket();

      // 3. Start Frame Sampling Timer Loop
      setIsStreaming(true);
      cameraFpsTrackerRef.current = { count: 0, lastTime: performance.now() };

      const intervalMs = Math.round(1000 / config.targetFps);
      frameIntervalRef.current = setInterval(() => {
        captureAndSendFrame();
      }, intervalMs);
    } catch (err: unknown) {
      const mediaError = err as Error;
      if (mediaError.name === "NotAllowedError" || mediaError.name === "PermissionDeniedError") {
        setCameraState("denied");
        setError("Camera permission was denied. Please allow camera access in browser settings.");
      } else if (mediaError.name === "NotFoundError" || mediaError.name === "DevicesNotFoundError") {
        setCameraState("unavailable");
        setError("No camera device was detected on this system.");
      } else {
        setCameraState("error");
        setError(`Failed to start camera: ${mediaError.message || "Unknown error"}`);
      }
      setIsStreaming(false);
    }
  }, [config, connectWebSocket, captureAndSendFrame]);

  /**
   * Stop camera feed, release hardware tracks, and terminate WebSocket connection.
   */
  const stop = useCallback(() => {
    setIsStreaming(false);

    // 1. Clear frame sampling interval
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    // 2. Stop all MediaStream tracks and release camera hardware
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // 3. Close WebSocket cleanly
    if (socketRef.current) {
      socketRef.current.close(1000, "User stopped stream");
      socketRef.current = null;
    }

    setCameraState("idle");
    setConnectionState("disconnected");
    setPredictions([]);
    setLatestResponse(null);
    setMetrics({ cameraFps: 0, inferenceFps: 0, latencyMs: 0, droppedFrames: 0 });
  }, []);

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  return {
    videoRef,
    canvasRef,
    cameraState,
    connectionState,
    isStreaming,
    predictions,
    latestResponse,
    metrics,
    error,
    config,
    setConfig,
    start,
    stop,
  };
}
