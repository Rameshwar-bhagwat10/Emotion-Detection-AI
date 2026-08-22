/**
 * TypeScript definitions for real-time WebSocket emotion detection streams.
 */

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export type EmotionClass =
  | "angry"
  | "disgust"
  | "fear"
  | "happy"
  | "sad"
  | "surprise"
  | "neutral"
  | "uncertain";

export interface DetectedFaceRealtime {
  face_id: number;
  bbox: BoundingBox;
  detection_confidence: number;
  emotion: EmotionClass;
  confidence: number;
  is_uncertain: boolean;
  probabilities: Record<string, number>;
  raw_emotion: EmotionClass;
  raw_confidence: number;
}

export interface RealtimeMetrics {
  inference_time_ms: number;
  total_processing_time_ms: number;
  fps: number;
  dropped_frames: number;
  server_timestamp: number;
}

export interface RealtimePredictionResponse {
  type: "prediction";
  frame_id: number;
  client_timestamp?: number;
  session_id?: string;
  faces_detected: number;
  faces: DetectedFaceRealtime[];
  metrics: RealtimeMetrics;
}

export interface RealtimeStatusMessage {
  type: "status";
  status: "connected" | "active" | "configured" | "paused" | "stopping" | "closed";
  session_id?: string;
  message?: string;
  details?: Record<string, unknown>;
}

export interface RealtimeErrorMessage {
  type: "error";
  code: string;
  message: string;
  frame_id?: number;
}

export type ServerRealtimeMessage =
  | RealtimePredictionResponse
  | RealtimeStatusMessage
  | RealtimeErrorMessage
  | { type: "pong"; client_timestamp?: number; server_timestamp?: number };

export type CameraPermissionState =
  | "idle"
  | "requesting"
  | "granted"
  | "denied"
  | "unavailable"
  | "error";

export type WebSocketConnectionState =
  | "idle"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected"
  | "error";

export interface RealtimeStreamConfig {
  targetFps: number;
  jpegQuality: number;
  processingWidth: number;
  processingHeight: number;
  smoothingEnabled: boolean;
  confidenceThreshold: number;
  isMirrored: boolean;
}
