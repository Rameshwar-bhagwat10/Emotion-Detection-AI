"use client";

import React from "react";
import { CameraPermission } from "@/components/live-detection/camera-permission";
import { CameraView } from "@/components/live-detection/camera-view";
import { DetectionControls } from "@/components/live-detection/detection-controls";
import { DetectionResult } from "@/components/live-detection/detection-result";
import { useRealtimeEmotion } from "@/hooks/useRealtimeEmotion";
import { Activity, Clock, Sparkles, Users } from "lucide-react";

export default function LiveEmotionDetectionPage() {
  const {
    videoRef,
    cameraState,
    connectionState,
    sessionState,
    isStreaming,
    predictions,
    sessionId,
    sessionDuration,
    totalPredictions,
    metrics,
    error,
    config,
    setConfig,
    start,
    stop,
  } = useRealtimeEmotion({
    targetFps: 10,
    jpegQuality: 0.8,
    processingWidth: 640,
    processingHeight: 480,
    smoothingEnabled: true,
    isMirrored: true,
  });

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-zinc-800 pb-6">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight bg-gradient-to-r from-white via-zinc-200 to-zinc-400 bg-clip-text text-transparent">
              Real-Time Webcam Emotion Detection
            </h1>
            <span className="px-2.5 py-0.5 text-xs font-bold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Phase 12
            </span>
          </div>
          <p className="text-sm text-zinc-400 mt-1">
            Low-latency bidirectional WebSocket streaming powered by ResNet-18 Champion and YuNet neural face detection.
          </p>
        </div>

        {/* Status Badges */}
        <div className="flex items-center gap-2">
          {/* Session Lifecycle Badge */}
          {sessionState === "ACTIVE" && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-indigo-500/10 border border-indigo-500/25 text-xs font-semibold text-indigo-300">
              <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
              <span>Session: Active</span>
            </div>
          )}

          {/* Connection State Badge */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-zinc-900 border border-zinc-800 text-xs font-semibold">
            <span
              className={`w-2 h-2 rounded-full ${
                connectionState === "connected"
                  ? "bg-emerald-400 animate-pulse"
                  : connectionState === "connecting"
                  ? "bg-amber-400 animate-ping"
                  : "bg-zinc-600"
              }`}
            />
            <span className="capitalize text-zinc-300">
              WebSocket: {connectionState}
            </span>
          </div>
        </div>
      </div>

      {/* Active Session Telemetry HUD Banner (if streaming) */}
      {isStreaming && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 rounded-2xl bg-zinc-900/70 border border-zinc-800/80 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-zinc-800/80 text-zinc-400">
              <Clock className="w-4 h-4 text-indigo-400" />
            </div>
            <div>
              <span className="text-[10px] uppercase font-semibold text-zinc-500 block">
                Session Duration
              </span>
              <span className="text-base font-bold font-mono text-zinc-100">
                {formatDuration(sessionDuration)}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-zinc-800/80 text-zinc-400">
              <Activity className="w-4 h-4 text-emerald-400" />
            </div>
            <div>
              <span className="text-[10px] uppercase font-semibold text-zinc-500 block">
                Frames Inferred
              </span>
              <span className="text-base font-bold font-mono text-zinc-100">
                {totalPredictions}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-zinc-800/80 text-zinc-400">
              <Users className="w-4 h-4 text-sky-400" />
            </div>
            <div>
              <span className="text-[10px] uppercase font-semibold text-zinc-500 block">
                Faces Tracked
              </span>
              <span className="text-base font-bold font-mono text-zinc-100">
                {predictions.length}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-zinc-800/80 text-zinc-400">
              <Sparkles className="w-4 h-4 text-purple-400" />
            </div>
            <div>
              <span className="text-[10px] uppercase font-semibold text-zinc-500 block">
                Active Session UUID
              </span>
              <span className="text-[11px] font-mono text-zinc-300 truncate max-w-[120px] block" title={sessionId || "Auto Session"}>
                {sessionId ? sessionId.slice(0, 8) + "..." : "Auto"}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Permission Alert (if applicable) */}
      {cameraState !== "idle" && cameraState !== "granted" && (
        <CameraPermission
          cameraState={cameraState}
          errorMessage={error}
          onRetry={start}
        />
      )}

      {/* Main Interface Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Video Viewport & Controls (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="relative">
            <CameraView
              videoRef={videoRef}
              isStreaming={isStreaming}
              isMirrored={config.isMirrored}
              predictions={predictions}
              className="w-full aspect-[4/3]"
            />

            {/* No Face Detected Floating Alert Overlay */}
            {isStreaming && predictions.length === 0 && (
              <div className="absolute top-4 left-1/2 -translate-x-1/2 px-4 py-2 rounded-full bg-black/70 backdrop-blur-md border border-zinc-700/60 text-xs font-medium text-amber-300 shadow-xl flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
                <span>No face detected — Position your face inside camera frame</span>
              </div>
            )}
          </div>

          <DetectionControls
            isStreaming={isStreaming}
            cameraState={cameraState}
            connectionState={connectionState}
            config={config}
            onConfigChange={(newConfig) => setConfig((prev) => ({ ...prev, ...newConfig }))}
            onStart={start}
            onStop={stop}
          />
        </div>

        {/* Right Column: Telemetry & Emotion Predictions (5 cols) */}
        <div className="lg:col-span-5">
          <DetectionResult
            predictions={predictions}
            metrics={metrics}
            isStreaming={isStreaming}
          />
        </div>
      </div>
    </div>
  );
}
