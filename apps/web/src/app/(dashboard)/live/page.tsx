"use client";

import React from "react";
import { CameraPermission } from "@/components/live-detection/camera-permission";
import { CameraView } from "@/components/live-detection/camera-view";
import { DetectionControls } from "@/components/live-detection/detection-controls";
import { DetectionResult } from "@/components/live-detection/detection-result";
import { useRealtimeEmotion } from "@/hooks/useRealtimeEmotion";

export default function LiveEmotionDetectionPage() {
  const {
    videoRef,
    cameraState,
    connectionState,
    isStreaming,
    predictions,
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

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100 p-6 md:p-10">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Page Header */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-zinc-800 pb-6">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight bg-gradient-to-r from-white via-zinc-200 to-zinc-400 bg-clip-text text-transparent">
                Real-Time Webcam Emotion Detection
              </h1>
              <span className="px-2.5 py-0.5 text-xs font-bold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Phase 11
              </span>
            </div>
            <p className="text-sm text-zinc-400 mt-1">
              Low-latency WebSocket streaming powered by ResNet-18 Champion and YuNet face detector.
            </p>
          </div>

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
            <CameraView
              videoRef={videoRef}
              isStreaming={isStreaming}
              isMirrored={config.isMirrored}
              predictions={predictions}
              className="w-full aspect-[4/3]"
            />

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
    </main>
  );
}
