"use client";

import React, { useRef, useState, useEffect } from "react";
import { Camera, Activity } from "lucide-react";
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
    sessionState,
    isStreaming,
    predictions,
    sessionDuration,
    totalPredictions,
    metrics,
    error,
    config,
    setConfig,
    processedFrameDims,
    start,
    stop,
  } = useRealtimeEmotion({
    targetFps: 10,
    jpegQuality: 0.8,
    processingWidth: 640,
    processingHeight: 360,
    smoothingEnabled: true,
    isMirrored: true,
  });

  const cameraContainerRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  const toggleFullscreen = async () => {
    if (!cameraContainerRef.current) return;
    try {
      if (!document.fullscreenElement) {
        await cameraContainerRef.current.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch (err) {
      console.error("Fullscreen error:", err);
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="space-y-6">
      {/* 1. Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-[#222222]">
        <div>
          <div className="font-mono text-xs uppercase tracking-[2px] text-[#888888] mb-1 flex items-center gap-2">
            <Camera className="w-3.5 h-3.5 text-emerald-400" />
            <span>LIVE DETECTION</span>
            <span className="text-[#3a3a3a]">/</span>
            <span className="text-[#c3d9f3]">WEBCAM FEED</span>
          </div>
          <h1 className="font-display text-2xl sm:text-3xl uppercase tracking-[2px] text-white">
            Real-Time Emotion Recognition
          </h1>
          <p className="font-sans text-xs text-[#888888] mt-1">
            Real-time face tracking and 7-class emotion probability analysis with instant visual feedback.
          </p>
        </div>

        {/* Status Indicators */}
        <div className="flex items-center gap-2.5 self-start sm:self-auto">
          {sessionState === "ACTIVE" && (
            <div className="flex items-center gap-2 px-3 py-1.5 border border-emerald-500/40 bg-emerald-950/20 font-mono text-[11px] uppercase tracking-[1.5px] text-emerald-400 rounded-none">
              <span className="w-2 h-2 rounded-none bg-emerald-400 animate-pulse" />
              <span>LIVE ACTIVE</span>
            </div>
          )}

          <div className="flex items-center gap-2 px-3 py-1.5 border border-[#262626] bg-[#0d0d0d] font-mono text-[11px] uppercase tracking-[1.5px] text-[#cccccc] rounded-none">
            <Activity className="w-3 h-3 text-[#c3d9f3]" />
            <span>SERVER: {connectionState === "connected" ? "ONLINE" : connectionState.toUpperCase()}</span>
          </div>
        </div>
      </div>

      {/* 2. Permission Alert (if applicable) */}
      {cameraState !== "idle" && cameraState !== "granted" && (
        <CameraPermission
          cameraState={cameraState}
          errorMessage={error}
          onRetry={start}
        />
      )}

      {/* 3. Main Interface Grid - Sized and Perfectly Aligned */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
        {/* Left Column: Enlarged Video Viewport & Single-Row Controls (8 cols on XL, 7 on LG) */}
        <div className="lg:col-span-7 xl:col-span-8 flex flex-col justify-between gap-3.5">
          {/* Camera Viewport Container */}
          <div
            ref={cameraContainerRef}
            className="relative border border-[#262626] bg-black rounded-none flex-1 overflow-hidden"
          >
            <CameraView
              videoRef={videoRef}
              isStreaming={isStreaming}
              isMirrored={config.isMirrored}
              predictions={predictions}
              frameWidth={processedFrameDims.width}
              frameHeight={processedFrameDims.height}
              smoothingEnabled={config.smoothingEnabled}
              onToggleFullscreen={toggleFullscreen}
              isFullscreen={isFullscreen}
              className="w-full h-full"
            />
          </div>

          {/* Single-Row Controls Bar */}
          <DetectionControls
            isStreaming={isStreaming}
            cameraState={cameraState}
            connectionState={connectionState}
            config={config}
            onConfigChange={(newConfig) => setConfig((prev) => ({ ...prev, ...newConfig }))}
            onStart={start}
            onStop={stop}
            onToggleFullscreen={toggleFullscreen}
            isFullscreen={isFullscreen}
          />
        </div>

        {/* Right Column: Emotion Analysis Box - Perfectly Aligned Height (4 cols on XL, 5 on LG) */}
        <div className="lg:col-span-5 xl:col-span-4 flex flex-col">
          <DetectionResult
            predictions={predictions}
            metrics={metrics}
            isStreaming={isStreaming}
            className="h-full"
          />
        </div>
      </div>

      {/* 4. Active Streaming Session Footer Info (if streaming) */}
      {isStreaming && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-px bg-[#222222] border border-[#222222] rounded-none">
          <div className="p-3 bg-[#0c0c0c]">
            <span className="font-mono text-[9px] uppercase tracking-[2px] text-[#666666] block mb-0.5">
              SESSION TIME
            </span>
            <span className="font-mono text-base text-white">
              {formatDuration(sessionDuration)}
            </span>
          </div>

          <div className="p-3 bg-[#0c0c0c]">
            <span className="font-mono text-[9px] uppercase tracking-[2px] text-[#666666] block mb-0.5">
              FRAMES PROCESSED
            </span>
            <span className="font-mono text-base text-white">
              {totalPredictions.toLocaleString()}
            </span>
          </div>

          <div className="p-3 bg-[#0c0c0c]">
            <span className="font-mono text-[9px] uppercase tracking-[2px] text-[#666666] block mb-0.5">
              FACES IN FRAME
            </span>
            <span className="font-mono text-base text-[#c3d9f3]">
              {predictions.length}
            </span>
          </div>

          <div className="p-3 bg-[#0c0c0c]">
            <span className="font-mono text-[9px] uppercase tracking-[2px] text-[#666666] block mb-0.5">
              TRACKING PIPELINE
            </span>
            <span className="font-mono text-xs text-emerald-400 truncate block mt-1">
              {predictions.length > 0 ? "LOCK ACTIVE" : "SCANNING..."}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
