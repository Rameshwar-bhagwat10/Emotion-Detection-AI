"use client";

import React, { useEffect, useState } from "react";
import { Camera, Maximize2, Minimize2 } from "lucide-react";
import { DetectedFaceRealtime } from "@/types/realtime";
import { FaceOverlay } from "./face-overlay";

export interface CameraViewProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isStreaming: boolean;
  isMirrored?: boolean;
  predictions: DetectedFaceRealtime[];
  frameWidth?: number;
  frameHeight?: number;
  smoothingEnabled?: boolean;
  className?: string;
  onToggleFullscreen?: () => void;
  isFullscreen?: boolean;
}

export function CameraView({
  videoRef,
  isStreaming,
  isMirrored = true,
  predictions,
  frameWidth = 640,
  frameHeight = 480,
  smoothingEnabled = true,
  className = "",
  onToggleFullscreen,
  isFullscreen = false,
}: CameraViewProps) {
  const [videoDims, setVideoDims] = useState<{ width: number; height: number }>({
    width: 640,
    height: 480,
  });

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => {
      if (video.videoWidth && video.videoHeight) {
        setVideoDims({ width: video.videoWidth, height: video.videoHeight });
      }
    };

    if (video.videoWidth && video.videoHeight) {
      setVideoDims({ width: video.videoWidth, height: video.videoHeight });
    }

    video.addEventListener("loadedmetadata", handleLoadedMetadata);
    video.addEventListener("playing", handleLoadedMetadata);
    video.addEventListener("resize", handleLoadedMetadata);
    return () => {
      video.removeEventListener("loadedmetadata", handleLoadedMetadata);
      video.removeEventListener("playing", handleLoadedMetadata);
      video.removeEventListener("resize", handleLoadedMetadata);
    };
  }, [videoRef, isStreaming]);

  return (
    <div
      className={`relative overflow-hidden bg-black border border-[#262626] aspect-video flex items-center justify-center rounded-none select-none ${
        isFullscreen ? "fixed inset-0 z-50 w-screen h-screen border-0 bg-black" : ""
      } ${className}`}
    >
      {/* Precision Corner Crosshairs */}
      <div className="absolute top-3 left-3 w-3 h-3 border-t border-l border-white/30 pointer-events-none z-20" />
      <div className="absolute top-3 right-3 w-3 h-3 border-t border-r border-white/30 pointer-events-none z-20" />
      <div className="absolute bottom-3 left-3 w-3 h-3 border-b border-l border-white/30 pointer-events-none z-20" />
      <div className="absolute bottom-3 right-3 w-3 h-3 border-b border-r border-white/30 pointer-events-none z-20" />

      {/* Top HUD Badges */}
      <div className="absolute top-3 left-3 z-20 flex items-center gap-2 pl-4">
        <div className="flex items-center gap-2 px-2.5 py-1 bg-black/80 backdrop-blur-sm border border-[#2a2a2a] text-[10px] font-mono uppercase tracking-[1px]">
          <span
            className={`w-1.5 h-1.5 rounded-none ${
              isStreaming
                ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]"
                : "bg-[#555555]"
            }`}
          />
          <span className={isStreaming ? "text-emerald-400" : "text-[#888888]"}>
            {isStreaming ? "LIVE FEED" : "FEED OFFLINE"}
          </span>
          {isStreaming && (
            <>
              <span className="text-[#3a3a3a]">·</span>
              <span className="text-[#777777]">
                {videoDims.width}x{videoDims.height}
              </span>
            </>
          )}
        </div>
      </div>

      {/* Fullscreen Toggle in Top-Right */}
      {onToggleFullscreen && (
        <div className="absolute top-3 right-3 z-20 pr-4">
          <button
            type="button"
            onClick={onToggleFullscreen}
            className="p-1.5 bg-black/80 hover:bg-black text-[#888888] hover:text-white border border-[#2a2a2a] rounded-none transition-colors cursor-pointer flex items-center gap-1.5 font-mono text-[10px] uppercase"
            title={isFullscreen ? "Exit Fullscreen (Esc)" : "Fullscreen View"}
            aria-label={isFullscreen ? "Exit Fullscreen" : "Fullscreen View"}
          >
            {isFullscreen ? (
              <>
                <Minimize2 className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">EXIT</span>
              </>
            ) : (
              <>
                <Maximize2 className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">FULLSCREEN</span>
              </>
            )}
          </button>
        </div>
      )}

      {/* Video Viewport */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={`w-full h-full object-cover transition-transform duration-200 ${
          isMirrored ? "scale-x-[-1]" : ""
        } ${!isStreaming ? "hidden" : "block"}`}
      />

      {/* Face Detection Canvas Overlay */}
      {isStreaming && (
        <FaceOverlay
          predictions={predictions}
          videoWidth={videoDims.width}
          videoHeight={videoDims.height}
          frameWidth={frameWidth}
          frameHeight={frameHeight}
          isMirrored={isMirrored}
          smoothingEnabled={smoothingEnabled}
          className="absolute inset-0 w-full h-full pointer-events-none z-10"
        />
      )}

      {/* No Face Detected Floating Notice */}
      {isStreaming && predictions.length === 0 && (
        <div className="absolute top-12 left-1/2 -translate-x-1/2 px-3.5 py-1.5 border border-amber-500/40 bg-black/85 backdrop-blur-sm text-amber-300 font-mono text-[11px] uppercase tracking-[1px] flex items-center gap-2 rounded-none z-20">
          <span className="w-1.5 h-1.5 bg-amber-400 animate-pulse" />
          <span>Face not detected · Look directly at camera</span>
        </div>
      )}

      {/* Clean Standby State Display */}
      {!isStreaming && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#090909] text-center p-6 z-10 select-none">
          <div className="w-12 h-12 rounded-none border border-[#282828] bg-[#121212] flex items-center justify-center mb-3 text-[#cccccc]">
            <Camera className="w-5 h-5 text-white" />
          </div>
          <h3 className="font-display text-xl uppercase tracking-[2px] text-white mb-1.5">
            Live Camera Standby
          </h3>
          <p className="font-sans text-xs text-[#777777] max-w-sm leading-relaxed mb-4">
            Click <span className="text-white font-mono uppercase">Start Camera</span> below to begin real-time facial emotion recognition.
          </p>
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-[#121212] border border-[#222222] font-mono text-[10px] uppercase tracking-[1px] text-[#888888] rounded-none">
            <span className="w-1.5 h-1.5 bg-emerald-400" />
            <span>Frames processed locally · Secure & private</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default CameraView;
