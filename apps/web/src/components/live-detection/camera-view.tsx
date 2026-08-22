"use client";

import React, { useEffect, useState } from "react";
import { DetectedFaceRealtime } from "@/types/realtime";
import { FaceOverlay } from "./face-overlay";

export interface CameraViewProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isStreaming: boolean;
  isMirrored?: boolean;
  predictions: DetectedFaceRealtime[];
  className?: string;
}

export function CameraView({
  videoRef,
  isStreaming,
  isMirrored = true,
  predictions,
  className = "",
}: CameraViewProps) {
  const [videoDims, setVideoDims] = useState<{ width: number; height: number }>({
    width: 640,
    height: 480,
  });

  // Track actual video resolution once loaded
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => {
      if (video.videoWidth && video.videoHeight) {
        setVideoDims({ width: video.videoWidth, height: video.videoHeight });
      }
    };

    video.addEventListener("loadedmetadata", handleLoadedMetadata);
    return () => {
      video.removeEventListener("loadedmetadata", handleLoadedMetadata);
    };
  }, [videoRef]);

  return (
    <div className={`relative overflow-hidden rounded-2xl bg-zinc-950 border border-zinc-800 shadow-2xl ${className}`}>
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

      {/* Face Detection & Emotion Canvas Overlay */}
      {isStreaming && (
        <FaceOverlay
          predictions={predictions}
          videoWidth={videoDims.width}
          videoHeight={videoDims.height}
          isMirrored={isMirrored}
        />
      )}

      {/* Inactive Standby Screen */}
      {!isStreaming && (
        <div className="flex flex-col items-center justify-center min-h-[420px] p-8 text-center bg-gradient-to-b from-zinc-900 to-zinc-950">
          <div className="w-20 h-20 mb-6 rounded-2xl bg-zinc-800/80 border border-zinc-700/50 flex items-center justify-center shadow-inner">
            <svg
              className="w-10 h-10 text-zinc-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-zinc-200 mb-2">Camera Inactive</h3>
          <p className="text-sm text-zinc-400 max-w-sm">
            Click <strong className="text-zinc-200">Start Camera</strong> below to enable your webcam and start real-time emotion detection.
          </p>
        </div>
      )}
    </div>
  );
}

export default CameraView;
