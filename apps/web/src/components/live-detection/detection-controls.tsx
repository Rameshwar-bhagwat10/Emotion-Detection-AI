"use client";

import React from "react";
import { CameraPermissionState, RealtimeStreamConfig, WebSocketConnectionState } from "@/types/realtime";

export interface DetectionControlsProps {
  isStreaming: boolean;
  cameraState: CameraPermissionState;
  connectionState: WebSocketConnectionState;
  config: RealtimeStreamConfig;
  onConfigChange: (newConfig: Partial<RealtimeStreamConfig>) => void;
  onStart: () => void;
  onStop: () => void;
  className?: string;
}

export function DetectionControls({
  isStreaming,
  cameraState,
  connectionState,
  config,
  onConfigChange,
  onStart,
  onStop,
  className = "",
}: DetectionControlsProps) {
  const isPending = cameraState === "requesting" || connectionState === "connecting";

  return (
    <div className={`flex flex-wrap items-center justify-between gap-4 p-4 rounded-2xl bg-zinc-900/80 border border-zinc-800 backdrop-blur-md shadow-xl ${className}`}>
      {/* Primary Action Button */}
      <div className="flex items-center gap-3">
        {!isStreaming ? (
          <button
            onClick={onStart}
            disabled={isPending}
            className="flex items-center gap-2.5 px-6 py-3 rounded-xl font-semibold text-white bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 shadow-lg shadow-emerald-950/40 active:scale-95 transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            {isPending ? (
              <>
                <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>Starting Camera...</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Start Detection</span>
              </>
            )}
          </button>
        ) : (
          <button
            onClick={onStop}
            className="flex items-center gap-2.5 px-6 py-3 rounded-xl font-semibold text-white bg-gradient-to-r from-rose-600 to-red-700 hover:from-rose-500 hover:to-red-600 shadow-lg shadow-rose-950/40 active:scale-95 transition-all duration-150 cursor-pointer"
          >
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
            </svg>
            <span>Stop Camera</span>
          </button>
        )}

        {/* Live Indicator Badge */}
        {isStreaming && (
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            LIVE STREAM
          </span>
        )}
      </div>

      {/* Stream Tuning Controls */}
      <div className="flex flex-wrap items-center gap-3 text-sm">
        {/* Mirror Toggle */}
        <button
          onClick={() => onConfigChange({ isMirrored: !config.isMirrored })}
          className={`flex items-center gap-1.5 px-3.5 py-2 rounded-lg border transition-colors cursor-pointer ${
            config.isMirrored
              ? "bg-zinc-800 border-zinc-600 text-zinc-200"
              : "bg-zinc-950 border-zinc-800 text-zinc-400 hover:text-zinc-300"
          }`}
          title="Toggle webcam mirror view"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
          </svg>
          <span>Mirror</span>
        </button>

        {/* Target FPS Selector */}
        <div className="flex items-center gap-2 bg-zinc-950 border border-zinc-800 px-3 py-1.5 rounded-lg">
          <span className="text-zinc-400 text-xs font-medium">Rate:</span>
          <select
            value={config.targetFps}
            onChange={(e) => onConfigChange({ targetFps: Number(e.target.value) })}
            className="bg-transparent text-zinc-200 font-medium text-xs focus:outline-none cursor-pointer"
          >
            <option value={5} className="bg-zinc-900 text-zinc-200">5 FPS</option>
            <option value={10} className="bg-zinc-900 text-zinc-200">10 FPS</option>
            <option value={15} className="bg-zinc-900 text-zinc-200">15 FPS</option>
            <option value={20} className="bg-zinc-900 text-zinc-200">20 FPS</option>
          </select>
        </div>

        {/* Smoothing Toggle */}
        <button
          onClick={() => onConfigChange({ smoothingEnabled: !config.smoothingEnabled })}
          className={`flex items-center gap-1.5 px-3.5 py-2 rounded-lg border transition-colors cursor-pointer ${
            config.smoothingEnabled
              ? "bg-indigo-950/60 border-indigo-700/60 text-indigo-300"
              : "bg-zinc-950 border-zinc-800 text-zinc-400 hover:text-zinc-300"
          }`}
          title="Enable/disable temporal emotion smoothing (EMA)"
        >
          <span>✨ Smooth</span>
        </button>
      </div>
    </div>
  );
}

export default DetectionControls;
