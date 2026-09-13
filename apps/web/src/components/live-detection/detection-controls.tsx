"use client";

import React from "react";
import {
  Play,
  Square,
  FlipHorizontal,
  SlidersHorizontal,
  Maximize2,
  Minimize2,
} from "lucide-react";
import { CameraPermissionState, RealtimeStreamConfig, WebSocketConnectionState } from "@/types/realtime";

export interface DetectionControlsProps {
  isStreaming: boolean;
  cameraState: CameraPermissionState;
  connectionState: WebSocketConnectionState;
  config: RealtimeStreamConfig;
  onConfigChange: (newConfig: Partial<RealtimeStreamConfig>) => void;
  onStart: () => void;
  onStop: () => void;
  onToggleFullscreen?: () => void;
  isFullscreen?: boolean;
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
  onToggleFullscreen,
  isFullscreen = false,
  className = "",
}: DetectionControlsProps) {
  const isPending = cameraState === "requesting" || connectionState === "connecting";

  return (
    <div
      className={`flex flex-wrap lg:flex-nowrap items-center justify-between gap-3 p-3 bg-[#0d0d0d] border border-[#222222] rounded-none ${className}`}
    >
      {/* Primary Action Button (Start / Stop) */}
      <div className="flex items-center gap-3">
        {!isStreaming ? (
          <button
            type="button"
            onClick={onStart}
            disabled={isPending}
            className="h-10 px-6 rounded-none bg-white hover:bg-[#e6e6e6] text-black font-mono text-xs uppercase tracking-[2px] font-semibold transition-all flex items-center gap-2 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed select-none"
          >
            {isPending ? (
              <>
                <span className="inline-block w-3 h-3 border-2 border-black border-t-transparent animate-spin" />
                <span>STARTING...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>START CAMERA</span>
              </>
            )}
          </button>
        ) : (
          <button
            type="button"
            onClick={onStop}
            className="h-10 px-6 rounded-none bg-rose-600 hover:bg-rose-700 text-white font-mono text-xs uppercase tracking-[2px] font-semibold transition-all flex items-center gap-2 cursor-pointer select-none"
          >
            <Square className="w-3 h-3 fill-current" />
            <span>STOP CAMERA</span>
          </button>
        )}
      </div>

      {/* Stream Controls - All in the Same Single Row */}
      <div className="flex items-center gap-2 overflow-x-auto w-full sm:w-auto justify-end">
        {/* Mirror Toggle Button */}
        <button
          type="button"
          onClick={() => onConfigChange({ isMirrored: !config.isMirrored })}
          className={`h-10 px-3.5 border rounded-none text-xs font-mono uppercase tracking-[1px] transition-colors cursor-pointer flex items-center gap-2 shrink-0 select-none ${
            config.isMirrored
              ? "bg-[#1f1f1f] border-white text-white"
              : "bg-[#141414] border-[#262626] text-[#888888] hover:text-white hover:border-[#3a3a3a]"
          }`}
          title="Toggle camera mirror mode"
        >
          <FlipHorizontal className="w-3.5 h-3.5" />
          <span>{config.isMirrored ? "Mirror: On" : "Mirror: Off"}</span>
        </button>

        {/* Target FPS Selector */}
        <div className="h-10 px-3 border border-[#262626] bg-[#141414] rounded-none flex items-center gap-2 shrink-0 font-mono text-xs uppercase tracking-[1px]">
          <span className="text-[#777777]">FPS:</span>
          <select
            value={config.targetFps}
            onChange={(e) => onConfigChange({ targetFps: Number(e.target.value) })}
            className="bg-transparent text-white font-mono text-xs focus:outline-none cursor-pointer tracking-wider"
          >
            <option value={5} className="bg-[#141414] text-white">5</option>
            <option value={10} className="bg-[#141414] text-white">10</option>
            <option value={15} className="bg-[#141414] text-white">15</option>
            <option value={20} className="bg-[#141414] text-white">20</option>
          </select>
        </div>

        {/* Smoothing Toggle Button */}
        <button
          type="button"
          onClick={() => onConfigChange({ smoothingEnabled: !config.smoothingEnabled })}
          className={`h-10 px-3.5 border rounded-none text-xs font-mono uppercase tracking-[1px] transition-colors cursor-pointer flex items-center gap-2 shrink-0 select-none ${
            config.smoothingEnabled
              ? "bg-[#1f1f1f] border-[#c3d9f3] text-[#c3d9f3]"
              : "bg-[#141414] border-[#262626] text-[#888888] hover:text-white hover:border-[#3a3a3a]"
          }`}
          title="Toggle prediction smoothing"
        >
          <SlidersHorizontal className="w-3.5 h-3.5" />
          <span>{config.smoothingEnabled ? "Smooth: On" : "Smooth: Off"}</span>
        </button>

        {/* Fullscreen Button in the Controls Row */}
        {onToggleFullscreen && (
          <button
            type="button"
            onClick={onToggleFullscreen}
            className={`h-10 px-3.5 border rounded-none text-xs font-mono uppercase tracking-[1px] transition-colors cursor-pointer flex items-center gap-2 shrink-0 select-none ${
              isFullscreen
                ? "bg-white text-black border-white"
                : "bg-[#141414] border-[#262626] text-[#888888] hover:text-white hover:border-[#3a3a3a]"
            }`}
            title={isFullscreen ? "Exit Fullscreen (Esc)" : "Fullscreen Mode"}
          >
            {isFullscreen ? (
              <>
                <Minimize2 className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Exit</span>
              </>
            ) : (
              <>
                <Maximize2 className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Fullscreen</span>
              </>
            )}
          </button>
        )}
      </div>
    </div>
  );
}

export default DetectionControls;
