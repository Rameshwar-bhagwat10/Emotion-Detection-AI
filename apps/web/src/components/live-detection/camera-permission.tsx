"use client";

import React from "react";
import { CameraPermissionState } from "@/types/realtime";

export interface CameraPermissionProps {
  cameraState: CameraPermissionState;
  errorMessage: string | null;
  onRetry: () => void;
  className?: string;
}

export function CameraPermission({
  cameraState,
  errorMessage,
  onRetry,
  className = "",
}: CameraPermissionProps) {
  if (cameraState === "idle" || cameraState === "granted") {
    return null;
  }

  const isDenied = cameraState === "denied";
  const isUnavailable = cameraState === "unavailable";

  return (
    <div className={`p-6 bg-[#0d0d0d] border border-red-900/50 text-[#cccccc] ${className}`}>
      <div className="flex items-start gap-4">
        <div className="p-2 border border-red-500/40 bg-red-950/20 text-red-400 font-mono text-xs">
          [!]
        </div>

        <div className="flex-1">
          <h4 className="font-mono text-xs uppercase tracking-[2px] text-white mb-1">
            {isDenied
              ? "OPTIC SENSOR ACCESS BLOCKED"
              : isUnavailable
              ? "NO OPTIC HARDWARE DETECTED"
              : "SENSOR LINK FAILURE"}
          </h4>
          <p className="font-serif text-sm text-[#999999] mb-4">
            {errorMessage ||
              "Sensor streaming requires camera permissions in your browser. Please grant device permissions."}
          </p>

          {isDenied && (
            <ul className="font-mono text-xs text-[#666666] list-disc list-inside space-y-1 mb-4 uppercase tracking-wider">
              <li>Open browser permissions dialog via address bar reticle</li>
              <li>Toggle camera permission state to allowed</li>
              <li>Engage re-synchronize protocol below</li>
            </ul>
          )}

          <button
            type="button"
            onClick={onRetry}
            className="inline-flex items-center justify-center px-5 py-2 rounded-full border border-white bg-transparent text-white font-mono text-xs uppercase tracking-[2px] hover:bg-white hover:text-black transition-all cursor-pointer"
          >
            RE-ENGAGE SENSOR
          </button>
        </div>
      </div>
    </div>
  );
}

export default CameraPermission;
