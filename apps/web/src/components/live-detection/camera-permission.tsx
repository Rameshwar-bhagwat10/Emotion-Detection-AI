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
    <div className={`p-6 rounded-2xl bg-zinc-900 border border-rose-900/50 text-zinc-200 shadow-xl ${className}`}>
      <div className="flex items-start gap-4">
        <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>

        <div className="flex-1">
          <h4 className="text-base font-semibold text-white mb-1">
            {isDenied
              ? "Camera Permission Required"
              : isUnavailable
              ? "No Camera Detected"
              : "Camera Connection Error"}
          </h4>
          <p className="text-sm text-zinc-400 mb-4">
            {errorMessage ||
              "Please allow camera access in your browser settings to enable real-time emotion detection."}
          </p>

          {isDenied && (
            <ul className="text-xs text-zinc-400 list-disc list-inside space-y-1 mb-4">
              <li>Click the camera/lock icon in your browser address bar.</li>
              <li>Change camera permissions from <em>Block</em> to <em>Allow</em>.</li>
              <li>Refresh the page or click <em>Retry Camera</em> below.</li>
            </ul>
          )}

          <button
            onClick={onRetry}
            className="px-4 py-2 rounded-xl text-sm font-semibold text-white bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 transition-colors cursor-pointer"
          >
            Retry Camera Access
          </button>
        </div>
      </div>
    </div>
  );
}

export default CameraPermission;
