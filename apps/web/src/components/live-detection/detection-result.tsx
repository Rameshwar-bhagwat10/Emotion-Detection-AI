"use client";

import React from "react";
import { DetectedFaceRealtime, EmotionClass } from "@/types/realtime";

export interface DetectionResultProps {
  predictions: DetectedFaceRealtime[];
  metrics: {
    cameraFps: number;
    inferenceFps: number;
    latencyMs: number;
    droppedFrames: number;
  };
  isStreaming: boolean;
  className?: string;
}

const EMOTION_BAR_COLORS: Record<EmotionClass, string> = {
  happy: "bg-emerald-500",
  neutral: "bg-sky-500",
  surprise: "bg-amber-500",
  sad: "bg-indigo-500",
  fear: "bg-purple-500",
  angry: "bg-rose-500",
  disgust: "bg-teal-500",
  uncertain: "bg-zinc-500",
};

const EMOTION_EMOJIS: Record<EmotionClass, string> = {
  happy: "😊",
  neutral: "😐",
  surprise: "😲",
  sad: "😢",
  fear: "😨",
  angry: "😡",
  disgust: "🤢",
  uncertain: "🤔",
};

export function DetectionResult({
  predictions,
  metrics,
  isStreaming,
  className = "",
}: DetectionResultProps) {
  const primaryFace = predictions.length > 0 ? predictions[0] : null;

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Telemetry Metrics HUD Card */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="p-4 rounded-2xl bg-zinc-900/90 border border-zinc-800 shadow-lg">
          <div className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">Inference FPS</div>
          <div className="text-2xl font-bold text-emerald-400">
            {isStreaming ? metrics.inferenceFps.toFixed(1) : "0.0"}
          </div>
          <div className="text-[11px] text-zinc-500 mt-0.5">Model processing rate</div>
        </div>

        <div className="p-4 rounded-2xl bg-zinc-900/90 border border-zinc-800 shadow-lg">
          <div className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">Roundtrip Latency</div>
          <div className="text-2xl font-bold text-sky-400">
            {isStreaming ? `${metrics.latencyMs} ms` : "0 ms"}
          </div>
          <div className="text-[11px] text-zinc-500 mt-0.5">Frame to overlay delay</div>
        </div>

        <div className="p-4 rounded-2xl bg-zinc-900/90 border border-zinc-800 shadow-lg">
          <div className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">Faces Detected</div>
          <div className="text-2xl font-bold text-indigo-400">
            {isStreaming ? predictions.length : 0}
          </div>
          <div className="text-[11px] text-zinc-500 mt-0.5">Tracked in session</div>
        </div>

        <div className="p-4 rounded-2xl bg-zinc-900/90 border border-zinc-800 shadow-lg">
          <div className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">Dropped Frames</div>
          <div className={`text-2xl font-bold ${metrics.droppedFrames > 0 ? "text-amber-400" : "text-zinc-400"}`}>
            {metrics.droppedFrames}
          </div>
          <div className="text-[11px] text-zinc-500 mt-0.5">Backpressure policy</div>
        </div>
      </div>

      {/* Primary Emotion Breakdown */}
      <div className="p-6 rounded-2xl bg-zinc-900/90 border border-zinc-800 shadow-xl">
        <h4 className="text-sm font-semibold text-zinc-300 uppercase tracking-wider mb-4 flex items-center justify-between">
          <span>Real-Time Emotion Probability Distribution</span>
          {primaryFace && (
            <span className="text-xs font-medium text-zinc-400">
              Track ID: #{primaryFace.face_id}
            </span>
          )}
        </h4>

        {primaryFace ? (
          <div className="space-y-3">
            {/* Dominant Emotion Highlight */}
            <div className="flex items-center justify-between p-3 rounded-xl bg-zinc-950/70 border border-zinc-800/80 mb-4">
              <div className="flex items-center gap-3">
                <span className="text-3xl">{EMOTION_EMOJIS[primaryFace.emotion] || "😐"}</span>
                <div>
                  <div className="text-lg font-bold text-white capitalize">{primaryFace.emotion}</div>
                  <div className="text-xs text-zinc-400">
                    Raw: {primaryFace.raw_emotion} ({Math.round(primaryFace.raw_confidence * 100)}%)
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-extrabold text-emerald-400">
                  {Math.round(primaryFace.confidence * 100)}%
                </div>
                <div className="text-xs text-zinc-500 font-medium">Smoothed Confidence</div>
              </div>
            </div>

            {/* 7-Class Distribution Bars */}
            <div className="space-y-2">
              {Object.entries(primaryFace.probabilities).map(([emotionKey, prob]) => {
                const emotion = emotionKey as EmotionClass;
                const pct = Math.round(prob * 100);
                const isDominant = emotion === primaryFace.emotion;

                return (
                  <div key={emotion} className="space-y-1">
                    <div className="flex justify-between text-xs font-medium">
                      <span className={`capitalize ${isDominant ? "text-zinc-100 font-bold" : "text-zinc-400"}`}>
                        {EMOTION_EMOJIS[emotion] || ""} {emotion}
                      </span>
                      <span className={isDominant ? "text-emerald-400 font-bold" : "text-zinc-400"}>
                        {pct}%
                      </span>
                    </div>
                    <div className="w-full h-2 rounded-full bg-zinc-950 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-150 ${
                          EMOTION_BAR_COLORS[emotion] || "bg-zinc-500"
                        } ${isDominant ? "opacity-100" : "opacity-60"}`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="py-12 text-center text-zinc-500 text-sm">
            {isStreaming ? "Looking for faces in camera view..." : "Start camera detection to view real-time emotion telemetry."}
          </div>
        )}
      </div>
    </div>
  );
}

export default DetectionResult;
