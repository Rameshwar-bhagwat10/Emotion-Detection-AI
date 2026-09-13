"use client";

import React from "react";
import { DetectedFaceRealtime, EmotionClass } from "@/types/realtime";
import { EMOTIONS, PredictionEmotion } from "@/types/emotion";

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

const EMOTIONS_ORDER: EmotionClass[] = [
  "happy",
  "neutral",
  "surprise",
  "sad",
  "fear",
  "angry",
  "disgust",
];

export function DetectionResult({
  predictions,
  metrics,
  isStreaming,
  className = "",
}: DetectionResultProps) {
  const primaryFace = predictions.length > 0 ? predictions[0] : null;
  const dominantMeta = primaryFace
    ? EMOTIONS[primaryFace.emotion as PredictionEmotion] || EMOTIONS.neutral
    : EMOTIONS.neutral;

  return (
    <div
      className={`bg-[#0c0c0c] border border-[#222222] rounded-none p-5 h-full flex flex-col justify-between space-y-4 ${className}`}
    >
      {/* 1. Header Bar */}
      <div className="flex items-center justify-between pb-3 border-b border-[#1f1f1f]">
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-none ${
              isStreaming ? "bg-emerald-400 animate-pulse shadow-[0_0_6px_rgba(52,211,153,0.6)]" : "bg-[#444444]"
            }`}
          />
          <span className="font-mono text-xs uppercase tracking-[2px] text-white">
            Live Emotions
          </span>
        </div>
        {primaryFace ? (
          <span className="font-mono text-[10px] text-[#c3d9f3] tracking-[1.5px] px-2 py-0.5 bg-[#141414] border border-[#262626] rounded-none">
            FACE #{primaryFace.face_id}
          </span>
        ) : (
          <span className="font-mono text-[10px] text-[#666666] tracking-[1px] uppercase">
            {isStreaming ? "SEARCHING..." : "STANDBY"}
          </span>
        )}
      </div>

      {/* 2. Dominant Emotion Hero Card */}
      {primaryFace ? (
        <div
          className="p-4 bg-[#121212] border flex items-center justify-between relative overflow-hidden rounded-none"
          style={{ borderColor: dominantMeta.color }}
        >
          {/* Subtle ambient emotion glow */}
          <div
            className="absolute -right-8 -top-8 w-24 h-24 blur-2xl opacity-15 pointer-events-none"
            style={{ backgroundColor: dominantMeta.color }}
          />

          <div className="flex items-center gap-3.5">
            <div
              className="w-12 h-12 rounded-none flex items-center justify-center text-3xl select-none"
              style={{
                backgroundColor: `${dominantMeta.color}15`,
                border: `1px solid ${dominantMeta.color}35`,
              }}
              role="img"
              aria-label={dominantMeta.label}
            >
              {dominantMeta.emoji}
            </div>
            <div>
              <span className="font-mono text-[9px] uppercase tracking-[2px] text-[#777777] block">
                Dominant Emotion
              </span>
              <div
                className="font-display text-2xl uppercase tracking-[1.5px] font-normal mt-0.5"
                style={{ color: dominantMeta.color }}
              >
                {dominantMeta.label}
              </div>
              <div className="font-sans text-[11px] text-[#888888] mt-0.5">
                {dominantMeta.description}
              </div>
            </div>
          </div>

          <div className="text-right pl-3">
            <div className="font-mono text-2xl text-white">
              {Math.round(primaryFace.confidence * 100)}%
            </div>
            <div className="font-mono text-[9px] uppercase tracking-[1px] text-[#777777] mt-0.5">
              Confidence
            </div>
          </div>
        </div>
      ) : (
        <div className="p-4 bg-[#121212] border border-[#1f1f1f] rounded-none flex items-center gap-3.5">
          <div className="w-12 h-12 bg-[#181818] border border-[#262626] rounded-none flex items-center justify-center text-2xl select-none">
            {isStreaming ? "👤" : "📷"}
          </div>
          <div>
            <div className="font-display text-lg uppercase tracking-[1px] text-white">
              {isStreaming ? "Looking For Face..." : "Live Feed Ready"}
            </div>
            <div className="font-sans text-xs text-[#777777] mt-0.5">
              {isStreaming
                ? "Position your face clearly within the camera view"
                : "Click Start Camera to begin real-time emotion detection"}
            </div>
          </div>
        </div>
      )}

      {/* 3. 7-Class Emotion Probability Bars */}
      <div className="space-y-2.5 flex-1 flex flex-col justify-center py-1">
        <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-[1.5px] text-[#666666] pb-1 border-b border-[#181818]">
          <span>Emotion Spectrum</span>
          <span>Probability</span>
        </div>

        {EMOTIONS_ORDER.map((emotion) => {
          const meta = EMOTIONS[emotion as PredictionEmotion] || EMOTIONS.neutral;
          const prob = primaryFace ? primaryFace.probabilities[emotion] || 0.0 : 0.0;
          const pct = Math.round(prob * 100);
          const isDominant = primaryFace ? emotion === primaryFace.emotion : false;

          return (
            <div key={emotion} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2 font-mono uppercase tracking-[1px]">
                  <span className="text-sm select-none" role="img" aria-label={meta.label}>
                    {meta.emoji}
                  </span>
                  <span className={isDominant ? "text-white font-medium" : "text-[#888888]"}>
                    {meta.label}
                  </span>
                </div>
                <span
                  className={`font-mono text-xs ${
                    isDominant ? "font-bold text-white" : "text-[#555555]"
                  }`}
                >
                  {primaryFace ? `${pct}%` : "—"}
                </span>
              </div>

              {/* Progress Bar with Emotion Color */}
              <div className="w-full h-1 bg-[#181818] overflow-hidden rounded-none">
                <div
                  className="h-full transition-all duration-200"
                  style={{
                    width: primaryFace ? `${Math.max(pct, 1)}%` : "0%",
                    backgroundColor: meta.color,
                    opacity: isDominant ? 1 : 0.35,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* 4. Integrated Telemetry Strip at Bottom */}
      <div className="grid grid-cols-4 gap-1.5 pt-3 border-t border-[#1f1f1f]">
        <div className="p-1.5 bg-[#111111] border border-[#1c1c1c] text-center rounded-none overflow-hidden">
          <div className="font-mono text-[8.5px] text-[#666666] uppercase tracking-[0.5px]">SPEED</div>
          <div className="font-mono text-[10.5px] text-white mt-0.5 whitespace-nowrap truncate" title={`${isStreaming ? metrics.inferenceFps.toFixed(1) : "0.0"} FPS`}>
            {isStreaming ? metrics.inferenceFps.toFixed(1) : "0.0"}{" "}
            <span className="text-[8.5px] text-[#777777]">FPS</span>
          </div>
        </div>
        <div className="p-1.5 bg-[#111111] border border-[#1c1c1c] text-center rounded-none overflow-hidden">
          <div className="font-mono text-[8.5px] text-[#666666] uppercase tracking-[0.5px]">LATENCY</div>
          <div className="font-mono text-[10.5px] text-white mt-0.5 whitespace-nowrap truncate" title={`${isStreaming ? metrics.latencyMs : 0} MS`}>
            {isStreaming ? metrics.latencyMs : 0}{" "}
            <span className="text-[8.5px] text-[#777777]">MS</span>
          </div>
        </div>
        <div className="p-1.5 bg-[#111111] border border-[#1c1c1c] text-center rounded-none overflow-hidden">
          <div className="font-mono text-[8.5px] text-[#666666] uppercase tracking-[0.5px]">FACES</div>
          <div className="font-mono text-[10.5px] text-[#c3d9f3] mt-0.5 whitespace-nowrap truncate">
            {isStreaming ? predictions.length : 0}
          </div>
        </div>
        <div className="p-1.5 bg-[#111111] border border-[#1c1c1c] text-center rounded-none overflow-hidden">
          <div className="font-mono text-[8.5px] text-[#666666] uppercase tracking-[0.5px]">DROPPED</div>
          <div
            className={`font-mono text-[10.5px] mt-0.5 whitespace-nowrap truncate ${
              metrics.droppedFrames > 0 ? "text-amber-400" : "text-[#555555]"
            }`}
          >
            {metrics.droppedFrames}
          </div>
        </div>
      </div>
    </div>
  );
}

export default DetectionResult;
