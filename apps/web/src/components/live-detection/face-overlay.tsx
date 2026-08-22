"use client";

import React, { useEffect, useRef } from "react";
import { DetectedFaceRealtime, EmotionClass } from "@/types/realtime";

export interface FaceOverlayProps {
  predictions: DetectedFaceRealtime[];
  videoWidth: number;
  videoHeight: number;
  isMirrored?: boolean;
  className?: string;
}

const EMOTION_COLORS: Record<EmotionClass, { stroke: string; fill: string; text: string; bg: string; emoji: string }> = {
  happy: { stroke: "#10b981", fill: "rgba(16, 185, 129, 0.2)", text: "#ecfdf5", bg: "#059669", emoji: "😊" },
  neutral: { stroke: "#38bdf8", fill: "rgba(56, 189, 248, 0.2)", text: "#f0f9ff", bg: "#0284c7", emoji: "😐" },
  surprise: { stroke: "#f59e0b", fill: "rgba(245, 158, 11, 0.2)", text: "#fffbeb", bg: "#d97706", emoji: "😲" },
  sad: { stroke: "#6366f1", fill: "rgba(99, 102, 241, 0.2)", text: "#eef2ff", bg: "#4f46e5", emoji: "😢" },
  fear: { stroke: "#a855f7", fill: "rgba(168, 85, 247, 0.2)", text: "#faf5ff", bg: "#9333ea", emoji: "😨" },
  angry: { stroke: "#f43f5e", fill: "rgba(244, 63, 94, 0.2)", text: "#fff1f2", bg: "#e11d48", emoji: "😡" },
  disgust: { stroke: "#14b8a6", fill: "rgba(20, 184, 166, 0.2)", text: "#f0fdfa", bg: "#0d9488", emoji: "🤢" },
  uncertain: { stroke: "#94a3b8", fill: "rgba(148, 163, 184, 0.2)", text: "#f8fafc", bg: "#64748b", emoji: "🤔" },
};

export function FaceOverlay({
  predictions,
  videoWidth,
  videoHeight,
  isMirrored = true,
  className = "",
}: FaceOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Reset canvas dimensions to match video viewport
    canvas.width = videoWidth || 640;
    canvas.height = videoHeight || 480;

    // Clear previous overlay
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!predictions || predictions.length === 0) {
      return;
    }

    predictions.forEach((face) => {
      const { bbox, emotion, confidence, face_id } = face;
      const theme = EMOTION_COLORS[emotion] || EMOTION_COLORS.neutral;

      // Adjust coordinate for video mirroring if mirrored
      const x = isMirrored ? canvas.width - (bbox.x + bbox.width) : bbox.x;
      const y = bbox.y;
      const width = bbox.width;
      const height = bbox.height;

      // Draw bounding box glow & fill
      ctx.save();
      ctx.shadowColor = theme.stroke;
      ctx.shadowBlur = 10;
      ctx.strokeStyle = theme.stroke;
      ctx.lineWidth = 3;
      ctx.fillStyle = theme.fill;

      // Draw rounded rectangle
      const radius = 8;
      ctx.beginPath();
      ctx.roundRect(x, y, width, height, radius);
      ctx.stroke();
      ctx.fill();
      ctx.restore();

      // Corner accent brackets
      const bracketLen = Math.min(20, width / 4);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 3;

      // Top-Left
      ctx.beginPath();
      ctx.moveTo(x, y + bracketLen);
      ctx.lineTo(x, y);
      ctx.lineTo(x + bracketLen, y);
      ctx.stroke();

      // Top-Right
      ctx.beginPath();
      ctx.moveTo(x + width - bracketLen, y);
      ctx.lineTo(x + width, y);
      ctx.lineTo(x + width, y + bracketLen);
      ctx.stroke();

      // Bottom-Left
      ctx.beginPath();
      ctx.moveTo(x, y + height - bracketLen);
      ctx.lineTo(x, y + height);
      ctx.lineTo(x + bracketLen, y + height);
      ctx.stroke();

      // Bottom-Right
      ctx.beginPath();
      ctx.moveTo(x + width - bracketLen, y + height);
      ctx.lineTo(x + width, y + height);
      ctx.lineTo(x + width, y + height - bracketLen);
      ctx.stroke();

      // Label Header Badge
      const labelText = `${theme.emoji} ${emotion.toUpperCase()} ${Math.round(confidence * 100)}% (ID: ${face_id})`;
      ctx.font = "bold 13px Inter, -apple-system, sans-serif";
      const textMetrics = ctx.measureText(labelText);
      const pillWidth = textMetrics.width + 16;
      const pillHeight = 26;
      const pillY = Math.max(10, y - pillHeight - 6);

      // Label Background
      ctx.fillStyle = theme.bg;
      ctx.beginPath();
      ctx.roundRect(x, pillY, pillWidth, pillHeight, 6);
      ctx.fill();

      // Label Text
      ctx.fillStyle = theme.text;
      ctx.textBaseline = "middle";
      ctx.fillText(labelText, x + 8, pillY + pillHeight / 2);
    });
  }, [predictions, videoWidth, videoHeight, isMirrored]);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-none w-full h-full object-cover ${className}`}
    />
  );
}

export default FaceOverlay;
