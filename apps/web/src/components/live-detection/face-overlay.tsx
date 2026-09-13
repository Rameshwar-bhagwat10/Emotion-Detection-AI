"use client";

import React, { useEffect, useRef } from "react";
import { DetectedFaceRealtime } from "@/types/realtime";
import { EMOTIONS, PredictionEmotion } from "@/types/emotion";

export interface FaceOverlayProps {
  predictions: DetectedFaceRealtime[];
  videoWidth: number;
  videoHeight: number;
  frameWidth?: number;
  frameHeight?: number;
  isMirrored?: boolean;
  smoothingEnabled?: boolean;
  className?: string;
}

interface AnimatedBox {
  face_id: number;
  emotion: string;
  confidence: number;
  // Source detection coordinates from backend
  srcX: number;
  srcY: number;
  srcW: number;
  srcH: number;
  // Current interpolated screen coordinates
  currX: number;
  currY: number;
  currW: number;
  currH: number;
  opacity: number;
  missingFrames: number;
  initialized: boolean;
}

function toRgba(rgbStr: string, alpha: number): string {
  if (!rgbStr) return `rgba(56, 189, 248, ${alpha})`;
  if (rgbStr.startsWith("rgb(")) {
    return rgbStr.replace("rgb(", "rgba(").replace(")", `, ${alpha})`);
  }
  if (rgbStr.startsWith("rgba(")) {
    return rgbStr.replace(/[\d.]+\)$/, `${alpha})`);
  }
  return rgbStr;
}

export function FaceOverlay({
  predictions,
  videoWidth,
  videoHeight,
  frameWidth = 640,
  frameHeight = 480,
  isMirrored = true,
  smoothingEnabled = true,
  className = "",
}: FaceOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const trackedBoxesRef = useRef<Map<number, AnimatedBox>>(new Map());
  const animFrameIdRef = useRef<number | null>(null);

  // Store active props in a ref for zero-latency access in the 60fps render loop
  const propsRef = useRef({
    videoWidth,
    videoHeight,
    frameWidth,
    frameHeight,
    isMirrored,
    smoothingEnabled,
  });

  propsRef.current = {
    videoWidth,
    videoHeight,
    frameWidth,
    frameHeight,
    isMirrored,
    smoothingEnabled,
  };

  // Sync incoming face predictions into the tracked faces map
  useEffect(() => {
    const currentMap = trackedBoxesRef.current;
    const incomingFaceIds = new Set<number>();

    predictions.forEach((face) => {
      const { bbox, emotion, confidence, face_id } = face;
      incomingFaceIds.add(face_id);

      if (currentMap.has(face_id)) {
        const existing = currentMap.get(face_id)!;
        existing.srcX = bbox.x;
        existing.srcY = bbox.y;
        existing.srcW = bbox.width;
        existing.srcH = bbox.height;
        existing.emotion = emotion;
        existing.confidence = confidence;
        existing.missingFrames = 0;
      } else {
        // New face detected: initialize with placeholder screen coords to snap on next render
        currentMap.set(face_id, {
          face_id,
          emotion,
          confidence,
          srcX: bbox.x,
          srcY: bbox.y,
          srcW: bbox.width,
          srcH: bbox.height,
          currX: 0,
          currY: 0,
          currW: 0,
          currH: 0,
          opacity: 0,
          missingFrames: 0,
          initialized: false,
        });
      }
    });

    // Mark missing faces for graceful decay
    currentMap.forEach((box, id) => {
      if (!incomingFaceIds.has(id)) {
        box.missingFrames += 1;
        if (box.missingFrames > 30) {
          currentMap.delete(id);
        }
      }
    });
  }, [predictions]);

  // High-performance 60FPS render loop with DPI scaling and adaptive velocity lerp
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let isRunning = true;

    const render = () => {
      if (!isRunning) return;

      const {
        videoWidth: vW,
        videoHeight: vH,
        frameWidth: fW,
        frameHeight: fH,
        isMirrored: mirrored,
        smoothingEnabled: smooth,
      } = propsRef.current;

      const dpr = typeof window !== "undefined" ? Math.min(window.devicePixelRatio || 1, 2) : 1;
      const rect = canvas.getBoundingClientRect();
      const displayW = Math.round(rect.width) || vW || 640;
      const displayH = Math.round(rect.height) || vH || 480;

      const targetPixelW = Math.round(displayW * dpr);
      const targetPixelH = Math.round(displayH * dpr);

      if (canvas.width !== targetPixelW || canvas.height !== targetPixelH) {
        canvas.width = targetPixelW;
        canvas.height = targetPixelH;
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, displayW, displayH);

      // Compute exact object-cover transformation matching the underlying <video> element
      const srcW = fW || vW || 640;
      const srcH = fH || vH || 480;
      const scale = Math.max(displayW / srcW, displayH / srcH);
      const renderedW = srcW * scale;
      const renderedH = srcH * scale;
      const offsetX = (displayW - renderedW) / 2;
      const offsetY = (displayH - renderedH) / 2;

      const map = trackedBoxesRef.current;

      map.forEach((box, id) => {
        // Calculate target screen coordinates in the current viewport
        const targetW = box.srcW * scale;
        const targetH = box.srcH * scale;
        let targetX = offsetX + box.srcX * scale;
        if (mirrored) {
          targetX = displayW - offsetX - (box.srcX + box.srcW) * scale;
        }
        const targetY = offsetY + box.srcY * scale;

        // First frame initialization: snap directly without lag
        if (!box.initialized) {
          box.currX = targetX;
          box.currY = targetY;
          box.currW = targetW;
          box.currH = targetH;
          box.initialized = true;
          box.opacity = 0.2;
        } else if (!smooth) {
          // Smoothing disabled: snap directly to latest detector position
          box.currX = targetX;
          box.currY = targetY;
          box.currW = targetW;
          box.currH = targetH;
        } else {
          // Dynamic adaptive velocity lerp:
          // When stationary (dist < 4px): lerp = 0.28 to eliminate camera sensor jitter
          // When moving actively (dist > 4px): accelerates up to 0.78 for instant lag-free following
          const dx = targetX - box.currX;
          const dy = targetY - box.currY;
          const dw = targetW - box.currW;
          const dh = targetH - box.currH;
          const dist = Math.hypot(dx, dy);

          if (dist > 140) {
            // Rapid leap or re-acquired track: snap to avoid long unnatural slide
            box.currX = targetX;
            box.currY = targetY;
            box.currW = targetW;
            box.currH = targetH;
          } else {
            let lerpFactor = 0.28;
            if (dist > 4) {
              const accel = Math.min(1, (dist - 4) / 36);
              lerpFactor = 0.28 + accel * 0.50; // scales smoothly from 0.28 up to 0.78
            }

            box.currX += dx * lerpFactor;
            box.currY += dy * lerpFactor;
            box.currW += dw * lerpFactor;
            box.currH += dh * lerpFactor;
          }
        }

        // Smooth opacity transition for fade-in and fade-out
        if (box.missingFrames > 0) {
          box.opacity = Math.max(0, box.opacity - 0.08);
        } else {
          box.opacity = Math.min(1.0, box.opacity + 0.16);
        }

        if (box.opacity <= 0.01) {
          if (box.missingFrames > 18) {
            map.delete(id);
          }
          return;
        }

        const x = Math.round(box.currX);
        const y = Math.round(box.currY);
        const w = Math.round(box.currW);
        const h = Math.round(box.currH);

        if (w < 10 || h < 10) return;

        // Emotion tokens and styling
        const meta = EMOTIONS[box.emotion as PredictionEmotion] || EMOTIONS.neutral;
        const strokeColor = meta.color || "rgb(56, 189, 248)";

        ctx.save();
        ctx.globalAlpha = box.opacity;

        // 1. Holographic Ambient Scanner Tint (Soft vertical gradient)
        const bgGrad = ctx.createLinearGradient(x, y, x, y + h);
        bgGrad.addColorStop(0, toRgba(strokeColor, 0.04));
        bgGrad.addColorStop(1, toRgba(strokeColor, 0.09));
        ctx.fillStyle = bgGrad;
        ctx.fillRect(x, y, w, h);

        // 2. High-Tech Hairline Perimeter (Dashed border)
        ctx.save();
        ctx.strokeStyle = toRgba(strokeColor, 0.3);
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 4]);
        ctx.strokeRect(x, y, w, h);
        ctx.restore();

        // 3. Tactical Sharp Corner Brackets with Neon Bloom
        const bLen = Math.max(16, Math.min(32, Math.round(w * 0.22)));

        ctx.save();
        ctx.shadowColor = strokeColor;
        ctx.shadowBlur = 8;
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2.5;
        ctx.lineCap = "square";
        ctx.lineJoin = "miter";

        // Top-Left Bracket
        ctx.beginPath();
        ctx.moveTo(x, y + bLen);
        ctx.lineTo(x, y);
        ctx.lineTo(x + bLen, y);
        ctx.stroke();

        // Top-Right Bracket
        ctx.beginPath();
        ctx.moveTo(x + w - bLen, y);
        ctx.lineTo(x + w, y);
        ctx.lineTo(x + w, y + bLen);
        ctx.stroke();

        // Bottom-Left Bracket
        ctx.beginPath();
        ctx.moveTo(x, y + h - bLen);
        ctx.lineTo(x, y + h);
        ctx.lineTo(x + bLen, y + h);
        ctx.stroke();

        // Bottom-Right Bracket
        ctx.beginPath();
        ctx.moveTo(x + w - bLen, y + h);
        ctx.lineTo(x + w, y + h);
        ctx.lineTo(x + w, y + h - bLen);
        ctx.stroke();
        ctx.restore();

        // Corner Vertex Micro-Accents (0px sharp tick marks)
        ctx.fillStyle = strokeColor;
        ctx.fillRect(x - 1, y - 1, 3, 3);
        ctx.fillRect(x + w - 2, y - 1, 3, 3);
        ctx.fillRect(x - 1, y + h - 2, 3, 3);
        ctx.fillRect(x + w - 2, y + h - 2, 3, 3);

        // 4. Centroid Precision Crosshair
        const cx = Math.round(x + w / 2);
        const cy = Math.round(y + h / 2);
        const crossLen = 5;

        ctx.strokeStyle = toRgba(strokeColor, 0.7);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx - crossLen, cy);
        ctx.lineTo(cx + crossLen, cy);
        ctx.moveTo(cx, cy - crossLen);
        ctx.lineTo(cx, cy + crossLen);
        ctx.stroke();

        // Centroid center micro-dot
        ctx.fillStyle = strokeColor;
        ctx.fillRect(cx - 1, cy - 1, 2, 2);

        // 5. Floating Emotion HUD Badge
        const pct = Math.round(box.confidence * 100);
        const emoji = meta.emoji || "😐";
        const label = (meta.label || "NEUTRAL").toUpperCase();
        const pctText = `${pct}%`;

        // Measure text layout
        ctx.font = 'bold 11px "JetBrains Mono", ui-monospace, SFMono-Regular, monospace';
        const labelWidth = ctx.measureText(label).width;
        ctx.font = 'bold 10px "JetBrains Mono", ui-monospace, monospace';
        const pctMetrics = ctx.measureText(pctText);
        const pctBoxWidth = Math.round(pctMetrics.width + 12);

        const emojiWidth = 20;
        const badgeW = Math.max(140, Math.round(emojiWidth + labelWidth + pctBoxWidth + 34));
        const badgeH = 28;

        // Vertical positioning: above frame or inside if near top boundary
        let badgeY = Math.round(y - badgeH - 8);
        if (badgeY < 8) {
          badgeY = Math.round(y + 8);
        }

        // Horizontal alignment: clamped within visible canvas edges
        let badgeX = Math.round(x);
        badgeX = Math.max(8, Math.min(badgeX, displayW - badgeW - 8));

        // Badge Background with Dark Surface & Hairline Border
        ctx.save();
        ctx.shadowColor = "rgba(0, 0, 0, 0.85)";
        ctx.shadowBlur = 8;
        ctx.shadowOffsetY = 2;

        ctx.fillStyle = "#080808";
        ctx.fillRect(badgeX, badgeY, badgeW, badgeH);

        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1;
        ctx.strokeRect(badgeX, badgeY, badgeW, badgeH);
        ctx.restore();

        // Left Status Accent Strip
        ctx.fillStyle = strokeColor;
        ctx.fillRect(badgeX, badgeY, 3.5, badgeH);

        // Emoji Indicator
        ctx.font = '14px "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif';
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(emoji, badgeX + 16, badgeY + badgeH / 2);

        // Emotion Label Text
        ctx.font = 'bold 11px "JetBrains Mono", ui-monospace, SFMono-Regular, monospace';
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, badgeX + 30, badgeY + badgeH / 2);

        // Confidence Percentage Pill
        const pctBoxX = badgeX + badgeW - pctBoxWidth - 6;
        const pctBoxY = badgeY + 4;
        const pctBoxH = badgeH - 8;

        ctx.fillStyle = "#141414";
        ctx.fillRect(pctBoxX, pctBoxY, pctBoxWidth, pctBoxH);
        ctx.strokeStyle = toRgba(strokeColor, 0.4);
        ctx.lineWidth = 1;
        ctx.strokeRect(pctBoxX, pctBoxY, pctBoxWidth, pctBoxH);

        ctx.font = 'bold 10px "JetBrains Mono", ui-monospace, monospace';
        ctx.fillStyle = strokeColor;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(pctText, pctBoxX + pctBoxWidth / 2, pctBoxY + pctBoxH / 2);

        // Bottom Confidence Progress Bar (2px height)
        const barTrackY = badgeY + badgeH - 2;
        ctx.fillStyle = "#1c1c1c";
        ctx.fillRect(badgeX, barTrackY, badgeW, 2);

        const fillW = Math.max(2, Math.round(badgeW * Math.min(1, Math.max(0, box.confidence))));
        ctx.fillStyle = strokeColor;
        ctx.fillRect(badgeX, barTrackY, fillW, 2);

        // 6. Corner Telemetry Tag
        const tagY = Math.min(displayH - 8, y + h + 14);
        if (tagY > y + h + 4) {
          ctx.font = '9px "JetBrains Mono", ui-monospace, monospace';
          ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          ctx.fillText(`ID #0${box.face_id} · TRACK LOCKED`, x, tagY);
        }

        ctx.restore();
      });

      ctx.restore();
      animFrameIdRef.current = requestAnimationFrame(render);
    };

    animFrameIdRef.current = requestAnimationFrame(render);

    return () => {
      isRunning = false;
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-none w-full h-full ${className}`}
    />
  );
}

export default FaceOverlay;
