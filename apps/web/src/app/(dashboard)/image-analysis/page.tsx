"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  Upload,
  AlertCircle,
  Sparkles,
  CheckCircle2,
  Trash2,
  FolderOpen,
  ScanFace,
  Activity,
  Maximize2,
  Minimize2,
} from "lucide-react";
import { predictImage } from "@/lib/api/endpoints";
import { FacePrediction, PredictionResponse } from "@/types/prediction";
import { EMOTIONS, PredictionEmotion, SUPPORTED_EMOTIONS } from "@/types/emotion";

const MAX_FILE_SIZE_MB = 10;
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];

export default function ImageAnalysisPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [naturalDimensions, setNaturalDimensions] = useState<{ width: number; height: number }>({
    width: 0,
    height: 0,
  });
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [selectedFaceId, setSelectedFaceId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const imageElementRef = useRef<HTMLImageElement | null>(null);
  const imageContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  const toggleFullscreen = async () => {
    if (!imageContainerRef.current) return;
    try {
      if (!document.fullscreenElement) {
        await imageContainerRef.current.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch (err) {
      console.error("Fullscreen toggle error:", err);
    }
  };

  // Validate and stage file
  const handleFile = (file: File) => {
    setError(null);
    setResult(null);
    setSelectedFaceId(null);

    if (!ALLOWED_TYPES.includes(file.type)) {
      setError("Unsupported format. Please upload a JPEG, PNG, or WebP image.");
      return;
    }

    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setError(`File size exceeds limit (${MAX_FILE_SIZE_MB}MB). Please select a smaller image.`);
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setImagePreviewUrl(url);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const clearSelection = () => {
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl);
    }
    setSelectedFile(null);
    setImagePreviewUrl(null);
    setResult(null);
    setSelectedFaceId(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const executeInference = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const response = await predictImage(selectedFile, undefined, selectedFile.name);
      setResult(response);
      if (response.faces.length > 0) {
        setSelectedFaceId(response.faces[0].face_id);
      }
    } catch (err: unknown) {
      const errorObj = err as Error;
      setError(errorObj.message || "Failed to analyze image. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Currently inspected face
  const activeFace: FacePrediction | null =
    result && result.faces.length > 0
      ? result.faces.find((f) => f.face_id === selectedFaceId) || result.faces[0]
      : null;

  const activeMeta = activeFace
    ? EMOTIONS[activeFace.emotion.toLowerCase() as PredictionEmotion] || EMOTIONS.neutral
    : EMOTIONS.neutral;

  // Compute adaptive SVG overlay scale factor based on image resolution
  const viewScale =
    naturalDimensions.width > 0 ? Math.max(0.75, Math.min(3.5, naturalDimensions.width / 800)) : 1;

  return (
    <div className="space-y-6">
      {/* 1. Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-[#222222]">
        <div>
          <div className="font-mono text-xs uppercase tracking-[2px] text-[#888888] mb-1 flex items-center gap-2">
            <ScanFace className="w-3.5 h-3.5 text-emerald-400" />
            <span>IMAGE ANALYSIS</span>
            <span className="text-[#3a3a3a]">/</span>
            <span className="text-[#c3d9f3]">PHOTO INSPECTION</span>
          </div>
          <h1 className="font-display text-2xl sm:text-3xl uppercase tracking-[2px] text-white">
            Photo Emotion Recognition
          </h1>
          <p className="font-sans text-xs text-[#888888] mt-1">
            Upload photos or portraits to detect faces and analyze 7-class emotion probabilities with high-precision overlay.
          </p>
        </div>

        {/* Status Indicators */}
        <div className="flex items-center gap-2.5 self-start sm:self-auto">
          {result ? (
            <div className="flex items-center gap-2 px-3 py-1.5 border border-emerald-500/40 bg-emerald-950/20 font-mono text-[11px] uppercase tracking-[1.5px] text-emerald-400 rounded-none">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              <span>{result.faces_detected} FACE(S) DETECTED</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1.5 border border-[#262626] bg-[#0d0d0d] font-mono text-[11px] uppercase tracking-[1.5px] text-[#cccccc] rounded-none">
              <Activity className="w-3 h-3 text-[#c3d9f3]" />
              <span>STATUS: READY</span>
            </div>
          )}
        </div>
      </div>

      {/* 2. Error Banner */}
      {error && (
        <div className="p-3.5 bg-[#141414] border border-rose-500/40 text-rose-400 font-mono text-xs flex items-center justify-between rounded-none">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            type="button"
            onClick={() => setError(null)}
            className="text-[#999999] hover:text-white uppercase tracking-wider text-[11px] cursor-pointer"
          >
            DISMISS
          </button>
        </div>
      )}

      {/* 3. Main Interface Grid - Sized and Symmetrically Aligned */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
        {/* Left Column: Image Canvas / Dropzone & Single-Row Controls (8 cols on XL, 7 on LG) */}
        <div className="lg:col-span-7 xl:col-span-8 flex flex-col justify-between gap-3.5">
          {!imagePreviewUrl ? (
            /* Upload Dropzone with Cyber-Tactical Corners */
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
              className={`relative border-2 border-dashed p-10 text-center cursor-pointer transition-colors bg-[#080808] flex flex-col items-center justify-center min-h-[460px] flex-1 rounded-none select-none ${
                isDragging
                  ? "border-white bg-[#121212]"
                  : "border-[#262626] hover:border-[#404040] hover:bg-[#0d0d0d]"
              }`}
            >
              {/* Precision Corner Crosshairs */}
              <div className="absolute top-3 left-3 w-3 h-3 border-t border-l border-white/30 pointer-events-none" />
              <div className="absolute top-3 right-3 w-3 h-3 border-t border-r border-white/30 pointer-events-none" />
              <div className="absolute bottom-3 left-3 w-3 h-3 border-b border-l border-white/30 pointer-events-none" />
              <div className="absolute bottom-3 right-3 w-3 h-3 border-b border-r border-white/30 pointer-events-none" />

              <input
                ref={fileInputRef}
                type="file"
                accept={ALLOWED_TYPES.join(",")}
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFile(e.target.files[0]);
                  }
                }}
                className="hidden"
              />

              <div className="w-12 h-12 rounded-none border border-[#282828] bg-[#121212] flex items-center justify-center mb-3 text-[#cccccc]">
                <Upload className="w-5 h-5 text-white" />
              </div>

              <h3 className="font-display text-xl uppercase tracking-[2px] text-white mb-1.5">
                Upload Photo For Analysis
              </h3>
              <p className="font-sans text-xs text-[#777777] text-center max-w-sm mb-5 leading-relaxed">
                Drag & drop your photo here or browse from your device. Supports JPG, PNG, and WebP up to 10MB.
              </p>

              <button
                type="button"
                className="px-5 py-2.5 bg-white text-black font-mono text-xs font-semibold uppercase tracking-[1.5px] hover:bg-[#eaeaea] transition-colors cursor-pointer rounded-none pointer-events-none"
              >
                BROWSE FILES
              </button>
            </div>
          ) : (
            /* Image Preview Container with State-of-the-Art Overlay Frame */
            <div className="space-y-3 flex-1 flex flex-col justify-between">
              <div
                ref={imageContainerRef}
                className={`transition-all select-none ${
                  isFullscreen
                    ? "fixed inset-0 z-50 w-screen h-screen border-0 bg-black flex items-center justify-center p-6"
                    : "relative border border-[#262626] bg-black rounded-none flex items-center justify-center overflow-hidden min-h-[420px] flex-1"
                }`}
              >
                {/* Precision Corner Crosshairs */}
                <div className="absolute top-3 left-3 w-3 h-3 border-t border-l border-white/30 pointer-events-none z-20" />
                <div className="absolute top-3 right-3 w-3 h-3 border-t border-r border-white/30 pointer-events-none z-20" />
                <div className="absolute bottom-3 left-3 w-3 h-3 border-b border-l border-white/30 pointer-events-none z-20" />
                <div className="absolute bottom-3 right-3 w-3 h-3 border-b border-r border-white/30 pointer-events-none z-20" />

                {/* Fullscreen Toggle on Image HUD */}
                <div className="absolute top-3 right-3 z-30 pr-1">
                  <button
                    type="button"
                    onClick={toggleFullscreen}
                    className="p-1.5 bg-black/80 hover:bg-black text-[#888888] hover:text-white border border-[#2a2a2a] rounded-none transition-colors cursor-pointer flex items-center gap-1.5 font-mono text-[10px] uppercase"
                    title={isFullscreen ? "Exit Fullscreen (Esc)" : "Fullscreen View"}
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

                {/* Displayed Image and SVG Overlay */}
                <div
                  className={`relative inline-block max-w-full ${
                    isFullscreen ? "max-h-[88vh]" : "max-h-[560px]"
                  }`}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    ref={imageElementRef}
                    src={imagePreviewUrl}
                    alt="Uploaded face for emotion analysis"
                    onLoad={(e) => {
                      const img = e.currentTarget;
                      setNaturalDimensions({ width: img.naturalWidth, height: img.naturalHeight });
                    }}
                    className={`w-full h-auto object-contain mx-auto block select-none ${
                      isFullscreen ? "max-h-[88vh]" : "max-h-[560px]"
                    }`}
                  />

                  {/* High-Precision Cyber-Tactical SVG Overlay */}
                  {result && result.faces.length > 0 && naturalDimensions.width > 0 && (
                    <svg
                      className="absolute inset-0 w-full h-full pointer-events-none"
                      viewBox={`0 0 ${naturalDimensions.width} ${naturalDimensions.height}`}
                      preserveAspectRatio="xMidYMid meet"
                    >
                      <defs>
                        {result.faces.map((face) => {
                          const meta =
                            EMOTIONS[face.emotion.toLowerCase() as PredictionEmotion] ||
                            EMOTIONS.neutral;
                          return (
                            <filter
                              key={`glow-${face.face_id}`}
                              id={`glow-${face.face_id}`}
                              x="-20%"
                              y="-20%"
                              width="140%"
                              height="140%"
                            >
                              <feDropShadow
                                dx="0"
                                dy="0"
                                stdDeviation={4 * viewScale}
                                floodColor={meta.color}
                                floodOpacity="0.8"
                              />
                            </filter>
                          );
                        })}
                      </defs>

                      {result.faces.map((face) => {
                        const isSelected = face.face_id === selectedFaceId;
                        const meta =
                          EMOTIONS[face.emotion.toLowerCase() as PredictionEmotion] ||
                          EMOTIONS.neutral;
                        const boxColor = meta.color || "rgb(56, 189, 248)";

                        const { x, y, width: w, height: h } = face.bbox;
                        const bLen = Math.max(
                          16 * viewScale,
                          Math.min(36 * viewScale, w * 0.22)
                        );
                        const cx = x + w / 2;
                        const cy = y + h / 2;
                        const crossLen = 5 * viewScale;

                        // Badge layout dimensions
                        const badgeH = 28 * viewScale;
                        const fontSize = 11 * viewScale;
                        const emojiSize = 13 * viewScale;
                        const pctSize = 9.5 * viewScale;
                        const labelText = (meta.label || "NEUTRAL").toUpperCase();
                        const pctText = `${Math.round(face.confidence * 100)}%`;

                        const badgeW = Math.max(
                          150 * viewScale,
                          (labelText.length * 8 + 88) * viewScale
                        );

                        // Vertical badge positioning
                        let badgeY = y - badgeH - 6 * viewScale;
                        if (badgeY < 6 * viewScale) {
                          badgeY = y + 6 * viewScale;
                        }

                        // Horizontal clamping
                        const badgeX = Math.max(
                          4 * viewScale,
                          Math.min(x, naturalDimensions.width - badgeW - 4 * viewScale)
                        );

                        return (
                          <g
                            key={face.face_id}
                            className="pointer-events-auto cursor-pointer"
                            onClick={() => setSelectedFaceId(face.face_id)}
                          >
                            {/* 1. Holographic Ambient Tint */}
                            <rect
                              x={x}
                              y={y}
                              width={w}
                              height={h}
                              fill={boxColor}
                              fillOpacity={isSelected ? 0.12 : 0.06}
                            />

                            {/* 2. Hairline Dashed Perimeter */}
                            <rect
                              x={x}
                              y={y}
                              width={w}
                              height={h}
                              fill="none"
                              stroke={boxColor}
                              strokeWidth={1 * viewScale}
                              strokeDasharray={`${3 * viewScale}, ${4 * viewScale}`}
                              strokeOpacity={isSelected ? 0.8 : 0.35}
                            />

                            {/* 3. Tactical Sharp Corner Brackets with Neon Glow */}
                            <path
                              d={`
                                M ${x} ${y + bLen} L ${x} ${y} L ${x + bLen} ${y}
                                M ${x + w - bLen} ${y} L ${x + w} ${y} L ${x + w} ${y + bLen}
                                M ${x} ${y + h - bLen} L ${x} ${y + h} L ${x + bLen} ${y + h}
                                M ${x + w - bLen} ${y + h} L ${x + w} ${y + h} L ${x + w} ${y + h - bLen}
                              `}
                              stroke={boxColor}
                              strokeWidth={(isSelected ? 3.5 : 2.5) * viewScale}
                              strokeLinecap="square"
                              strokeLinejoin="miter"
                              fill="none"
                              filter={`url(#glow-${face.face_id})`}
                            />

                            {/* Corner Vertex Micro-Ticks (0px sharp accents) */}
                            <rect
                              x={x - 1 * viewScale}
                              y={y - 1 * viewScale}
                              width={3 * viewScale}
                              height={3 * viewScale}
                              fill={boxColor}
                            />
                            <rect
                              x={x + w - 2 * viewScale}
                              y={y - 1 * viewScale}
                              width={3 * viewScale}
                              height={3 * viewScale}
                              fill={boxColor}
                            />
                            <rect
                              x={x - 1 * viewScale}
                              y={y + h - 2 * viewScale}
                              width={3 * viewScale}
                              height={3 * viewScale}
                              fill={boxColor}
                            />
                            <rect
                              x={x + w - 2 * viewScale}
                              y={y + h - 2 * viewScale}
                              width={3 * viewScale}
                              height={3 * viewScale}
                              fill={boxColor}
                            />

                            {/* 4. Centroid Precision Crosshair */}
                            <line
                              x1={cx - crossLen}
                              y1={cy}
                              x2={cx + crossLen}
                              y2={cy}
                              stroke={boxColor}
                              strokeWidth={1.5 * viewScale}
                              strokeOpacity={0.7}
                            />
                            <line
                              x1={cx}
                              y1={cy - crossLen}
                              x2={cx}
                              y2={cy + crossLen}
                              stroke={boxColor}
                              strokeWidth={1.5 * viewScale}
                              strokeOpacity={0.7}
                            />
                            <rect
                              x={cx - 1 * viewScale}
                              y={cy - 1 * viewScale}
                              width={2 * viewScale}
                              height={2 * viewScale}
                              fill={boxColor}
                            />

                            {/* 5. Floating Obsidian Emotion HUD Badge */}
                            <g>
                              {/* Badge Surface */}
                              <rect
                                x={badgeX}
                                y={badgeY}
                                width={badgeW}
                                height={badgeH}
                                fill="#080808"
                                stroke={boxColor}
                                strokeWidth={1 * viewScale}
                              />

                              {/* Left Accent Strip */}
                              <rect
                                x={badgeX}
                                y={badgeY}
                                width={3.5 * viewScale}
                                height={badgeH}
                                fill={boxColor}
                              />

                              {/* Canonical Emotion Emoji */}
                              <text
                                x={badgeX + 16 * viewScale}
                                y={badgeY + badgeH / 2 + 0.5 * viewScale}
                                fontSize={emojiSize}
                                textAnchor="middle"
                                dominantBaseline="central"
                              >
                                {meta.emoji}
                              </text>

                              {/* Emotion Label */}
                              <text
                                x={badgeX + 28 * viewScale}
                                y={badgeY + badgeH / 2 + 0.5 * viewScale}
                                fill="#ffffff"
                                fontSize={fontSize}
                                fontFamily="monospace"
                                fontWeight="bold"
                                dominantBaseline="central"
                              >
                                {labelText}
                              </text>

                              {/* Confidence Pill */}
                              <rect
                                x={badgeX + badgeW - 46 * viewScale}
                                y={badgeY + 4 * viewScale}
                                width={40 * viewScale}
                                height={badgeH - 8 * viewScale}
                                fill="#141414"
                                stroke={boxColor}
                                strokeWidth={1 * viewScale}
                                strokeOpacity={0.4}
                              />
                              <text
                                x={badgeX + badgeW - 26 * viewScale}
                                y={badgeY + badgeH / 2 + 0.5 * viewScale}
                                fill={boxColor}
                                fontSize={pctSize}
                                fontFamily="monospace"
                                fontWeight="bold"
                                textAnchor="middle"
                                dominantBaseline="central"
                              >
                                {pctText}
                              </text>

                              {/* Bottom Probability Meter (2px) */}
                              <rect
                                x={badgeX}
                                y={badgeY + badgeH - 2 * viewScale}
                                width={badgeW}
                                height={2 * viewScale}
                                fill="#1c1c1c"
                              />
                              <rect
                                x={badgeX}
                                y={badgeY + badgeH - 2 * viewScale}
                                width={badgeW * Math.min(1, Math.max(0, face.confidence))}
                                height={2 * viewScale}
                                fill={boxColor}
                              />
                            </g>

                            {/* 6. Corner Telemetry Tag */}
                            <text
                              x={x}
                              y={y + h + 14 * viewScale}
                              fill="rgba(255,255,255,0.45)"
                              fontSize={9 * viewScale}
                              fontFamily="monospace"
                            >
                              ID #0{face.face_id} · DETECTED
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  )}
                </div>
              </div>

              {/* Single-Row Action Controls Bar */}
              <div className="flex flex-wrap items-center justify-between gap-2.5 p-3 bg-[#0c0c0c] border border-[#222222] rounded-none">
                <div className="flex items-center gap-2 font-mono text-xs text-[#888888]">
                  <span className="text-white truncate max-w-[180px] font-medium">
                    {selectedFile?.name}
                  </span>
                  {naturalDimensions.width > 0 && (
                    <>
                      <span className="text-[#444444]">·</span>
                      <span className="text-[#777777]">
                        {naturalDimensions.width}×{naturalDimensions.height} PX
                      </span>
                    </>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={clearSelection}
                    disabled={isAnalyzing}
                    className="px-3 py-2 bg-[#121212] hover:bg-[#1a1a1a] text-[#888888] hover:text-white border border-[#2a2a2a] rounded-none font-mono text-[11px] uppercase tracking-[1px] transition-colors cursor-pointer flex items-center gap-1.5 disabled:opacity-40"
                    title="Remove selected image"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>CLEAR</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isAnalyzing}
                    className="px-3 py-2 bg-[#121212] hover:bg-[#1a1a1a] text-[#cccccc] hover:text-white border border-[#2a2a2a] rounded-none font-mono text-[11px] uppercase tracking-[1px] transition-colors cursor-pointer flex items-center gap-1.5 disabled:opacity-40"
                    title="Upload different image"
                  >
                    <FolderOpen className="w-3.5 h-3.5" />
                    <span>CHOOSE ANOTHER</span>
                  </button>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept={ALLOWED_TYPES.join(",")}
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        handleFile(e.target.files[0]);
                      }
                    }}
                    className="hidden"
                  />

                  <button
                    type="button"
                    onClick={toggleFullscreen}
                    className="px-3 py-2 bg-[#121212] hover:bg-[#1a1a1a] text-[#cccccc] hover:text-white border border-[#2a2a2a] rounded-none font-mono text-[11px] uppercase tracking-[1px] transition-colors cursor-pointer flex items-center gap-1.5"
                    title={isFullscreen ? "Exit Fullscreen (Esc)" : "Fullscreen View"}
                  >
                    {isFullscreen ? (
                      <>
                        <Minimize2 className="w-3.5 h-3.5" />
                        <span>EXIT</span>
                      </>
                    ) : (
                      <>
                        <Maximize2 className="w-3.5 h-3.5" />
                        <span>FULLSCREEN</span>
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={executeInference}
                    disabled={isAnalyzing}
                    className="px-4 py-2 bg-white hover:bg-[#eaeaea] text-black font-mono text-[11px] font-semibold uppercase tracking-[1px] rounded-none transition-colors cursor-pointer flex items-center gap-2 disabled:opacity-40 shadow-sm"
                  >
                    {isAnalyzing ? (
                      <>
                        <span className="inline-block w-3 h-3 border-2 border-black border-t-transparent animate-spin rounded-none" />
                        <span>ANALYZING...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>{result ? "RE-ANALYZE" : "ANALYZE EMOTIONS"}</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Multiple Faces Selector Strip - Zero Scrollbar, Clean Wrapping */}
              {result && result.faces.length > 1 && (
                <div className="p-3 bg-[#0c0c0c] border border-[#222222] rounded-none flex flex-wrap items-center gap-2">
                  <span className="font-mono text-[10px] uppercase tracking-[1.5px] text-[#777777] shrink-0 pl-1 mr-1">
                    FACES ({result.faces.length}):
                  </span>
                  {result.faces.map((f) => {
                    const isSelected = f.face_id === selectedFaceId;
                    const fMeta =
                      EMOTIONS[f.emotion.toLowerCase() as PredictionEmotion] || EMOTIONS.neutral;

                    return (
                      <button
                        key={f.face_id}
                        type="button"
                        onClick={() => setSelectedFaceId(f.face_id)}
                        className={`px-3 py-1.5 font-mono text-[11px] uppercase tracking-[1px] flex items-center gap-2 border transition-all cursor-pointer rounded-none ${
                          isSelected
                            ? "bg-[#181818] text-white border-white shadow-sm"
                            : "bg-[#111111] text-[#888888] border-[#222222] hover:text-white hover:border-[#333333]"
                        }`}
                      >
                        <span>{fMeta.emoji}</span>
                        <span>FACE #{f.face_id}</span>
                        <span className="font-bold" style={{ color: fMeta.color }}>
                          {Math.round(f.confidence * 100)}%
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Emotion Analysis Box - Symmetrically Aligned Height (4 cols on XL, 5 on LG) */}
        <div className="lg:col-span-5 xl:col-span-4 flex flex-col">
          <div className="bg-[#0c0c0c] border border-[#222222] rounded-none p-5 h-full flex flex-col justify-between space-y-4">
            {/* 1. Header Bar */}
            <div className="flex items-center justify-between pb-3 border-b border-[#1f1f1f]">
              <div className="flex items-center gap-2">
                <span
                  className={`w-2 h-2 rounded-none ${
                    result && activeFace
                      ? "bg-emerald-400 animate-pulse shadow-[0_0_6px_rgba(52,211,153,0.6)]"
                      : "bg-[#444444]"
                  }`}
                />
                <span className="font-mono text-xs uppercase tracking-[2px] text-white">
                  Emotion Spectrum
                </span>
              </div>
              {result && activeFace ? (
                <span className="font-mono text-[10px] text-[#c3d9f3] tracking-[1.5px] px-2 py-0.5 bg-[#141414] border border-[#262626] rounded-none">
                  FACE #{activeFace.face_id}
                </span>
              ) : (
                <span className="font-mono text-[10px] text-[#666666] tracking-[1px] uppercase">
                  {isAnalyzing ? "ANALYZING..." : "STANDBY"}
                </span>
              )}
            </div>

            {/* 2. Dominant Emotion Hero Card */}
            {result && activeFace ? (
              <div
                className="p-4 bg-[#121212] border flex items-center justify-between relative overflow-hidden rounded-none"
                style={{ borderColor: activeMeta.color }}
              >
                {/* Subtle ambient emotion glow */}
                <div
                  className="absolute -right-8 -top-8 w-24 h-24 blur-2xl opacity-15 pointer-events-none"
                  style={{ backgroundColor: activeMeta.color }}
                />

                <div className="flex items-center gap-3.5">
                  <div
                    className="w-12 h-12 rounded-none flex items-center justify-center text-3xl select-none"
                    style={{
                      backgroundColor: `${activeMeta.color}15`,
                      border: `1px solid ${activeMeta.color}35`,
                    }}
                    role="img"
                    aria-label={activeMeta.label}
                  >
                    {activeMeta.emoji}
                  </div>
                  <div>
                    <span className="font-mono text-[9px] uppercase tracking-[2px] text-[#777777] block">
                      Dominant Emotion
                    </span>
                    <div
                      className="font-display text-2xl uppercase tracking-[1.5px] font-normal mt-0.5"
                      style={{ color: activeMeta.color }}
                    >
                      {activeMeta.label}
                    </div>
                    <div className="font-sans text-[11px] text-[#888888] mt-0.5">
                      {activeMeta.description}
                    </div>
                  </div>
                </div>

                <div className="text-right pl-3">
                  <div className="font-mono text-2xl text-white">
                    {Math.round(activeFace.confidence * 100)}%
                  </div>
                  <div className="font-mono text-[9px] uppercase tracking-[1px] text-[#777777] mt-0.5">
                    Confidence
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-4 bg-[#121212] border border-[#1f1f1f] rounded-none flex items-center gap-3.5">
                <div className="w-12 h-12 bg-[#181818] border border-[#262626] rounded-none flex items-center justify-center text-2xl select-none">
                  {isAnalyzing ? "⚡" : "🖼️"}
                </div>
                <div>
                  <div className="font-display text-lg uppercase tracking-[1px] text-white">
                    {isAnalyzing ? "Analyzing Facial Expression..." : "Photo Analysis Ready"}
                  </div>
                  <div className="font-sans text-xs text-[#777777] mt-0.5">
                    {isAnalyzing
                      ? "Classifying facial affect across 7 emotional states"
                      : "Upload a photo and click Analyze Emotions to inspect results"}
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

              {SUPPORTED_EMOTIONS.map((emotionKey) => {
                const meta = EMOTIONS[emotionKey as PredictionEmotion] || EMOTIONS.neutral;
                const prob = activeFace ? activeFace.probabilities[emotionKey] || 0.0 : 0.0;
                const pct = Math.round(prob * 100);
                const isDominant = activeFace ? emotionKey === activeFace.emotion : false;

                return (
                  <div key={emotionKey} className="space-y-1">
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
                        {activeFace ? `${pct}%` : "—"}
                      </span>
                    </div>

                    {/* Progress Bar with 0px Sharp Corners */}
                    <div className="w-full h-1 bg-[#181818] overflow-hidden rounded-none">
                      <div
                        className="h-full transition-all duration-200"
                        style={{
                          width: activeFace ? `${Math.max(pct, 1)}%` : "0%",
                          backgroundColor: meta.color,
                          opacity: isDominant ? 1 : 0.35,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* 4. Integrated Telemetry Strip at Bottom (Zero-Overflow Layout) */}
            <div className="grid grid-cols-4 gap-1.5 pt-3 border-t border-[#1f1f1f]">
              <div className="p-1.5 bg-[#111111] border border-[#1c1c1c] text-center rounded-none overflow-hidden">
                <div className="font-mono text-[8.5px] text-[#666666] uppercase tracking-[0.5px]">
                  INFERENCE
                </div>
                <div
                  className="font-mono text-[10.5px] text-white mt-0.5 whitespace-nowrap truncate"
                  title={`${result ? (result.timing?.total_ms ?? 0).toFixed(0) : "0"} MS`}
                >
                  {result ? (result.timing?.total_ms ?? 0).toFixed(0) : "0"}{" "}
                  <span className="text-[8.5px] text-[#777777]">MS</span>
                </div>
              </div>

              <div className="p-1.5 bg-[#111111] border border-[#1c1c1c] text-center rounded-none overflow-hidden">
                <div className="font-mono text-[8.5px] text-[#666666] uppercase tracking-[0.5px]">
                  FACES
                </div>
                <div className="font-mono text-[10.5px] text-[#c3d9f3] mt-0.5 whitespace-nowrap truncate">
                  {result ? result.faces_detected : 0}
                </div>
              </div>

              <div className="p-1.5 bg-[#111111] border border-[#1c1c1c] text-center rounded-none overflow-hidden">
                <div className="font-mono text-[8.5px] text-[#666666] uppercase tracking-[0.5px]">
                  SCORE
                </div>
                <div className="font-mono text-[10.5px] text-white mt-0.5 whitespace-nowrap truncate">
                  {activeFace ? `${(activeFace.detection_confidence * 100).toFixed(0)}%` : "—"}
                </div>
              </div>

              <div className="p-1.5 bg-[#111111] border border-[#1c1c1c] text-center rounded-none overflow-hidden">
                <div className="font-mono text-[8.5px] text-[#666666] uppercase tracking-[0.5px]">
                  SIZE
                </div>
                <div
                  className="font-mono text-[10.5px] text-white mt-0.5 whitespace-nowrap truncate"
                  title={
                    activeFace ? `${activeFace.bbox.width}×${activeFace.bbox.height} px` : undefined
                  }
                >
                  {activeFace ? `${activeFace.bbox.width}×${activeFace.bbox.height}` : "—"}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
