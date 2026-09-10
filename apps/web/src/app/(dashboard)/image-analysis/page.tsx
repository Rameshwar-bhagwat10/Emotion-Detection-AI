"use client";

import React, { useRef, useState } from "react";
import {
  AlertCircle,
  Clock,
  Cpu,
  FileImage,
  HelpCircle,
  RefreshCw,
  Sparkles,
  Upload,
  User,
  Users,
  X,
} from "lucide-react";
import { predictImage } from "@/lib/api/endpoints";
import { FacePrediction, PredictionResponse } from "@/types/prediction";
import { EMOTIONS, SUPPORTED_EMOTIONS } from "@/types/emotion";

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

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const imageElementRef = useRef<HTMLImageElement | null>(null);

  // Validate and stage file
  const handleFile = (file: File) => {
    setError(null);
    setResult(null);
    setSelectedFaceId(null);

    if (!ALLOWED_TYPES.includes(file.type)) {
      setError("Unsupported format. Please upload a JPEG, PNG, or WEBP image.");
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

  // Currently inspected face (defaults to face 1 or first in list)
  const activeFace: FacePrediction | null =
    result && result.faces.length > 0
      ? result.faces.find((f) => f.face_id === selectedFaceId) || result.faces[0]
      : null;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-6">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl md:text-3xl font-extrabold text-white tracking-tight">
              Image Facial Expression Analysis
            </h1>
            <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
              YuNet + ResNet-18
            </span>
          </div>
          <p className="text-sm text-zinc-400 mt-1">
            Detect facial boundaries and classify expressions across 7 discrete categories with deep confidence distributions.
          </p>
        </div>

        {selectedFile && (
          <button
            onClick={clearSelection}
            className="self-start sm:self-auto flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-xs font-medium text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition-colors cursor-pointer"
          >
            <X className="w-3.5 h-3.5" />
            <span>Reset / Upload Another</span>
          </button>
        )}
      </div>

      {/* Error Banner */}
      {error && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-rose-500/10 border border-rose-500/25 text-rose-400 text-sm">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <div className="flex-1">
            <strong className="font-semibold">Analysis Error: </strong>
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Main Content Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left: Upload & Preview Area (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          {!imagePreviewUrl ? (
            /* Upload Dropzone */
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
              className={`relative flex flex-col items-center justify-center min-h-[420px] p-8 rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-200 ${
                isDragging
                  ? "border-indigo-500 bg-indigo-500/10 scale-[0.99]"
                  : "border-zinc-800 hover:border-zinc-700 bg-zinc-900/40 hover:bg-zinc-900/70"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept={ALLOWED_TYPES.join(",")}
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) {
                    handleFile(e.target.files[0]);
                  }
                }}
                className="hidden"
              />

              <div className="w-16 h-16 mb-4 rounded-2xl bg-zinc-800/80 border border-zinc-700/60 flex items-center justify-center shadow-lg">
                <Upload className="w-8 h-8 text-indigo-400" />
              </div>

              <h3 className="text-base font-semibold text-zinc-200 mb-1">
                Select or drag a photo here
              </h3>
              <p className="text-xs text-zinc-400 text-center max-w-sm mb-4">
                Supports JPG, PNG, and WEBP photographs up to 10MB. Clear front-facing facial expressions produce highest confidence.
              </p>

              <button
                type="button"
                className="px-4 py-2 text-xs font-semibold rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white shadow-md transition-colors pointer-events-none"
              >
                Browse Files
              </button>
            </div>
          ) : (
            /* Image Preview with Bounding Box Overlays */
            <div className="relative overflow-hidden rounded-2xl bg-zinc-950 border border-zinc-800 shadow-2xl">
              <div className="relative inline-block w-full">
                <img
                  ref={imageElementRef}
                  src={imagePreviewUrl}
                  alt="Uploaded face for emotion analysis"
                  onLoad={(e) => {
                    const img = e.currentTarget;
                    setNaturalDimensions({ width: img.naturalWidth, height: img.naturalHeight });
                  }}
                  className="w-full h-auto max-h-[550px] object-contain mx-auto block"
                />

                {/* Face Bounding Box SVG Overlay */}
                {result && result.faces.length > 0 && naturalDimensions.width > 0 && (
                  <svg
                    className="absolute inset-0 w-full h-full pointer-events-none"
                    viewBox={`0 0 ${naturalDimensions.width} ${naturalDimensions.height}`}
                    preserveAspectRatio="xMidYMid meet"
                  >
                    {result.faces.map((face) => {
                      const isSelected = face.face_id === selectedFaceId;
                      const emotionMeta = EMOTIONS[face.emotion] || EMOTIONS.uncertain;

                      return (
                        <g key={face.face_id} className="pointer-events-auto cursor-pointer">
                          {/* Face Box */}
                          <rect
                            x={face.bbox.x}
                            y={face.bbox.y}
                            width={face.bbox.width}
                            height={face.bbox.height}
                            fill="none"
                            stroke={isSelected ? emotionMeta.color : "rgba(255, 255, 255, 0.7)"}
                            strokeWidth={isSelected ? "4" : "2"}
                            strokeDasharray={face.is_uncertain ? "6 3" : undefined}
                            className="transition-all"
                            onClick={() => setSelectedFaceId(face.face_id)}
                          />

                          {/* Label Pill */}
                          <g
                            transform={`translate(${face.bbox.x}, ${Math.max(
                              26,
                              face.bbox.y - 8
                            )})`}
                            onClick={() => setSelectedFaceId(face.face_id)}
                          >
                            <rect
                              x="0"
                              y="-22"
                              width={Math.max(120, face.emotion.length * 12 + 50)}
                              height="24"
                              rx="6"
                              fill="rgba(24, 24, 27, 0.9)"
                              stroke={emotionMeta.color}
                              strokeWidth="1.5"
                            />
                            <text
                              x="8"
                              y="-6"
                              fill="#f4f4f5"
                              fontSize="13"
                              fontWeight="600"
                              fontFamily="sans-serif"
                            >
                              {emotionMeta.emoji} {emotionMeta.label}{" "}
                              {Math.round(face.confidence * 100)}%
                            </text>
                          </g>
                        </g>
                      );
                    })}
                  </svg>
                )}
              </div>

              {/* Action Toolbar Below Image */}
              <div className="flex items-center justify-between p-4 bg-zinc-900/90 border-t border-zinc-800">
                <div className="flex items-center gap-2 text-xs text-zinc-400">
                  <FileImage className="w-4 h-4 text-zinc-500" />
                  <span className="font-medium text-zinc-300 truncate max-w-[200px]">
                    {selectedFile?.name}
                  </span>
                  {naturalDimensions.width > 0 && (
                    <span className="text-zinc-600 font-mono">
                      ({naturalDimensions.width}×{naturalDimensions.height}px)
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isAnalyzing}
                    className="px-3 py-1.5 text-xs font-semibold text-zinc-300 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors cursor-pointer disabled:opacity-50"
                  >
                    Change Photo
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept={ALLOWED_TYPES.join(",")}
                    onChange={(e) => {
                      if (e.target.files && e.target.files.length > 0) {
                        handleFile(e.target.files[0]);
                      }
                    }}
                    className="hidden"
                  />

                  <button
                    onClick={executeInference}
                    disabled={isAnalyzing}
                    className="flex items-center gap-2 px-5 py-2 text-xs font-semibold text-white bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 rounded-lg shadow-md transition-all cursor-pointer disabled:opacity-50"
                  >
                    {isAnalyzing ? (
                      <>
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        <span>Analyzing Expression...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>{result ? "Re-Analyze" : "Analyze Expression"}</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Multiple Faces Selector Tabs (if > 1 face) */}
          {result && result.faces.length > 1 && (
            <div className="p-4 rounded-xl bg-zinc-900/80 border border-zinc-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                  Select Detected Face ({result.faces_detected} detected)
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {result.faces.map((f) => {
                  const meta = EMOTIONS[f.emotion] || EMOTIONS.uncertain;
                  const isSelected = f.face_id === selectedFaceId;

                  return (
                    <button
                      key={f.face_id}
                      onClick={() => setSelectedFaceId(f.face_id)}
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all cursor-pointer ${
                        isSelected
                          ? "bg-indigo-600/20 border-indigo-500 text-white shadow-sm"
                          : "bg-zinc-950 border-zinc-800 text-zinc-400 hover:text-zinc-200"
                      }`}
                    >
                      <User className="w-3.5 h-3.5" />
                      <span>Face #{f.face_id}</span>
                      <span className="text-[11px] opacity-75">
                        ({meta.label} {Math.round(f.confidence * 100)}%)
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Right: Results & Confidence Distribution (5 Cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* State 1: Awaiting Analysis */}
          {!result && !isAnalyzing && (
            <div className="p-8 rounded-2xl bg-zinc-900/60 border border-zinc-800 text-center space-y-3">
              <div className="w-12 h-12 mx-auto rounded-xl bg-zinc-800/80 flex items-center justify-center text-zinc-500">
                <Sparkles className="w-6 h-6" />
              </div>
              <h4 className="text-base font-semibold text-zinc-200">No Analysis Results Yet</h4>
              <p className="text-xs text-zinc-400 max-w-xs mx-auto leading-relaxed">
                Upload a photograph on the left and click{" "}
                <strong className="text-zinc-200 font-semibold">Analyze Expression</strong> to execute
                neural facial detection and view probability metrics.
              </p>
            </div>
          )}

          {/* State 2: Loading Analysis */}
          {isAnalyzing && (
            <div className="p-8 rounded-2xl bg-zinc-900/60 border border-zinc-800 text-center space-y-4">
              <div className="relative w-14 h-14 mx-auto flex items-center justify-center">
                <div className="absolute inset-0 rounded-full border-2 border-indigo-500/20 border-t-indigo-500 animate-spin" />
                <Cpu className="w-6 h-6 text-indigo-400" />
              </div>
              <div className="space-y-1">
                <h4 className="text-base font-semibold text-zinc-200">Analyzing Facial Expression</h4>
                <p className="text-xs text-zinc-400">
                  Executing YuNet face detector and Phase 09 ResNet-18 neural classifier...
                </p>
              </div>
            </div>
          )}

          {/* State 3: No Face Detected */}
          {result && result.faces_detected === 0 && (
            <div className="p-8 rounded-2xl bg-zinc-900/60 border border-amber-500/20 text-center space-y-3">
              <div className="w-12 h-12 mx-auto rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
                <Users className="w-6 h-6" />
              </div>
              <h4 className="text-base font-bold text-zinc-100">No Face Detected</h4>
              <p className="text-xs text-zinc-400 max-w-sm mx-auto leading-relaxed">
                The YuNet detector could not locate any clear human faces in this image. Please upload a clear photo with the subject facing the camera under good lighting.
              </p>
            </div>
          )}

          {/* State 4: Prediction Results Available */}
          {result && activeFace && (
            <div className="space-y-6">
              {/* Primary Prediction Hero Card */}
              <div className="p-6 rounded-2xl bg-zinc-900/90 border border-zinc-800 shadow-xl space-y-4">
                <div className="flex items-center justify-between text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                  <span>Predicted Facial Expression</span>
                  <span className="font-mono text-zinc-500">Face #{activeFace.face_id}</span>
                </div>

                {/* Dominant Emotion Banner */}
                {(() => {
                  const meta = EMOTIONS[activeFace.emotion] || EMOTIONS.uncertain;
                  const confidencePct = (activeFace.confidence * 100).toFixed(1);

                  return (
                    <div className={`p-4 rounded-xl border ${meta.bgLight} ${meta.borderColor} space-y-2`}>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className="text-4xl">{meta.emoji}</span>
                          <div>
                            <span className="text-xs text-zinc-400 font-medium uppercase tracking-wider block">
                              Dominant Class
                            </span>
                            <h3 className={`text-2xl font-extrabold ${meta.textColor}`}>
                              {meta.label}
                            </h3>
                          </div>
                        </div>

                        <div className="text-right">
                          <span className="text-xs text-zinc-400 font-medium uppercase tracking-wider block">
                            Model Confidence
                          </span>
                          <span className="text-2xl font-black text-white">{confidencePct}%</span>
                        </div>
                      </div>

                      {/* Uncertainty Alert if active */}
                      {activeFace.is_uncertain && (
                        <div className="flex items-center gap-2 pt-2 border-t border-zinc-700/50 text-xs text-amber-300">
                          <HelpCircle className="w-4 h-4 flex-shrink-0" />
                          <span>
                            Top prediction confidence is below the calibrated certainty threshold (40%).
                          </span>
                        </div>
                      )}
                    </div>
                  );
                })()}

                {/* Face Detector Bounding Metadata */}
                <div className="grid grid-cols-2 gap-3 pt-2 text-xs">
                  <div className="p-3 rounded-lg bg-zinc-950/60 border border-zinc-800/80">
                    <span className="text-zinc-500 block">Detector Confidence</span>
                    <span className="font-semibold text-zinc-200">
                      {(activeFace.detection_confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="p-3 rounded-lg bg-zinc-950/60 border border-zinc-800/80">
                    <span className="text-zinc-500 block">Bounding Box</span>
                    <span className="font-mono text-zinc-200">
                      {activeFace.bbox.width}×{activeFace.bbox.height}px
                    </span>
                  </div>
                </div>
              </div>

              {/* Softmax Probability Distribution */}
              <div className="p-6 rounded-2xl bg-zinc-900/90 border border-zinc-800 shadow-xl space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">
                    Full Probability Distribution
                  </h4>
                  <span className="text-[11px] text-zinc-500 font-mono">Softmax [0–100%]</span>
                </div>

                <div className="space-y-2.5">
                  {SUPPORTED_EMOTIONS.map((emotionKey) => {
                    const prob = activeFace.probabilities[emotionKey] || 0.0;
                    const pct = (prob * 100).toFixed(1);
                    const meta = EMOTIONS[emotionKey];
                    const isTop = activeFace.emotion === emotionKey;

                    return (
                      <div key={emotionKey} className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="flex items-center gap-2 font-medium text-zinc-300">
                            <span>{meta.emoji}</span>
                            <span className={isTop ? "font-bold text-white" : ""}>
                              {meta.label}
                            </span>
                          </span>
                          <span className={`font-mono text-xs ${isTop ? "font-bold text-white" : "text-zinc-400"}`}>
                            {pct}%
                          </span>
                        </div>

                        {/* Progress Bar */}
                        <div className="h-2 w-full rounded-full bg-zinc-950 overflow-hidden border border-zinc-800/60">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${Math.max(1, parseFloat(pct))}%`,
                              backgroundColor: meta.color,
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Latency & Telemetry Breakdown Card */}
              {result.timing && (
                <div className="p-5 rounded-2xl bg-zinc-900/60 border border-zinc-800 space-y-3">
                  <div className="flex items-center justify-between text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                    <span className="flex items-center gap-1.5">
                      <Clock className="w-3.5 h-3.5 text-zinc-500" />
                      <span>Pipeline Latency Telemetry</span>
                    </span>
                    <span className="font-mono text-indigo-400 font-bold">
                      {result.timing.total_ms.toFixed(1)} ms total
                    </span>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
                    <div className="p-2 rounded-lg bg-zinc-950/60 border border-zinc-800/60">
                      <span className="text-[10px] text-zinc-500 block">Detection</span>
                      <span className="font-mono font-medium text-zinc-300">
                        {result.timing.face_detection_ms.toFixed(1)}ms
                      </span>
                    </div>
                    <div className="p-2 rounded-lg bg-zinc-950/60 border border-zinc-800/60">
                      <span className="text-[10px] text-zinc-500 block">Preprocess</span>
                      <span className="font-mono font-medium text-zinc-300">
                        {result.timing.preprocessing_ms.toFixed(1)}ms
                      </span>
                    </div>
                    <div className="p-2 rounded-lg bg-zinc-950/60 border border-zinc-800/60">
                      <span className="text-[10px] text-zinc-500 block">Neural Net</span>
                      <span className="font-mono font-medium text-zinc-300">
                        {result.timing.inference_ms.toFixed(1)}ms
                      </span>
                    </div>
                    <div className="p-2 rounded-lg bg-zinc-950/60 border border-zinc-800/60">
                      <span className="text-[10px] text-zinc-500 block">Postprocess</span>
                      <span className="font-mono font-medium text-zinc-300">
                        {result.timing.postprocessing_ms.toFixed(1)}ms
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
