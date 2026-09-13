"use client";

import React, { useState, useRef, useCallback } from "react";
import { Upload, Film, AlertCircle, FileVideo, CheckCircle2 } from "lucide-react";

export interface VideoUploaderProps {
  onUpload: (file: File, samplingFps: number, sessionName?: string) => Promise<void>;
  isUploading?: boolean;
  className?: string;
}

const SUPPORTED_EXTENSIONS = [".mp4", ".webm", ".mov", ".avi", ".mkv"];
const MAX_FILE_SIZE_MB = 100;

export function VideoUploader({ onUpload, isUploading = false, className = "" }: VideoUploaderProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [samplingFps, setSamplingFps] = useState<number>(5.0);
  const [sessionName, setSessionName] = useState<string>("");
  const [dragOver, setDragOver] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndSetFile = useCallback((file: File) => {
    setErrorMessage(null);
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(ext)) {
      setErrorMessage(`Unsupported format '${ext}'. Allowed: ${SUPPORTED_EXTENSIONS.join(", ")}`);
      return;
    }

    const sizeMb = file.size / (1024 * 1024);
    if (sizeMb > MAX_FILE_SIZE_MB) {
      setErrorMessage(`File size (${sizeMb.toFixed(1)} MB) exceeds maximum limit of ${MAX_FILE_SIZE_MB} MB.`);
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    if (!sessionName) {
      setSessionName(file.name.replace(/\.[^/.]+$/, ""));
    }
  }, [previewUrl, sessionName]);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        validateAndSetFile(e.dataTransfer.files[0]);
      }
    },
    [validateAndSetFile]
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile || isUploading) return;
    try {
      await onUpload(selectedFile, samplingFps, sessionName.trim() || undefined);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to initiate video upload.";
      setErrorMessage(msg);
    }
  };

  return (
    <div className={`p-6 bg-[#0d0d0d] border border-[#262626] rounded-none ${className}`}>
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-6 pb-4 border-b border-[#262626]">
        <div>
          <div className="font-mono text-xs uppercase tracking-[2px] text-[#999999] mb-1 flex items-center gap-2">
            <Film className="w-3.5 h-3.5 text-[#c3d9f3]" />
            <span>VIDEO UPLOAD & ANALYSIS</span>
          </div>
          <h2 className="font-display text-2xl uppercase tracking-[2px] text-white">
            UPLOAD VIDEO FOR EMOTION ANALYSIS
          </h2>
          <p className="font-sans text-xs text-[#999999] mt-1">
            Analyze facial expressions, track multiple people simultaneously, and generate a chronological emotion timeline.
          </p>
        </div>
        <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[1.5px] text-emerald-400 bg-[#141414] px-3 py-1.5 border border-[#262626] rounded-none">
          <span className="w-2 h-2 rounded-none bg-emerald-400" />
          <span>READY TO ANALYZE</span>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Dropzone with Tactical Corner Crosshairs */}
        {!selectedFile ? (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`relative border border-dashed p-10 text-center cursor-pointer transition-all rounded-none ${
              dragOver
                ? "border-[#c3d9f3] bg-[#141414]"
                : "border-[#3a3a3a] hover:border-white bg-[#000000] hover:bg-[#141414]"
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
              accept={SUPPORTED_EXTENSIONS.join(",")}
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="flex flex-col items-center justify-center space-y-3">
              <div className="w-12 h-12 border border-[#3a3a3a] bg-[#141414] flex items-center justify-center text-white rounded-none">
                <Upload className="w-5 h-5 text-[#c3d9f3]" />
              </div>
              <p className="font-display text-lg uppercase tracking-[2px] text-white">
                DRAG & DROP VIDEO HERE OR CLICK TO BROWSE
              </p>
              <p className="font-sans text-xs text-[#888888] max-w-sm">
                Supports MP4, WebM, MOV, and AVI formats up to {MAX_FILE_SIZE_MB}MB.
              </p>
              <div className="pt-2">
                <span className="px-4 py-2 bg-white text-black font-mono text-xs font-semibold uppercase tracking-[1.5px] rounded-none pointer-events-none">
                  BROWSE FILES
                </span>
              </div>
            </div>
          </div>
        ) : (
          /* Selected File Preview */
          <div className="bg-[#141414] border border-[#262626] p-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 rounded-none">
            <div className="flex items-center gap-4 min-w-0">
              {previewUrl && (
                <div className="relative w-28 h-18 bg-black shrink-0 border border-[#262626] overflow-hidden rounded-none">
                  <video src={previewUrl} className="w-full h-full object-cover" muted />
                </div>
              )}
              <div className="min-w-0 font-mono">
                <div className="flex items-center gap-2">
                  <FileVideo className="w-3.5 h-3.5 text-[#c3d9f3] shrink-0" />
                  <p className="text-xs uppercase tracking-wider text-white truncate font-medium">{selectedFile.name}</p>
                </div>
                <div className="flex items-center gap-2.5 text-[11px] text-[#999999] mt-1.5">
                  <span>{(selectedFile.size / (1024 * 1024)).toFixed(1)} MB</span>
                  <span className="text-[#3a3a3a]">/</span>
                  <span className="uppercase">{selectedFile.name.split(".").pop()}</span>
                  <span className="text-[#3a3a3a]">/</span>
                  <span className="text-emerald-400 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>READY</span>
                  </span>
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={handleClear}
              disabled={isUploading}
              className="px-3 py-1.5 border border-[#3a3a3a] text-xs font-mono uppercase tracking-[1.5px] text-[#999999] hover:text-white hover:border-white transition-colors cursor-pointer rounded-none disabled:opacity-40"
            >
              REMOVE
            </button>
          </div>
        )}

        {/* Error banner */}
        {errorMessage && (
          <div className="p-3 bg-[#141414] border border-red-900/60 font-mono text-xs text-red-400 flex items-center gap-2 rounded-none">
            <AlertCircle className="w-4 h-4 shrink-0 text-red-400" />
            <span>[ERROR] {errorMessage}</span>
          </div>
        )}

        {/* Analysis Configuration */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 p-4 bg-[#141414] border border-[#262626] rounded-none">
          <div>
            <label className="font-mono text-xs uppercase tracking-[1.5px] text-[#cccccc] flex items-center justify-between mb-2">
              <span>SAMPLING RATE (FRAMES PER SECOND)</span>
              <span className="text-[#c3d9f3]">{samplingFps.toFixed(0)} FPS</span>
            </label>
            <input
              type="range"
              min="1"
              max="15"
              step="1"
              value={samplingFps}
              disabled={isUploading}
              onChange={(e) => setSamplingFps(parseFloat(e.target.value))}
              className="w-full accent-white h-1 bg-[#262626] cursor-pointer rounded-none"
            />
            <p className="font-sans text-xs text-[#888888] mt-2">
              Samples {samplingFps} frame(s) per second for multi-face tracking and emotion extraction.
            </p>
          </div>

          <div>
            <label className="font-mono text-xs uppercase tracking-[1.5px] text-[#cccccc] block mb-2">
              SESSION TITLE (OPTIONAL)
            </label>
            <input
              type="text"
              placeholder="e.g. Interview Analysis, Focus Group"
              value={sessionName}
              disabled={isUploading}
              onChange={(e) => setSessionName(e.target.value)}
              className="input-valence w-full rounded-none"
            />
          </div>
        </div>

        {/* Submit Action */}
        <div className="flex justify-end pt-2">
          <button
            type="submit"
            disabled={!selectedFile || isUploading}
            className="btn-valence flex items-center gap-2 cursor-pointer rounded-none disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isUploading ? (
              <>
                <span className="inline-block w-3 h-3 border border-white border-t-transparent animate-spin rounded-none" />
                <span>UPLOADING & ANALYZING...</span>
              </>
            ) : (
              <span>START VIDEO ANALYSIS</span>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}

export default VideoUploader;
