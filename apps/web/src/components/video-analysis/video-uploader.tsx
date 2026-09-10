"use client";

import React, { useState, useRef, useCallback } from "react";
import { UploadCloud, FileVideo, X, Settings2, AlertCircle, ArrowRight, Play } from "lucide-react";

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
    <div className={`bg-card/70 backdrop-blur-md border border-border/80 rounded-2xl p-6 sm:p-8 shadow-xl ${className}`}>
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-6 pb-5 border-b border-border/60">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-foreground flex items-center gap-2.5">
            <span className="p-2 rounded-lg bg-primary/10 text-primary">
              <FileVideo className="w-5 h-5" />
            </span>
            Temporal Video Expression Analysis
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            Upload video to detect multi-face trajectories, temporal expression transitions, and synchronized timeline analytics.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground bg-muted/60 px-3 py-1.5 rounded-full border border-border/50">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          Production Champion Engine
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Dropzone */}
        {!selectedFile ? (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`relative border-2 border-dashed rounded-xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-200 ${
              dragOver
                ? "border-primary bg-primary/5 scale-[1.005]"
                : "border-border/80 hover:border-primary/60 hover:bg-muted/30"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept={SUPPORTED_EXTENSIONS.join(",")}
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="flex flex-col items-center justify-center space-y-4">
              <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center text-primary group-hover:scale-110 transition-transform">
                <UploadCloud className="w-8 h-8" />
              </div>
              <div className="space-y-1.5">
                <p className="text-base font-semibold text-foreground">
                  Drag and drop your video file, or <span className="text-primary underline">browse</span>
                </p>
                <p className="text-xs text-muted-foreground">
                  Supports MP4, WebM, MOV, AVI, MKV up to {MAX_FILE_SIZE_MB}MB
                </p>
              </div>
            </div>
          </div>
        ) : (
          /* Selected File Preview */
          <div className="bg-muted/40 border border-border/80 rounded-xl p-4 sm:p-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div className="flex items-center gap-4 min-w-0">
              {previewUrl && (
                <div className="relative w-28 h-20 bg-black rounded-lg overflow-hidden shrink-0 border border-border/50 group">
                  <video src={previewUrl} className="w-full h-full object-cover" muted />
                  <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-80 group-hover:opacity-100 transition-opacity">
                    <Play className="w-5 h-5 text-white/90" />
                  </div>
                </div>
              )}
              <div className="min-w-0">
                <p className="text-sm font-semibold text-foreground truncate">{selectedFile.name}</p>
                <div className="flex items-center gap-3 text-xs text-muted-foreground mt-1">
                  <span>{(selectedFile.size / (1024 * 1024)).toFixed(1)} MB</span>
                  <span>•</span>
                  <span className="uppercase">{selectedFile.name.split(".").pop()}</span>
                  <span>•</span>
                  <span className="text-emerald-500 font-medium">Ready to analyze</span>
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={handleClear}
              disabled={isUploading}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-destructive hover:bg-destructive/10 border border-destructive/20 transition-colors disabled:opacity-50"
            >
              <X className="w-3.5 h-3.5" />
              Remove File
            </button>
          </div>
        )}

        {/* Error banner */}
        {errorMessage && (
          <div className="flex items-center gap-2.5 p-3.5 rounded-xl bg-destructive/10 border border-destructive/30 text-destructive text-sm animate-in fade-in">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <p>{errorMessage}</p>
          </div>
        )}

        {/* Analysis Configuration */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 p-4 rounded-xl bg-muted/25 border border-border/50">
          <div>
            <label className="text-xs font-semibold text-foreground flex items-center justify-between mb-1.5">
              <span className="flex items-center gap-1.5">
                <Settings2 className="w-3.5 h-3.5 text-muted-foreground" />
                Sampling Rate (Analysis FPS)
              </span>
              <span className="px-2 py-0.5 rounded bg-primary/10 text-primary font-mono text-xs">
                {samplingFps.toFixed(1)} FPS
              </span>
            </label>
            <input
              type="range"
              min="1"
              max="15"
              step="1"
              value={samplingFps}
              disabled={isUploading}
              onChange={(e) => setSamplingFps(parseFloat(e.target.value))}
              className="w-full accent-primary h-2 bg-muted rounded-lg cursor-pointer"
            />
            <p className="text-[11px] text-muted-foreground mt-1.5">
              Evaluates {samplingFps} frames per second. 5–10 FPS provides optimal temporal resolution without latency overhead.
            </p>
          </div>

          <div>
            <label className="text-xs font-semibold text-foreground block mb-1.5">
              Session Label (Optional)
            </label>
            <input
              type="text"
              placeholder="e.g. Interview Analysis #4"
              value={sessionName}
              disabled={isUploading}
              onChange={(e) => setSessionName(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-background border border-border/80 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/40"
            />
            <p className="text-[11px] text-muted-foreground mt-1.5">
              Groups temporal video records into application analytics history.
            </p>
          </div>
        </div>

        {/* Submit Action */}
        <div className="flex justify-end pt-2">
          <button
            type="submit"
            disabled={!selectedFile || isUploading}
            className="inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-primary text-primary-foreground font-semibold text-sm shadow-md hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-primary/40 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {isUploading ? (
              <>
                <span className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                Uploading & Scheduling...
              </>
            ) : (
              <>
                Start Video Analysis
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}

export default VideoUploader;
