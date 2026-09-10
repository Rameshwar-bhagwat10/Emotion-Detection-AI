"use client";

import React, { useEffect, useState } from "react";
import { Loader2, AlertTriangle, CheckCircle2, Clock, Cpu, Film, Sparkles, XCircle } from "lucide-react";
import { VideoAnalysisStatus } from "@/types/video-analysis";

export interface ProcessingProgressProps {
  status: VideoAnalysisStatus;
  filename?: string;
  onCancel?: () => Promise<void>;
  className?: string;
}

const STAGES = [
  { key: "queued", label: "Queued", icon: Clock },
  { key: "processing", label: "Inference & Tracking", icon: Cpu },
  { key: "generating_timeline", label: "Temporal Smoothing", icon: Sparkles },
  { key: "saving", label: "Persisting Results", icon: Film },
  { key: "completed", label: "Complete", icon: CheckCircle2 },
];

export function ProcessingProgress({
  status,
  filename,
  onCancel,
  className = "",
}: ProcessingProgressProps) {
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [isCancelling, setIsCancelling] = useState<boolean>(false);

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const formatElapsed = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  const getActiveStageIndex = () => {
    const stage = (status.current_stage || "queued").toLowerCase();
    if (stage === "completed") return 4;
    if (stage === "saving") return 3;
    if (stage.includes("timeline") || stage.includes("smooth")) return 2;
    if (stage.includes("process") || stage.includes("infer") || stage.includes("track") || stage.includes("decod")) return 1;
    return 0;
  };

  const activeIdx = getActiveStageIndex();

  const handleCancelClick = async () => {
    if (!onCancel || isCancelling) return;
    setIsCancelling(true);
    try {
      await onCancel();
    } finally {
      setIsCancelling(false);
    }
  };

  return (
    <div className={`bg-card/80 backdrop-blur-md border border-border/80 rounded-2xl p-6 sm:p-8 shadow-xl ${className}`}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-6 border-b border-border/60">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="p-2 rounded-lg bg-primary/10 text-primary animate-spin">
              <Loader2 className="w-5 h-5" />
            </span>
            <div>
              <h3 className="text-lg font-bold text-foreground">
                Analyzing Video Expressions...
              </h3>
              <p className="text-xs text-muted-foreground mt-0.5">
                {filename ? `Processing "${filename}"` : "Executing streaming frame inference"}
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground bg-muted/60 px-3 py-1.5 rounded-lg border border-border/40">
            <Clock className="w-3.5 h-3.5 text-primary" />
            Elapsed: {formatElapsed(elapsedSeconds)}
          </div>
          {onCancel && (
            <button
              type="button"
              onClick={handleCancelClick}
              disabled={isCancelling}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-destructive hover:bg-destructive/10 border border-destructive/20 transition-colors disabled:opacity-50"
            >
              <XCircle className="w-3.5 h-3.5" />
              {isCancelling ? "Cancelling..." : "Cancel Job"}
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar & Percentage */}
      <div className="my-8 space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span className="font-semibold text-foreground flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-primary animate-ping" />
            {status.current_stage ? status.current_stage.replace(/_/g, " ").toUpperCase() : "PROCESSING"}
          </span>
          <span className="font-mono text-lg font-extrabold text-primary">
            {status.progress_percent.toFixed(1)}%
          </span>
        </div>

        <div className="relative h-3 w-full bg-muted/60 rounded-full overflow-hidden p-0.5 border border-border/40">
          <div
            className="h-full bg-gradient-to-r from-primary via-indigo-500 to-purple-500 rounded-full transition-all duration-300 shadow-sm"
            style={{ width: `${Math.max(2, Math.min(100, status.progress_percent))}%` }}
          />
        </div>

        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>
            {status.frames_analyzed > 0
              ? `${status.frames_analyzed} sampled frames evaluated`
              : "Ingesting video stream..."}
          </span>
          {status.total_frames > 0 && <span>{status.total_frames} total container frames</span>}
        </div>
      </div>

      {/* Stage Stepper */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5 pt-2">
        {STAGES.map((s, idx) => {
          const isDone = idx < activeIdx;
          const isCurrent = idx === activeIdx;
          const Icon = s.icon;

          return (
            <div
              key={s.key}
              className={`flex flex-col items-center justify-center p-3 rounded-xl border text-center transition-all ${
                isCurrent
                  ? "bg-primary/10 border-primary/40 text-primary shadow-sm"
                  : isDone
                  ? "bg-muted/40 border-emerald-500/30 text-emerald-500"
                  : "bg-muted/20 border-border/30 text-muted-foreground opacity-60"
              }`}
            >
              <div className="p-1.5 rounded-full mb-1.5">
                <Icon className={`w-4 h-4 ${isCurrent ? "animate-pulse" : ""}`} />
              </div>
              <span className="text-[11px] font-semibold leading-tight">{s.label}</span>
            </div>
          );
        })}
      </div>

      {/* Failure alert */}
      {status.status === "FAILED" && (
        <div className="mt-6 flex items-center gap-2.5 p-4 rounded-xl bg-destructive/10 border border-destructive/30 text-destructive text-sm">
          <AlertTriangle className="w-5 h-5 shrink-0" />
          <div>
            <p className="font-semibold">Analysis Failed</p>
            <p className="text-xs mt-0.5 opacity-90">{status.error_message || "An unexpected error occurred during processing."}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default ProcessingProgress;
