"use client";

import React, { useEffect, useState } from "react";
import { Activity, Clock, Layers, AlertCircle } from "lucide-react";
import { VideoAnalysisStatus } from "@/types/video-analysis";

export interface ProcessingProgressProps {
  status: VideoAnalysisStatus;
  filename?: string;
  onCancel?: () => Promise<void>;
  className?: string;
}

const STAGES = [
  { key: "queued", label: "01 / QUEUED" },
  { key: "processing", label: "02 / FRAME EXTRACTION" },
  { key: "generating_timeline", label: "03 / EMOTION INFERENCE" },
  { key: "saving", label: "04 / TEMPORAL SMOOTHING" },
  { key: "completed", label: "05 / COMPLETE" },
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
    <div className={`p-6 bg-[#0d0d0d] border border-[#262626] rounded-none ${className}`}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-5 border-b border-[#262626]">
        <div>
          <div className="font-mono text-xs uppercase tracking-[2px] text-[#999999] mb-1 flex items-center gap-2">
            <Activity className="w-3.5 h-3.5 text-[#c3d9f3]" />
            <span>TEMPORAL PIPELINE</span>
            <span className="text-[#3a3a3a]">/</span>
            <span className="text-[#c3d9f3]">BATCH PROCESSING</span>
          </div>
          <h3 className="font-display text-2xl uppercase tracking-[2.5px] text-white">
            EXECUTING VIDEO EMOTION ANALYSIS
          </h3>
          <p className="font-sans text-xs text-[#999999] mt-1">
            {filename ? `Analyzing specimen archive "${filename}"` : "Sampling frames and computing emotion vectors"}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 font-mono text-xs text-[#999999] bg-[#141414] px-3 py-1.5 border border-[#262626] rounded-none">
            <Clock className="w-3.5 h-3.5 text-[#c3d9f3]" />
            <span>ELAPSED: <span className="text-white font-semibold">{formatElapsed(elapsedSeconds)}</span></span>
          </div>
          {onCancel && (
            <button
              type="button"
              onClick={handleCancelClick}
              disabled={isCancelling}
              className="px-4 py-1.5 border border-red-500/60 bg-red-950/20 text-red-400 font-mono text-xs uppercase tracking-[1.5px] hover:bg-red-900/40 hover:text-white transition-all cursor-pointer rounded-none disabled:opacity-40"
            >
              {isCancelling ? "ABORTING..." : "ABORT JOB"}
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar & Percentage */}
      <div className="my-8 space-y-3">
        <div className="flex items-center justify-between font-mono text-xs uppercase tracking-[2px]">
          <span className="text-white flex items-center gap-2">
            <span className="w-2 h-2 rounded-none bg-[#c3d9f3] animate-pulse" />
            <Layers className="w-3.5 h-3.5 text-[#c3d9f3] inline" />
            <span>STAGE: {status.current_stage ? status.current_stage.replace(/_/g, " ") : "PROCESSING"}</span>
          </span>
          <span className="text-2xl text-white font-mono">
            {status.progress_percent.toFixed(1)}
            <span className="text-sm text-[#666666] ml-0.5">%</span>
          </span>
        </div>

        {/* 2px Precision Hairline Progress Bar */}
        <div className="h-1.5 w-full bg-[#1a1a1a] overflow-hidden rounded-none">
          <div
            className="h-full bg-[#c3d9f3] transition-all duration-300 rounded-none"
            style={{ width: `${Math.max(1, Math.min(100, status.progress_percent))}%` }}
          />
        </div>

        <div className="flex items-center justify-between font-mono text-[11px] text-[#666666]">
          <span>
            {status.frames_analyzed > 0
              ? `${status.frames_analyzed.toLocaleString()} FRAMES EVALUATED`
              : "BUFFERING FRAME STREAM..."}
          </span>
          {status.total_frames > 0 && (
            <span>{status.total_frames.toLocaleString()} TOTAL FRAMES</span>
          )}
        </div>
      </div>

      {/* Stage Stepper Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-px bg-[#262626] border border-[#262626] rounded-none">
        {STAGES.map((s, idx) => {
          const isDone = idx < activeIdx;
          const isCurrent = idx === activeIdx;

          return (
            <div
              key={s.key}
              className={`p-3 text-center transition-all ${
                isCurrent
                  ? "bg-[#1f1f1f] text-[#c3d9f3]"
                  : isDone
                  ? "bg-[#141414] text-white"
                  : "bg-[#0d0d0d] text-[#666666]"
              }`}
            >
              <div className="font-mono text-[10px] uppercase tracking-[1.5px]">
                {s.label}
              </div>
              <div className="font-mono text-[9px] mt-1 text-[#666666]">
                {isDone ? "[DONE]" : isCurrent ? "[ACTIVE]" : "[PENDING]"}
              </div>
            </div>
          );
        })}
      </div>

      {/* Failure alert */}
      {status.status === "FAILED" && (
        <div className="mt-6 p-4 bg-[#141414] border border-red-900/60 font-mono text-xs text-red-400 flex items-center gap-2 rounded-none">
          <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
          <span>[FAILED] {status.error_message || "Execution terminated unexpectedly during video analysis."}</span>
        </div>
      )}
    </div>
  );
}

export default ProcessingProgress;
