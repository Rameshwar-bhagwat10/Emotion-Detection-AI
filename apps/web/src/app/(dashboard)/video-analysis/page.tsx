"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { VideoUploader } from "@/components/video-analysis/video-uploader";
import { ProcessingProgress } from "@/components/video-analysis/processing-progress";
import { VideoAnalysisResult } from "@/components/video-analysis/video-analysis-result";
import {
  cancelVideoAnalysis,
  getRecentVideos,
  getVideoAnalysisDetail,
  getVideoAnalysisStatus,
  getVideoTimeline,
  uploadVideoForAnalysis,
} from "@/lib/api/endpoints";
import {
  RecentVideoItem,
  VideoAnalysisDetail,
  VideoAnalysisStatus,
  VideoTimelineResponse,
} from "@/types/video-analysis";
import {
  AlertCircle,
  Eye,
  Film,
  History,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  Loader2,
} from "lucide-react";

type AnalysisWorkflowState = "idle" | "uploading" | "processing" | "completed" | "error";

export default function VideoAnalysisPage() {
  const [workflowState, setWorkflowState] = useState<AnalysisWorkflowState>("idle");
  const [currentVideoId, setCurrentVideoId] = useState<string | null>(null);
  const [currentFilename, setCurrentFilename] = useState<string>("");
  const [statusData, setStatusData] = useState<VideoAnalysisStatus | null>(null);
  const [detailData, setDetailData] = useState<VideoAnalysisDetail | null>(null);
  const [timelineData, setTimelineData] = useState<VideoTimelineResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [recentVideos, setRecentVideos] = useState<RecentVideoItem[]>([]);
  const [isLoadingRecent, setIsLoadingRecent] = useState<boolean>(false);
  const [loadingVideoId, setLoadingVideoId] = useState<string | null>(null);

  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  const fetchRecent = useCallback(async () => {
    try {
      setIsLoadingRecent(true);
      const items = await getRecentVideos(15);
      setRecentVideos(items);
    } catch (err) {
      console.warn("Failed to load recent videos:", err);
    } finally {
      setIsLoadingRecent(false);
    }
  }, []);

  useEffect(() => {
    fetchRecent();
    return () => {
      stopPolling();
    };
  }, [fetchRecent, stopPolling]);

  // Load an existing completed analysis
  const handleLoadExisting = async (videoId: string, filename?: string) => {
    setErrorMessage(null);
    setLoadingVideoId(videoId);
    setCurrentVideoId(videoId);
    setCurrentFilename(filename || "video.mp4");

    try {
      const [detail, timeline] = await Promise.all([
        getVideoAnalysisDetail(videoId),
        getVideoTimeline(videoId),
      ]);
      setDetailData(detail);
      setTimelineData(timeline);
      setWorkflowState("completed");
    } catch (err: unknown) {
      console.error("Failed to load video analysis:", err);
      const msg = err instanceof Error ? err.message : "Failed to load video analysis.";
      setErrorMessage(msg);
      setWorkflowState("error");
    } finally {
      setLoadingVideoId(null);
    }
  };

  // Upload handler
  const handleUpload = async (file: File, samplingFps: number, _sessionName?: string) => {
    setErrorMessage(null);
    setWorkflowState("uploading");
    setCurrentFilename(file.name);

    try {
      const job = await uploadVideoForAnalysis(file, samplingFps, undefined, file.name);
      setCurrentVideoId(job.video_id);
      setStatusData({
        video_id: job.video_id,
        status: "QUEUED",
        current_stage: "queued",
        progress_percent: 0.0,
        frames_analyzed: 0,
        total_frames: 0,
        error_message: null,
      });
      setWorkflowState("processing");
      startPolling(job.video_id);
    } catch (err: unknown) {
      console.error("Upload error:", err);
      const msg = err instanceof Error ? err.message : "Failed to upload video.";
      setErrorMessage(msg);
      setWorkflowState("error");
    }
  };

  // Resilient status polling with transient error tolerance
  const startPolling = useCallback(
    (videoId: string) => {
      stopPolling();

      let consecutiveErrors = 0;
      const MAX_CONSECUTIVE_ERRORS = 5;

      pollIntervalRef.current = setInterval(async () => {
        try {
          const status = await getVideoAnalysisStatus(videoId);
          consecutiveErrors = 0; // Reset on success
          setStatusData(status);

          if (status.status === "COMPLETED") {
            stopPolling();
            fetchRecent();
            // Fetch final details & timeline
            const [detail, timeline] = await Promise.all([
              getVideoAnalysisDetail(videoId),
              getVideoTimeline(videoId),
            ]);
            setDetailData(detail);
            setTimelineData(timeline);
            setWorkflowState("completed");
          } else if (status.status === "FAILED") {
            stopPolling();
            fetchRecent();
            setErrorMessage(status.error_message || "Video analysis processing failed.");
            setWorkflowState("error");
          } else if (status.status === "CANCELLED") {
            stopPolling();
            fetchRecent();
            setErrorMessage("Video analysis was cancelled.");
            setWorkflowState("idle");
          }
        } catch (err: unknown) {
          consecutiveErrors += 1;
          console.warn(`Transient polling warning (${consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS}):`, err);
          if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
            stopPolling();
            setErrorMessage("Lost connection to video analysis server after multiple attempts.");
            setWorkflowState("error");
          }
        }
      }, 1500);
    },
    [stopPolling, fetchRecent]
  );

  const handleCancel = async () => {
    if (!currentVideoId) return;
    try {
      await cancelVideoAnalysis(currentVideoId);
    } catch (err) {
      console.error("Error cancelling video job:", err);
    } finally {
      stopPolling();
      setWorkflowState("idle");
      setCurrentVideoId(null);
      setStatusData(null);
      fetchRecent();
    }
  };

  const handleReset = () => {
    stopPolling();
    setWorkflowState("idle");
    setCurrentVideoId(null);
    setStatusData(null);
    setDetailData(null);
    setTimelineData(null);
    setErrorMessage(null);
    fetchRecent();
  };

  return (
    <div className="container max-w-7xl mx-auto px-4 py-8 space-y-8 animate-in fade-in duration-300">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-primary mb-1">
            <Film className="w-3.5 h-3.5" /> Temporal Machine Learning
          </div>
          <h1 className="text-3xl font-black tracking-tight text-foreground">
            Video Facial Expression Analysis
          </h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
            Analyze facial expressions across video duration with multi-face spatial tracking, temporal probability smoothing, transition detection, and synchronized interactive timeline playback.
          </p>
        </div>
      </div>

      {/* State: Idle / Uploading */}
      {(workflowState === "idle" || workflowState === "uploading") && (
        <div className="space-y-8">
          <VideoUploader
            onUpload={handleUpload}
            isUploading={workflowState === "uploading"}
          />

          {/* Recent Analyzed Videos List */}
          <div className="bg-card/70 backdrop-blur-md border border-border/80 rounded-2xl p-6 shadow-xl space-y-4">
            <div className="flex items-center justify-between border-b border-border/60 pb-3">
              <div className="flex items-center gap-2 text-foreground font-bold text-base">
                <History className="w-4 h-4 text-primary" />
                <span>Previously Analyzed Videos</span>
                <span className="text-xs font-normal text-muted-foreground">({recentVideos.length})</span>
              </div>
              <button
                type="button"
                onClick={fetchRecent}
                disabled={isLoadingRecent}
                className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 transition-colors"
                title="Refresh video list"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isLoadingRecent ? "animate-spin" : ""}`} />
                <span>Refresh</span>
              </button>
            </div>

            {isLoadingRecent && recentVideos.length === 0 ? (
              <div className="py-8 text-center text-xs text-muted-foreground">
                <Loader2 className="w-5 h-5 animate-spin mx-auto mb-2 text-primary" />
                Loading recent video records...
              </div>
            ) : recentVideos.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-border/60 text-xs font-semibold text-muted-foreground uppercase">
                      <th className="pb-2.5">Video Name</th>
                      <th className="pb-2.5">Status</th>
                      <th className="pb-2.5 text-right">Duration</th>
                      <th className="pb-2.5 text-right">Frames</th>
                      <th className="pb-2.5 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/40">
                    {recentVideos.map((v) => {
                      const isCompleted = v.status === "COMPLETED";
                      const isCurrentLoading = loadingVideoId === v.video_id;

                      return (
                        <tr key={v.video_id} className="hover:bg-muted/20 transition-colors">
                          <td className="py-3 font-medium text-foreground max-w-xs truncate">
                            <div className="flex items-center gap-2">
                              <Film className="w-4 h-4 text-primary shrink-0" />
                              <span className="truncate">{v.filename}</span>
                            </div>
                          </td>
                          <td className="py-3 text-xs">
                            {isCompleted ? (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-500 font-semibold border border-emerald-500/30">
                                <CheckCircle2 className="w-3 h-3" /> Completed
                              </span>
                            ) : v.status === "PROCESSING" ? (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-primary/10 text-primary font-semibold border border-primary/30">
                                <Loader2 className="w-3 h-3 animate-spin" /> {v.progress_percent}%
                              </span>
                            ) : v.status === "FAILED" ? (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-destructive/10 text-destructive font-semibold border border-destructive/30">
                                <AlertTriangle className="w-3 h-3" /> Failed
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-semibold">
                                {v.status}
                              </span>
                            )}
                          </td>
                          <td className="py-3 text-right font-mono text-xs text-muted-foreground">
                            {v.duration_seconds > 0 ? `${v.duration_seconds.toFixed(1)}s` : "—"}
                          </td>
                          <td className="py-3 text-right font-mono text-xs text-muted-foreground">
                            {v.frames_analyzed || "—"}
                          </td>
                          <td className="py-3 text-right">
                            {isCompleted ? (
                              <button
                                type="button"
                                onClick={() => handleLoadExisting(v.video_id, v.filename)}
                                disabled={isCurrentLoading}
                                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground font-semibold text-xs hover:bg-primary/90 transition-colors shadow-sm disabled:opacity-50"
                              >
                                {isCurrentLoading ? (
                                  <>
                                    <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading...
                                  </>
                                ) : (
                                  <>
                                    <Eye className="w-3.5 h-3.5" /> View Analysis
                                  </>
                                )}
                              </button>
                            ) : (
                              <span className="text-xs text-muted-foreground/60">—</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="py-6 text-center text-xs text-muted-foreground">
                No videos analyzed yet. Upload your first video above to start temporal expression analysis.
              </div>
            )}
          </div>
        </div>
      )}

      {/* State: Processing */}
      {workflowState === "processing" && statusData && (
        <ProcessingProgress
          status={statusData}
          filename={currentFilename}
          onCancel={handleCancel}
        />
      )}

      {/* State: Completed */}
      {workflowState === "completed" && detailData && timelineData && (
        <VideoAnalysisResult
          detail={detailData}
          timeline={timelineData}
          onReset={handleReset}
        />
      )}

      {/* State: Error */}
      {workflowState === "error" && (
        <div className="bg-card/70 backdrop-blur-md border border-destructive/30 rounded-2xl p-8 shadow-xl text-center space-y-4 max-w-xl mx-auto">
          <div className="w-12 h-12 rounded-full bg-destructive/10 text-destructive flex items-center justify-center mx-auto">
            <AlertCircle className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-foreground">Analysis Error</h3>
            <p className="text-sm text-muted-foreground mt-1">{errorMessage || "An unexpected error occurred."}</p>
          </div>
          <div className="flex items-center justify-center gap-3 pt-2">
            <button
              type="button"
              onClick={handleReset}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary text-primary-foreground font-semibold text-sm hover:bg-primary/90 transition-colors shadow-md"
            >
              <RefreshCw className="w-4 h-4" /> Try Again
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
