"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { Film, Activity, RefreshCw, AlertCircle, Trash2 } from "lucide-react";
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
  getUserVideos,
  saveUserVideo,
  updateUserVideo,
  deleteUserVideo,
} from "@/lib/storage/user-storage";
import {
  RecentVideoItem,
  VideoAnalysisDetail,
  VideoAnalysisStatus,
  VideoTimelineResponse,
} from "@/types/video-analysis";

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
      // 1. Read authoritative user video records from browser localStorage
      const userLocalVideos = getUserVideos();
      if (userLocalVideos.length === 0) {
        setRecentVideos([]);
        return;
      }

      // 2. In local development (with SQLite), reconcile with backend records
      try {
        const backendItems = await getRecentVideos(50);
        const backendMap = new Map(backendItems.map((b) => [b.video_id, b]));

        // Strictly filter so ONLY this user's videos are ever rendered
        const reconciled: RecentVideoItem[] = userLocalVideos.map((lv) => {
          const match = backendMap.get(lv.video_id);
          if (match) {
            // Sync any newly populated metrics from SQLite into localStorage
            if (match.status !== lv.status || match.duration_seconds !== lv.duration_seconds) {
              updateUserVideo(lv.video_id, {
                status: match.status,
                current_stage: match.current_stage,
                progress_percent: match.progress_percent,
                duration_seconds: match.duration_seconds,
                frames_analyzed: match.frames_analyzed,
              });
            }
            return match;
          }
          // Fallback to local storage version (for production or offline)
          return {
            video_id: lv.video_id,
            filename: lv.filename,
            status: lv.status,
            current_stage: lv.current_stage || "queued",
            progress_percent: lv.progress_percent,
            duration_seconds: lv.duration_seconds,
            created_at: lv.created_at,
            frames_analyzed: lv.frames_analyzed,
            error_message: lv.error_message || null,
          };
        });

        setRecentVideos(reconciled);
      } catch (err) {
        console.warn("Backend recent video query unavailable, using local storage:", err);
        // Fallback directly to localStorage records
        setRecentVideos(
          userLocalVideos.map((lv) => ({
            video_id: lv.video_id,
            filename: lv.filename,
            status: lv.status,
            current_stage: lv.current_stage || "queued",
            progress_percent: lv.progress_percent,
            duration_seconds: lv.duration_seconds,
            created_at: lv.created_at,
            frames_analyzed: lv.frames_analyzed,
            error_message: lv.error_message || null,
          }))
        );
      }
    } catch (err) {
      console.warn("Failed to load user videos:", err);
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

  // Upload handler with user localStorage persistence
  const handleUpload = async (file: File, samplingFps: number, _sessionName?: string) => {
    setErrorMessage(null);
    setWorkflowState("uploading");
    setCurrentFilename(file.name);

    try {
      const job = await uploadVideoForAnalysis(file, samplingFps, undefined, file.name);
      setCurrentVideoId(job.video_id);

      // Persist to user localStorage immediately
      saveUserVideo({
        video_id: job.video_id,
        filename: file.name,
        status: "QUEUED",
        current_stage: "queued",
        progress_percent: 0.0,
        frames_analyzed: 0,
        duration_seconds: 0,
        created_at: new Date().toISOString(),
      });

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

  // Resilient status polling with user localStorage progress updates
  const startPolling = useCallback(
    (videoId: string) => {
      stopPolling();

      let consecutiveErrors = 0;
      const MAX_CONSECUTIVE_ERRORS = 5;

      pollIntervalRef.current = setInterval(async () => {
        try {
          const status = await getVideoAnalysisStatus(videoId);
          consecutiveErrors = 0;
          setStatusData(status);

          // Update user video record in localStorage
          updateUserVideo(videoId, {
            status: status.status,
            current_stage: status.current_stage,
            progress_percent: status.progress_percent,
            frames_analyzed: status.frames_analyzed,
          });

          if (status.status === "COMPLETED") {
            stopPolling();
            const [detail, timeline] = await Promise.all([
              getVideoAnalysisDetail(videoId),
              getVideoTimeline(videoId),
            ]);
            updateUserVideo(videoId, {
              status: "COMPLETED",
              current_stage: "completed",
              progress_percent: 100,
              duration_seconds: detail.metadata.duration_seconds,
              frames_analyzed: detail.metadata.total_frames,
            });
            fetchRecent();
            setDetailData(detail);
            setTimelineData(timeline);
            setWorkflowState("completed");
          } else if (status.status === "FAILED") {
            stopPolling();
            updateUserVideo(videoId, {
              status: "FAILED",
              current_stage: "failed",
              error_message: status.error_message || "Video analysis processing failed.",
            });
            fetchRecent();
            setErrorMessage(status.error_message || "Video analysis processing failed.");
            setWorkflowState("error");
          } else if (status.status === "CANCELLED") {
            stopPolling();
            updateUserVideo(videoId, {
              status: "CANCELLED",
              current_stage: "cancelled",
            });
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

  const handleDeleteVideo = (videoId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    deleteUserVideo(videoId);
    setRecentVideos((prev) => prev.filter((v) => v.video_id !== videoId));
  };

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
    <div className="space-y-6">
      {/* 1. Page Header with Symmetrical Cyber-Tactical Layout */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-[#222222]">
        <div>
          <div className="font-mono text-xs uppercase tracking-[2px] text-[#888888] mb-1 flex items-center gap-2">
            <Film className="w-3.5 h-3.5 text-emerald-400" />
            <span>VIDEO ANALYSIS</span>
            <span className="text-[#3a3a3a]">/</span>
            <span className="text-[#c3d9f3]">EMOTION TRACKING</span>
          </div>
          <h1 className="font-display text-2xl sm:text-3xl uppercase tracking-[2px] text-white">
            Video Emotion Analysis
          </h1>
          <p className="font-sans text-xs text-[#888888] mt-1">
            Upload video files to detect expressions, track multiple faces simultaneously, and inspect emotion shifts across the timeline.
          </p>
        </div>

        {/* Status Indicators */}
        <div className="flex items-center gap-2.5 self-start sm:self-auto">
          {workflowState === "completed" && (
            <div className="flex items-center gap-2 px-3 py-1.5 border border-emerald-500/40 bg-emerald-950/20 font-mono text-[11px] uppercase tracking-[1.5px] text-emerald-400 rounded-none">
              <span className="w-2 h-2 rounded-none bg-emerald-400 animate-pulse" />
              <span>ANALYSIS COMPLETE</span>
            </div>
          )}

          {workflowState === "processing" && (
            <div className="flex items-center gap-2 px-3 py-1.5 border border-[#c3d9f3]/40 bg-[#c3d9f3]/10 font-mono text-[11px] uppercase tracking-[1.5px] text-[#c3d9f3] rounded-none">
              <span className="w-2 h-2 rounded-none bg-[#c3d9f3] animate-pulse" />
              <span>PROCESSING VIDEO</span>
            </div>
          )}

          <div className="flex items-center gap-2 px-3 py-1.5 border border-[#262626] bg-[#0d0d0d] font-mono text-[11px] uppercase tracking-[1.5px] text-[#cccccc] rounded-none">
            <Activity className="w-3 h-3 text-[#c3d9f3]" />
            <span>ENGINE: MULTI-TRACK YUNET</span>
          </div>
        </div>
      </div>

      {/* State: Idle / Uploading */}
      {(workflowState === "idle" || workflowState === "uploading") && (
        <div className="space-y-6">
          <VideoUploader
            onUpload={handleUpload}
            isUploading={workflowState === "uploading"}
          />

          {/* Recent Analyzed Videos List */}
          <div className="p-6 bg-[#0d0d0d] border border-[#262626] space-y-4 rounded-none">
            <div className="flex items-center justify-between pb-3 border-b border-[#262626]">
              <div className="font-mono text-xs uppercase tracking-[2px] text-white flex items-center gap-2">
                <Film className="w-3.5 h-3.5 text-[#c3d9f3]" />
                <span>YOUR ANALYZED VIDEOS</span>
                <span className="text-[#666666]">({recentVideos.length})</span>
                <span className="ml-2 px-1.5 py-0.5 border border-[#333333] bg-[#141414] text-[9px] uppercase tracking-[1px] text-[#888888] font-mono">
                  USER ISOLATED
                </span>
              </div>
              <button
                type="button"
                onClick={fetchRecent}
                disabled={isLoadingRecent}
                className="font-mono text-[11px] uppercase tracking-[1.5px] text-[#999999] hover:text-white transition-colors cursor-pointer flex items-center gap-1.5"
                title="Refresh video list"
              >
                <RefreshCw className={`w-3 h-3 ${isLoadingRecent ? "animate-spin text-[#c3d9f3]" : ""}`} />
                <span>{isLoadingRecent ? "[REFRESHING...]" : "[REFRESH LIST]"}</span>
              </button>
            </div>

            {isLoadingRecent && recentVideos.length === 0 ? (
              <div className="py-10 text-center font-mono text-xs text-[#666666]">
                Loading your analyzed videos...
              </div>
            ) : recentVideos.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs">
                  <thead>
                    <tr className="border-b border-[#262626] text-[#666666] uppercase tracking-[1.5px]">
                      <th className="pb-3">VIDEO NAME</th>
                      <th className="pb-3">STATUS</th>
                      <th className="pb-3 text-right">DURATION</th>
                      <th className="pb-3 text-right">FRAMES</th>
                      <th className="pb-3 text-right">ACTION</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1f1f1f]">
                    {recentVideos.map((v) => {
                      const isCompleted = v.status === "COMPLETED";
                      const isCurrentLoading = loadingVideoId === v.video_id;

                      return (
                        <tr key={v.video_id} className="hover:bg-[#141414] transition-colors group">
                          <td className="py-3 text-white max-w-xs truncate font-medium">
                            {v.filename}
                          </td>
                          <td className="py-3 text-[11px] tracking-wider">
                            {isCompleted ? (
                              <span className="px-2 py-0.5 border border-emerald-500/30 bg-emerald-950/20 text-emerald-400 font-mono text-[10px] uppercase tracking-wider rounded-none">
                                COMPLETED
                              </span>
                            ) : v.status === "PROCESSING" ? (
                              <span className="px-2 py-0.5 border border-[#c3d9f3]/40 bg-[#c3d9f3]/10 text-[#c3d9f3] font-mono text-[10px] uppercase tracking-wider animate-pulse rounded-none">
                                PROCESSING ({v.progress_percent}%)
                              </span>
                            ) : v.status === "FAILED" ? (
                              <span className="px-2 py-0.5 border border-red-500/40 bg-red-950/20 text-red-400 font-mono text-[10px] uppercase tracking-wider rounded-none">
                                FAILED
                              </span>
                            ) : (
                              <span className="text-[#666666]">{v.status}</span>
                            )}
                          </td>
                          <td className="py-3 text-right text-[#999999]">
                            {v.duration_seconds > 0 ? `${v.duration_seconds.toFixed(1)}S` : "—"}
                          </td>
                          <td className="py-3 text-right text-[#999999]">
                            {v.frames_analyzed || "—"}
                          </td>
                          <td className="py-3 text-right">
                            <div className="flex items-center justify-end gap-2">
                              {isCompleted && (
                                <button
                                  type="button"
                                  onClick={() => handleLoadExisting(v.video_id, v.filename)}
                                  disabled={isCurrentLoading}
                                  className="px-3 py-1.5 border border-white bg-white text-black font-mono text-[10px] uppercase tracking-[1.5px] hover:bg-[#eaeaea] transition-all cursor-pointer rounded-none disabled:opacity-40 font-semibold"
                                >
                                  {isCurrentLoading ? "LOADING..." : "VIEW RESULTS"}
                                </button>
                              )}
                              <button
                                type="button"
                                onClick={(e) => handleDeleteVideo(v.video_id, e)}
                                className="p-1.5 border border-[#262626] hover:border-red-500/50 bg-[#111111] hover:bg-red-950/20 text-[#666666] hover:text-red-400 transition-colors cursor-pointer rounded-none"
                                title="Remove from your history"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="py-10 text-center font-sans text-xs text-[#666666]">
                No analyzed videos found for your account. Upload your first video above to get started.
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
        <div className="p-8 bg-[#0d0d0d] border border-red-900/60 text-center space-y-4 max-w-xl mx-auto rounded-none">
          <div className="w-10 h-10 border border-red-500/40 bg-red-950/20 flex items-center justify-center mx-auto rounded-none text-red-400">
            <AlertCircle className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-display text-2xl uppercase tracking-[2px] text-white">ANALYSIS FAILED</h3>
            <p className="font-sans text-xs text-[#999999] mt-1">{errorMessage || "An unexpected error occurred during execution."}</p>
          </div>
          <div className="pt-2">
            <button
              type="button"
              onClick={handleReset}
              className="btn-valence cursor-pointer rounded-none"
            >
              TRY AGAIN
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
