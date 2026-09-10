"use client";

import React, { useState, useEffect } from "react";
import {
  Sparkles,
  User,
  Clock,
  ArrowLeft,
  Activity,
  Layers,
  BarChart3,
  Eye,
  CheckCircle2,
  HelpCircle,
} from "lucide-react";
import { VideoPlayer } from "./video-player";
import { EmotionTimeline } from "./emotion-timeline";
import { PredictionInspectModal } from "@/components/history/prediction-inspect-modal";
import { EMOTIONS, PredictionEmotion } from "@/types/emotion";
import { HistoricalPrediction } from "@/types/analytics";
import {
  VideoAnalysisDetail,
  VideoPredictionItem,
  VideoTimelineResponse,
} from "@/types/video-analysis";
import { getVideoStreamUrl, getVideoPredictions } from "@/lib/api/endpoints";

export interface VideoAnalysisResultProps {
  detail: VideoAnalysisDetail;
  timeline: VideoTimelineResponse;
  onReset: () => void;
  className?: string;
}

export function VideoAnalysisResult({
  detail,
  timeline,
  onReset,
  className = "",
}: VideoAnalysisResultProps) {
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [seekToTime, setSeekToTime] = useState<number | null>(null);
  const [predictions, setPredictions] = useState<VideoPredictionItem[]>([]);
  const [isLoadingPredictions, setIsLoadingPredictions] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<"transitions" | "distribution" | "predictions">("transitions");
  const [inspectedPrediction, setInspectedPrediction] = useState<HistoricalPrediction | null>(null);

  const videoId = detail.video_id;
  const streamUrl = getVideoStreamUrl(videoId);
  const analytics = detail.analytics;

  // Load initial batch of sampled predictions for overlay and inspection
  useEffect(() => {
    async function loadPreds() {
      setIsLoadingPredictions(true);
      try {
        const res = await getVideoPredictions(videoId, { page: 1, page_size: 500 });
        setPredictions(res.predictions || []);
      } catch (err) {
        console.error("Failed to load video predictions:", err);
      } finally {
        setIsLoadingPredictions(false);
      }
    }
    loadPreds();
  }, [videoId]);

  const handleSeek = (seconds: number) => {
    setSeekToTime(seconds);
    setCurrentTime(seconds);
    // Reset seekToTime after a tick so subsequent seeks to same timestamp trigger
    setTimeout(() => setSeekToTime(null), 50);
  };

  const formatTimestamp = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 10);
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}.${ms}`;
  };

  const handleInspect = (p: VideoPredictionItem) => {
    const histPred: HistoricalPrediction = {
      prediction_id: p.id,
      session_id: detail.session_id,
      request_id: `frame-${p.frame_index}`,
      timestamp: new Date().toISOString(),
      model_version: "champion-pruning-30",
      face_id: p.track_id,
      emotion: p.smoothed_emotion,
      confidence: p.smoothed_confidence,
      is_uncertain: p.is_uncertain,
      bbox: p.bbox,
      probabilities: p.probabilities,
      processing_time_ms: 0,
    };
    setInspectedPrediction(histPred);
  };

  const dominantMeta = analytics
    ? EMOTIONS[analytics.dominant_expression as PredictionEmotion] || EMOTIONS.neutral
    : EMOTIONS.neutral;

  return (
    <div className={`space-y-8 ${className}`}>
      {/* Top Banner & Actions */}
      <div className="bg-card/70 backdrop-blur-md border border-border/80 rounded-2xl p-6 shadow-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-500 border border-emerald-500/30 flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Analysis Completed
            </span>
            <span className="text-xs text-muted-foreground">•</span>
            <span className="text-xs text-muted-foreground font-mono">
              {detail.metadata.frames_analyzed} sampled frames @ {detail.metadata.analysis_fps} FPS
            </span>
          </div>
          <h2 className="text-2xl font-extrabold tracking-tight text-foreground mt-1.5 truncate max-w-xl">
            {detail.metadata.filename}
          </h2>
        </div>

        <button
          type="button"
          onClick={onReset}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-muted/80 hover:bg-muted text-foreground text-sm font-semibold border border-border transition-colors shadow-sm"
        >
          <ArrowLeft className="w-4 h-4" />
          Analyze Another Video
        </button>
      </div>

      {/* KPI Metrics Summary Grid */}
      {analytics && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3.5">
          <div className="bg-card/60 backdrop-blur-sm border border-border/60 rounded-xl p-4 shadow-sm">
            <span className="text-[11px] font-semibold text-muted-foreground uppercase flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-primary" /> Duration
            </span>
            <p className="text-xl font-bold font-mono text-foreground mt-1">
              {formatTimestamp(analytics.duration_seconds)}
            </p>
            <p className="text-[11px] text-muted-foreground mt-0.5">
              {detail.metadata.source_fps} FPS container
            </p>
          </div>

          <div className="bg-card/60 backdrop-blur-sm border border-border/60 rounded-xl p-4 shadow-sm">
            <span className="text-[11px] font-semibold text-muted-foreground uppercase flex items-center gap-1.5">
              <User className="w-3.5 h-3.5 text-primary" /> Tracked Faces
            </span>
            <p className="text-xl font-bold font-mono text-foreground mt-1">
              {analytics.tracked_faces_count}
            </p>
            <p className="text-[11px] text-muted-foreground mt-0.5">Spatial trajectories</p>
          </div>

          <div className="bg-card/60 backdrop-blur-sm border border-border/60 rounded-xl p-4 shadow-sm">
            <span className="text-[11px] font-semibold text-muted-foreground uppercase flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-primary" /> Predictions
            </span>
            <p className="text-xl font-bold font-mono text-foreground mt-1">
              {analytics.total_predictions_count.toLocaleString()}
            </p>
            <p className="text-[11px] text-muted-foreground mt-0.5">Model evaluations</p>
          </div>

          <div className="bg-card/60 backdrop-blur-sm border border-border/60 rounded-xl p-4 shadow-sm">
            <span className="text-[11px] font-semibold text-muted-foreground uppercase flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-primary" /> Transitions
            </span>
            <p className="text-xl font-bold font-mono text-foreground mt-1">
              {analytics.total_transitions_count}
            </p>
            <p className="text-[11px] text-muted-foreground mt-0.5">Confirmed shifts</p>
          </div>

          <div className="bg-card/60 backdrop-blur-sm border border-border/60 rounded-xl p-4 shadow-sm">
            <span className="text-[11px] font-semibold text-muted-foreground uppercase flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-primary" /> Dominant
            </span>
            <div className="flex items-center gap-1.5 mt-1">
              <span className="text-xl">{dominantMeta.emoji}</span>
              <p className="text-base font-bold text-foreground capitalize truncate">
                {analytics.dominant_expression}
              </p>
            </div>
            <p className="text-[11px] text-emerald-500 font-semibold mt-0.5">
              {analytics.dominant_expression_time_share.toFixed(1)}% time share
            </p>
          </div>

          <div className="bg-card/60 backdrop-blur-sm border border-border/60 rounded-xl p-4 shadow-sm">
            <span className="text-[11px] font-semibold text-muted-foreground uppercase flex items-center gap-1.5">
              <BarChart3 className="w-3.5 h-3.5 text-primary" /> Avg Confidence
            </span>
            <p className="text-xl font-bold font-mono text-emerald-500 mt-1">
              {(analytics.average_confidence * 100).toFixed(1)}%
            </p>
            <p className="text-[11px] text-muted-foreground mt-0.5">
              {analytics.processing_ratio}x realtime speed
            </p>
          </div>
        </div>
      )}

      {/* Main Interactive Stage: Video Player + Synchronized Timeline */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6 flex flex-col justify-start">
          <VideoPlayer
            streamUrl={streamUrl}
            metadata={detail.metadata}
            predictions={predictions}
            currentTime={currentTime}
            onTimeUpdate={setCurrentTime}
            seekToTime={seekToTime}
          />
        </div>

        <div className="lg:col-span-6 flex flex-col justify-start">
          <EmotionTimeline
            durationSeconds={detail.metadata.duration_seconds}
            segments={timeline.segments || []}
            events={timeline.events || []}
            tracks={detail.tracks || []}
            currentTime={currentTime}
            onSeek={handleSeek}
          />
        </div>
      </div>

      {/* Secondary Tabs: Transition Events, Distribution Analytics, Prediction Log */}
      <div className="bg-card/70 backdrop-blur-md border border-border/80 rounded-2xl p-6 shadow-xl space-y-6">
        <div className="flex items-center justify-between border-b border-border/60 pb-4">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setActiveTab("transitions")}
              className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
                activeTab === "transitions"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "bg-muted/60 text-muted-foreground hover:text-foreground"
              }`}
            >
              Expression Transitions ({timeline.events?.length || 0})
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("distribution")}
              className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
                activeTab === "distribution"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "bg-muted/60 text-muted-foreground hover:text-foreground"
              }`}
            >
              Time vs Prediction Share
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("predictions")}
              className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
                activeTab === "predictions"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "bg-muted/60 text-muted-foreground hover:text-foreground"
              }`}
            >
              Sampled Predictions Log ({predictions.length})
            </button>
          </div>

          <div className="hidden sm:flex items-center gap-1.5 text-xs text-muted-foreground">
            <HelpCircle className="w-3.5 h-3.5" />
            <span>Click any item to seek video</span>
          </div>
        </div>

        {/* Tab 1: Expression Transition Events */}
        {activeTab === "transitions" && (
          <div className="space-y-4">
            {timeline.events && timeline.events.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                {timeline.events.map((ev, idx) => {
                  const fromMeta = EMOTIONS[ev.from_emotion] || EMOTIONS.neutral;
                  const toMeta = EMOTIONS[ev.to_emotion] || EMOTIONS.neutral;

                  return (
                    <div
                      key={idx}
                      onClick={() => handleSeek(ev.timestamp)}
                      className="group p-3.5 rounded-xl bg-muted/40 border border-border/60 hover:border-primary/50 hover:bg-muted/60 cursor-pointer transition-all shadow-sm"
                    >
                      <div className="flex items-center justify-between text-xs mb-2">
                        <span className="font-mono font-bold text-primary flex items-center gap-1">
                          <Clock className="w-3.5 h-3.5" />
                          {formatTimestamp(ev.timestamp)}
                        </span>
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-muted text-muted-foreground border border-border/40">
                          Track {ev.track_id}
                        </span>
                      </div>

                      <div className="flex items-center gap-2 text-sm font-semibold">
                        <span className="flex items-center gap-1 text-muted-foreground">
                          <span>{fromMeta.emoji}</span>
                          <span className="capitalize">{ev.from_emotion}</span>
                        </span>
                        <span className="text-muted-foreground/60">→</span>
                        <span style={{ color: toMeta.color }} className="flex items-center gap-1 capitalize font-bold">
                          <span>{toMeta.emoji}</span>
                          <span>{ev.to_emotion}</span>
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-[11px] text-muted-foreground mt-2 pt-2 border-t border-border/30">
                        <span>Transition Conf:</span>
                        <span className="font-mono font-bold text-emerald-500">
                          {(ev.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-10 text-sm text-muted-foreground">
                No expression transitions detected in this video. The facial expression remained stable throughout.
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Expression Distribution (Time Share vs Frame Share) */}
        {activeTab === "distribution" && analytics && (
          <div className="space-y-4">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-border/60 text-xs font-semibold text-muted-foreground uppercase">
                    <th className="pb-3">Expression</th>
                    <th className="pb-3 text-right">Time Duration</th>
                    <th className="pb-3 text-right">Time Share (%)</th>
                    <th className="pb-3 text-right">Frame Count</th>
                    <th className="pb-3 text-right">Frame Share (%)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/40">
                  {analytics.expression_distribution.map((item) => {
                    const meta = EMOTIONS[item.emotion] || EMOTIONS.neutral;
                    return (
                      <tr key={item.emotion} className="hover:bg-muted/20 transition-colors">
                        <td className="py-3 font-medium flex items-center gap-2 text-foreground">
                          <span className="text-base">{meta.emoji}</span>
                          <span className="capitalize">{meta.label}</span>
                        </td>
                        <td className="py-3 text-right font-mono font-semibold text-foreground">
                          {item.time_seconds.toFixed(1)}s
                        </td>
                        <td className="py-3 text-right font-mono text-emerald-500 font-bold">
                          {item.time_share_percent.toFixed(1)}%
                        </td>
                        <td className="py-3 text-right font-mono text-foreground">
                          {item.count}
                        </td>
                        <td className="py-3 text-right font-mono text-muted-foreground">
                          {item.percentage.toFixed(1)}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab 3: Predictions Log with Inspect Modal */}
        {activeTab === "predictions" && (
          <div className="space-y-4">
            {isLoadingPredictions ? (
              <div className="text-center py-10 text-sm text-muted-foreground">
                Loading frame predictions...
              </div>
            ) : predictions.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-border/60 text-xs font-semibold text-muted-foreground uppercase">
                      <th className="pb-3">Time</th>
                      <th className="pb-3">Track</th>
                      <th className="pb-3">Predicted Expression</th>
                      <th className="pb-3 text-right">Confidence</th>
                      <th className="pb-3 text-right">Bounding Box</th>
                      <th className="pb-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/40">
                    {predictions.slice(0, 50).map((p) => {
                      const meta = EMOTIONS[p.smoothed_emotion] || EMOTIONS.neutral;
                      return (
                        <tr key={p.id} className="hover:bg-muted/20 transition-colors">
                          <td
                            onClick={() => handleSeek(p.timestamp)}
                            className="py-2.5 font-mono text-xs font-bold text-primary hover:underline cursor-pointer"
                          >
                            {formatTimestamp(p.timestamp)}
                          </td>
                          <td className="py-2.5 text-xs text-muted-foreground">
                            Track {p.track_id}
                          </td>
                          <td className="py-2.5 font-medium flex items-center gap-1.5 text-foreground capitalize">
                            <span>{meta.emoji}</span>
                            <span>{p.smoothed_emotion}</span>
                          </td>
                          <td className="py-2.5 text-right font-mono text-xs text-emerald-500 font-bold">
                            {(p.smoothed_confidence * 100).toFixed(1)}%
                          </td>
                          <td className="py-2.5 text-right font-mono text-[11px] text-muted-foreground">
                            [{p.bbox.x}, {p.bbox.y}, {p.bbox.width}×{p.bbox.height}]
                          </td>
                          <td className="py-2.5 text-right">
                            <button
                              type="button"
                              onClick={() => handleInspect(p)}
                              className="inline-flex items-center gap-1 px-2 py-1 rounded bg-primary/10 hover:bg-primary/20 text-primary text-xs font-medium transition-colors"
                            >
                              <Eye className="w-3.5 h-3.5" />
                              Inspect
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-10 text-sm text-muted-foreground">
                No sampled frame predictions found for this video.
              </div>
            )}
          </div>
        )}
      </div>

      {/* Prediction Explainability Inspect Modal */}
      <PredictionInspectModal
        prediction={inspectedPrediction}
        isOpen={!!inspectedPrediction}
        onClose={() => setInspectedPrediction(null)}
      />
    </div>
  );
}

export default VideoAnalysisResult;
