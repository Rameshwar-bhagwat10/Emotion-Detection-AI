"use client";

import React, { useState, useEffect } from "react";
import { ArrowLeft, Eye, Clock, Users, Zap, TrendingUp, Sparkles, CheckCircle2 } from "lucide-react";
import { VideoPlayer } from "./video-player";
import { EmotionTimeline } from "./emotion-timeline";
import { PredictionInspectModal } from "@/components/history/prediction-inspect-modal";
import { HistoricalPrediction } from "@/types/analytics";
import {
  VideoAnalysisDetail,
  VideoPredictionItem,
  VideoTimelineResponse,
} from "@/types/video-analysis";
import { getVideoStreamUrl, getVideoPredictions } from "@/lib/api/endpoints";
import { EMOTIONS, PredictionEmotion } from "@/types/emotion";

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

  const dominantEmotionKey = analytics?.dominant_expression?.toLowerCase() || "neutral";
  const dominantMeta = EMOTIONS[dominantEmotionKey as PredictionEmotion] || EMOTIONS.neutral;

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
      model_version: "champion-resnet18-cbam",
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

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Top Banner & Actions */}
      <div className="p-6 bg-[#0d0d0d] border border-[#262626] rounded-none flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-[2px] text-emerald-400">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <span>ANALYSIS COMPLETE</span>
            <span className="text-[#3a3a3a]">/</span>
            <span className="text-[#999999]">
              {detail.metadata.frames_analyzed} Frames Processed
            </span>
          </div>
          <h2 className="font-display text-2xl md:text-3xl uppercase tracking-[2px] text-white mt-1 truncate max-w-xl">
            {detail.metadata.filename}
          </h2>
        </div>

        <button
          type="button"
          onClick={onReset}
          className="btn-valence flex items-center gap-2 cursor-pointer rounded-none"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>ANALYZE ANOTHER VIDEO</span>
        </button>
      </div>

      {/* KPI Metrics Summary Grid (6 Spec Cells) - Zero-Overflow Numbers */}
      {analytics && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-px bg-[#262626] border border-[#262626] rounded-none">
          <div className="p-4 bg-[#0d0d0d]">
            <div className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[1.5px] text-[#666666] whitespace-nowrap truncate">
              <Clock className="w-3 h-3 text-[#c3d9f3] shrink-0" />
              <span>DURATION</span>
            </div>
            <p className="font-mono text-xl text-white mt-1 truncate">
              {formatTimestamp(analytics.duration_seconds)}
            </p>
            <p className="font-sans text-[11px] text-[#666666] mt-0.5 truncate">
              {detail.metadata.source_fps} FPS video
            </p>
          </div>

          <div className="p-4 bg-[#0d0d0d]">
            <div className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[1.5px] text-[#666666] whitespace-nowrap truncate">
              <Users className="w-3 h-3 text-[#c3d9f3] shrink-0" />
              <span>FACES TRACKED</span>
            </div>
            <p className="font-mono text-xl text-[#c3d9f3] mt-1 truncate">
              {analytics.tracked_faces_count}
            </p>
            <p className="font-sans text-[11px] text-[#666666] mt-0.5 truncate">Persons identified</p>
          </div>

          <div className="p-4 bg-[#0d0d0d]">
            <div className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[1.5px] text-[#666666] whitespace-nowrap truncate">
              <Zap className="w-3 h-3 text-[#c3d9f3] shrink-0" />
              <span>DETECTIONS</span>
            </div>
            <p className="font-mono text-xl text-white mt-1 truncate">
              {analytics.total_predictions_count.toLocaleString()}
            </p>
            <p className="font-sans text-[11px] text-[#666666] mt-0.5 truncate">Total face frames</p>
          </div>

          <div className="p-4 bg-[#0d0d0d]">
            <div className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[1.5px] text-[#666666] whitespace-nowrap truncate">
              <TrendingUp className="w-3 h-3 text-[#c3d9f3] shrink-0" />
              <span>EMOTION SHIFTS</span>
            </div>
            <p className="font-mono text-xl text-white mt-1 truncate">
              {analytics.total_transitions_count}
            </p>
            <p className="font-sans text-[11px] text-[#666666] mt-0.5 truncate">Expression changes</p>
          </div>

          <div className="p-4 bg-[#0d0d0d] relative overflow-hidden">
            <div className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[1.5px] text-[#666666] whitespace-nowrap truncate">
              <Sparkles className="w-3 h-3 shrink-0" style={{ color: dominantMeta.color }} />
              <span>DOMINANT EMOTION</span>
            </div>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-xl select-none">{dominantMeta.emoji}</span>
              <p className="font-display text-xl uppercase tracking-[1.5px] truncate font-normal" style={{ color: dominantMeta.color }}>
                {dominantMeta.label}
              </p>
            </div>
            <p className="font-mono text-[11px] text-[#999999] mt-0.5 truncate">
              {analytics.dominant_expression_time_share.toFixed(1)}% of duration
            </p>
          </div>

          <div className="p-4 bg-[#0d0d0d]">
            <span className="font-mono text-[10px] uppercase tracking-[1.5px] text-[#666666] block whitespace-nowrap truncate">
              AVG CONFIDENCE
            </span>
            <p className="font-mono text-xl text-white mt-1 truncate">
              {(analytics.average_confidence * 100).toFixed(1)}%
            </p>
            <p className="font-sans text-[11px] text-[#666666] mt-0.5 truncate">
              {analytics.processing_ratio}x real-time speed
            </p>
          </div>
        </div>
      )}

      {/* Main Interactive Stage: Video Player + Synchronized Timeline (Symmetrical Alignment) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        <div className="lg:col-span-6 flex flex-col justify-between">
          <VideoPlayer
            streamUrl={streamUrl}
            metadata={detail.metadata}
            predictions={predictions}
            currentTime={currentTime}
            onTimeUpdate={setCurrentTime}
            seekToTime={seekToTime}
          />
        </div>

        <div className="lg:col-span-6 flex flex-col justify-between">
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

      {/* Secondary Tabs */}
      <div className="p-6 bg-[#0d0d0d] border border-[#262626] rounded-none space-y-6">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-[#262626]">
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setActiveTab("transitions")}
              className={`px-4 py-1.5 font-mono text-xs uppercase tracking-[1.5px] transition-all cursor-pointer rounded-none ${
                activeTab === "transitions"
                  ? "bg-[#1f1f1f] text-white border border-white"
                  : "bg-[#141414] text-[#666666] border border-[#262626] hover:text-white"
              }`}
            >
              EMOTION CHANGES ({timeline.events?.length || 0})
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("distribution")}
              className={`px-4 py-1.5 font-mono text-xs uppercase tracking-[1.5px] transition-all cursor-pointer rounded-none ${
                activeTab === "distribution"
                  ? "bg-[#1f1f1f] text-white border border-white"
                  : "bg-[#141414] text-[#666666] border border-[#262626] hover:text-white"
              }`}
            >
              TIME PER EMOTION
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("predictions")}
              className={`px-4 py-1.5 font-mono text-xs uppercase tracking-[1.5px] transition-all cursor-pointer rounded-none ${
                activeTab === "predictions"
                  ? "bg-[#1f1f1f] text-white border border-white"
                  : "bg-[#141414] text-[#666666] border border-[#262626] hover:text-white"
              }`}
            >
              DETECTION LOGS ({predictions.length})
            </button>
          </div>

          <span className="font-mono text-[10px] uppercase tracking-[1.5px] text-[#666666]">
            CLICK ANY ROW TO JUMP TO THAT TIMESTAMP
          </span>
        </div>

        {/* Tab 1: Expression Transition Events */}
        {activeTab === "transitions" && (
          <div className="space-y-4">
            {timeline.events && timeline.events.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                {timeline.events.map((ev, idx) => {
                  const fromMeta = EMOTIONS[ev.from_emotion?.toLowerCase() as PredictionEmotion] || EMOTIONS.neutral;
                  const toMeta = EMOTIONS[ev.to_emotion?.toLowerCase() as PredictionEmotion] || EMOTIONS.neutral;

                  return (
                    <div
                      key={idx}
                      onClick={() => handleSeek(ev.timestamp)}
                      className="p-4 bg-[#141414] border border-[#262626] hover:border-white cursor-pointer transition-colors rounded-none"
                    >
                      <div className="flex items-center justify-between font-mono text-xs mb-2">
                        <span className="text-[#c3d9f3]">
                          {formatTimestamp(ev.timestamp)}
                        </span>
                        <span className="text-[10px] text-[#666666]">
                          FACE #{ev.track_id}
                        </span>
                      </div>

                      <div className="flex items-center gap-2 text-sm">
                        <span className="flex items-center gap-1 font-mono uppercase" style={{ color: fromMeta.color }}>
                          <span>{fromMeta.emoji}</span>
                          <span>{fromMeta.label}</span>
                        </span>
                        <span className="text-[#666666]">→</span>
                        <span className="flex items-center gap-1 font-mono uppercase font-bold" style={{ color: toMeta.color }}>
                          <span>{toMeta.emoji}</span>
                          <span>{toMeta.label}</span>
                        </span>
                      </div>

                      <div className="flex items-center justify-between font-mono text-[11px] text-[#666666] mt-3 pt-2 border-t border-[#262626]">
                        <span>CONFIDENCE:</span>
                        <span className="text-white">
                          {(ev.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-10 font-sans text-xs text-[#666666]">
                No emotion changes detected. The subject maintained a uniform expression throughout the video.
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Expression Distribution Table */}
        {activeTab === "distribution" && analytics && (
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead>
                <tr className="border-b border-[#262626] text-[#666666] uppercase tracking-[1.5px]">
                  <th className="pb-3">EMOTION</th>
                  <th className="pb-3 text-right">TOTAL TIME</th>
                  <th className="pb-3 text-right">TIME SHARE</th>
                  <th className="pb-3 text-right">FRAME COUNT</th>
                  <th className="pb-3 text-right">FRAME SHARE</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1f1f1f]">
                {analytics.expression_distribution.map((item) => {
                  const meta = EMOTIONS[item.emotion?.toLowerCase() as PredictionEmotion] || EMOTIONS.neutral;

                  return (
                    <tr key={item.emotion} className="hover:bg-[#141414] transition-colors">
                      <td className="py-3 text-white">
                        <div className="flex items-center gap-2">
                          <span
                            className="w-2 h-2 rounded-none inline-block"
                            style={{ backgroundColor: meta.color }}
                          />
                          <span className="text-base select-none">{meta.emoji}</span>
                          <span className="uppercase tracking-wider font-medium">{meta.label}</span>
                        </div>
                      </td>
                      <td className="py-3 text-right text-white">
                        {item.time_seconds.toFixed(1)}s
                      </td>
                      <td className="py-3 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <div className="w-16 h-1.5 bg-[#1f1f1f] rounded-none overflow-hidden hidden sm:block">
                            <div
                              className="h-full rounded-none"
                              style={{ width: `${item.time_share_percent}%`, backgroundColor: meta.color }}
                            />
                          </div>
                          <span style={{ color: meta.color }} className="font-bold">
                            {item.time_share_percent.toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="py-3 text-right text-white">
                        {item.count}
                      </td>
                      <td className="py-3 text-right text-[#666666]">
                        {item.percentage.toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Tab 3: Predictions Log */}
        {activeTab === "predictions" && (
          <div className="space-y-4">
            {isLoadingPredictions ? (
              <div className="text-center py-10 font-mono text-xs text-[#666666]">
                Loading predictions...
              </div>
            ) : predictions.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs">
                  <thead>
                    <tr className="border-b border-[#262626] text-[#666666] uppercase tracking-[1.5px]">
                      <th className="pb-3">TIMESTAMP</th>
                      <th className="pb-3">FACE ID</th>
                      <th className="pb-3">DETECTED EMOTION</th>
                      <th className="pb-3 text-right">CONFIDENCE</th>
                      <th className="pb-3 text-right">BOUNDING BOX</th>
                      <th className="pb-3 text-right">DETAILS</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1f1f1f]">
                    {predictions.slice(0, 50).map((p) => {
                      const meta = EMOTIONS[p.smoothed_emotion?.toLowerCase() as PredictionEmotion] || EMOTIONS.neutral;

                      return (
                        <tr key={p.id} className="hover:bg-[#141414] transition-colors">
                          <td
                            onClick={() => handleSeek(p.timestamp)}
                            className="py-2.5 text-[#c3d9f3] hover:underline cursor-pointer"
                          >
                            {formatTimestamp(p.timestamp)}
                          </td>
                          <td className="py-2.5 text-[#666666]">
                            FACE #{p.track_id}
                          </td>
                          <td className="py-2.5">
                            <div className="flex items-center gap-1.5 uppercase tracking-wider" style={{ color: meta.color }}>
                              <span>{meta.emoji}</span>
                              <span className="font-medium">{meta.label}</span>
                            </div>
                          </td>
                          <td className="py-2.5 text-right text-white">
                            {(p.smoothed_confidence * 100).toFixed(1)}%
                          </td>
                          <td className="py-2.5 text-right text-[#666666]">
                            [{p.bbox.x}, {p.bbox.y}, {p.bbox.width}×{p.bbox.height}]
                          </td>
                          <td className="py-2.5 text-right">
                            <button
                              type="button"
                              onClick={() => handleInspect(p)}
                              className="px-2.5 py-1 border border-[#3a3a3a] text-white hover:border-white font-mono text-[10px] uppercase tracking-wider transition-colors cursor-pointer rounded-none"
                            >
                              <Eye className="w-3 h-3 inline mr-1" />
                              INSPECT
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-10 font-sans text-xs text-[#666666]">
                No frame evaluations recorded for this video.
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
