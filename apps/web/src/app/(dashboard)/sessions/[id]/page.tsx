"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  getSessionAnalytics,
  getSessionTimeline,
  getPredictionHistory,
} from "@/lib/api/endpoints";
import {
  SessionAnalytics,
  TimelineAnalytics,
  HistoricalPrediction,
} from "@/types/analytics";
import { EMOTIONS, EmotionType } from "@/types/emotion";
import { EmotionDistributionChart } from "@/components/analytics/emotion-distribution-chart";
import { ConfidenceChart } from "@/components/analytics/confidence-chart";
import { EmotionTimelineChart } from "@/components/analytics/emotion-timeline-chart";
import { PredictionInspectModal } from "@/components/history/prediction-inspect-modal";
import {
  ArrowLeft,
  Eye,
} from "lucide-react";

export default function SessionDetailPage() {
  const params = useParams();
  const sessionId = params?.id as string;

  const [analytics, setAnalytics] = useState<SessionAnalytics | null>(null);
  const [timeline, setTimeline] = useState<TimelineAnalytics | null>(null);
  const [sessionPredictions, setSessionPredictions] = useState<HistoricalPrediction[]>([]);
  const [selectedPrediction, setSelectedPrediction] = useState<HistoricalPrediction | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSessionData = React.useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const [analyticsData, timelineData, historyData] = await Promise.all([
        getSessionAnalytics(sessionId),
        getSessionTimeline(sessionId),
        getPredictionHistory({ session_id: sessionId, limit: 50 }),
      ]);

      setAnalytics(analyticsData);
      setTimeline(timelineData);
      setSessionPredictions(historyData.items);
    } catch (err: unknown) {
      console.error("Failed to load session details:", err);
      const msg = err instanceof Error ? err.message : "Failed to load session analytics. The session may not exist.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    fetchSessionData();
  }, [fetchSessionData]);

  const formatDuration = (totalSeconds: number) => {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = Math.floor(totalSeconds % 60);
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  };

  if (loading) {
    return (
      <div className="py-24 text-center font-mono text-xs text-[#666666] space-y-2">
        <div className="text-white">INGESTING SESSION ARCHIVE TELEMETRY...</div>
        <div>SESSION: {sessionId}</div>
      </div>
    );
  }

  if (error || !analytics) {
    return (
      <div className="max-w-xl mx-auto py-16 text-center space-y-4 font-mono text-xs">
        <div className="text-red-500">[ARCHIVE RECORD NOT LOCATED]</div>
        <h2 className="font-display text-2xl uppercase tracking-[2px] text-white">SESSION RECORD UNAVAILABLE</h2>
        <p className="font-serif text-sm text-[#999999]">
          {error || "Could not retrieve telemetry records for this session ID."}
        </p>
        <div className="pt-2 flex items-center justify-center gap-3">
          <Link
            href="/dashboard"
            className="px-4 py-2 rounded-none border border-[#3a3a3a] text-[#cccccc] hover:text-white uppercase tracking-wider font-mono text-xs"
          >
            RETURN TO DASHBOARD
          </Link>
          <button
            type="button"
            onClick={fetchSessionData}
            className="btn-valence rounded-none"
          >
            RETRY INGESTION
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-200">
      {/* Navigation Breadcrumb */}
      <div className="flex items-center gap-2 font-mono text-xs text-[#666666] uppercase tracking-wider">
        <Link href="/dashboard" className="hover:text-white transition-colors flex items-center gap-1">
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>DASHBOARD</span>
        </Link>
        <span>/</span>
        <span className="text-[#c3d9f3]">{analytics.name || sessionId.slice(0, 8)}</span>
      </div>

      {/* Header Banner */}
      <div className="p-6 bg-[#0d0d0d] border border-[#262626] flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="font-display text-2xl md:text-3xl uppercase tracking-[2.5px] text-white">
              {analytics.name || `SESSION #${sessionId.slice(0, 8).toUpperCase()}`}
            </h1>
            <span
              className={`font-mono text-[10px] px-2.5 py-0.5 border uppercase tracking-wider ${
                analytics.status === "active"
                  ? "border-[#c3d9f3] text-[#c3d9f3] bg-[#141414]"
                  : "border-[#262626] text-[#666666] bg-[#000000]"
              }`}
            >
              {analytics.status}
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-4 text-xs text-[#999999] mt-2 font-mono">
            <span>TIMESTAMP: {new Date(analytics.started_at).toLocaleString()}</span>
            <span>/</span>
            <span>DURATION: {formatDuration(analytics.duration_seconds)}</span>
            <span>/</span>
            <span className="text-[#666666]">UUID: {sessionId}</span>
          </div>
        </div>

        <button
          type="button"
          onClick={fetchSessionData}
          className="px-3 py-1.5 border border-[#262626] font-mono text-[10px] uppercase tracking-[1.5px] text-[#999999] hover:text-white hover:border-white transition-colors cursor-pointer"
          title="Refresh Session"
        >
          [SYNC DATA]
        </button>
      </div>

      {/* KPI Cards (5 Spec Cells in Hairline Grid) */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-px bg-[#262626] border border-[#262626]">
        <div className="p-4 bg-[#0d0d0d]">
          <span className="font-mono text-[10px] uppercase tracking-[2px] text-[#666666] block">
            DETECTIONS
          </span>
          <div className="font-mono text-2xl text-white mt-1">
            {analytics.total_predictions.toLocaleString()}
          </div>
          <span className="font-serif text-xs text-[#666666] mt-0.5 block">Evaluated frames</span>
        </div>

        <div className="p-4 bg-[#0d0d0d]">
          <span className="font-mono text-[10px] uppercase tracking-[2px] text-[#666666] block">
            FACES FOUND
          </span>
          <div className="font-mono text-2xl text-[#c3d9f3] mt-1">
            {analytics.total_faces.toLocaleString()}
          </div>
          <span className="font-serif text-xs text-[#666666] mt-0.5 block">Detected faces</span>
        </div>

        <div className="p-4 bg-[#0d0d0d]">
          <span className="font-mono text-[10px] uppercase tracking-[2px] text-[#666666] block">
            DOMINANT EMOTION
          </span>
          {(() => {
            const emoKey = (analytics.dominant_expression || "").toLowerCase() as EmotionType;
            const meta = EMOTIONS[emoKey];
            return (
              <div className="flex items-center gap-1.5 mt-1">
                {meta && <span className="text-xl">{meta.emoji}</span>}
                <div
                  className="font-display text-xl uppercase tracking-[1px] truncate"
                  style={{ color: meta ? meta.color : "#ffffff" }}
                >
                  {meta ? meta.label : analytics.dominant_expression || "NONE"}
                </div>
              </div>
            );
          })()}
          <span className="font-serif text-xs text-[#666666] mt-0.5 block">Most frequent</span>
        </div>

        <div className="p-4 bg-[#0d0d0d]">
          <span className="font-mono text-[10px] uppercase tracking-[2px] text-[#666666] block">
            AVG CONFIDENCE
          </span>
          <div className="font-mono text-2xl text-white mt-1">
            {(analytics.confidence_analytics.average_confidence * 100).toFixed(1)}%
          </div>
          <span className="font-serif text-xs text-[#666666] mt-0.5 block">Model certainty</span>
        </div>

        <div className="p-4 bg-[#0d0d0d]">
          <span className="font-mono text-[10px] uppercase tracking-[2px] text-[#666666] block">
            SPEED
          </span>
          <div className="font-mono text-2xl text-white mt-1">
            {analytics.prediction_rate_per_minute.toFixed(1)}
            <span className="text-xs text-[#666666]">/MIN</span>
          </div>
          <span className="font-serif text-xs text-[#666666] mt-0.5 block">Detection rate</span>
        </div>
      </div>

      {/* Expression Timeline */}
      {timeline && (
        <EmotionTimelineChart
          timeline={timeline}
          title="Session Emotion Progression"
        />
      )}

      {/* Distribution & Confidence Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <EmotionDistributionChart
          distribution={analytics.expression_distribution}
          title="Emotion Distribution"
        />
        <ConfidenceChart
          confidenceAnalytics={analytics.confidence_analytics}
          title="Confidence Intervals"
        />
      </div>

      {/* Session Predictions Table */}
      <div className="bg-[#0d0d0d] border border-[#262626] space-y-4">
        <div className="p-5 border-b border-[#262626] flex items-center justify-between">
          <div>
            <h3 className="font-display text-xl uppercase tracking-[2px] text-white">
              DETECTED FRAMES LOG
            </h3>
            <p className="font-serif text-xs text-[#999999] mt-0.5">
              Individual classified frames recorded in this session.
            </p>
          </div>
          <span className="font-mono text-xs text-[#666666]">
            {sessionPredictions.length} DISPLAYED
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs">
            <thead>
              <tr className="border-b border-[#262626] text-[#666666] uppercase tracking-[1.5px]">
                <th className="py-3 px-4">TIMESTAMP</th>
                <th className="py-3 px-4">PREDICTED EMOTION</th>
                <th className="py-3 px-4">CONFIDENCE</th>
                <th className="py-3 px-4">STATUS</th>
                <th className="py-3 px-4">FACE POSITION</th>
                <th className="py-3 px-4 text-right">ACTION</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1f1f1f]">
              {sessionPredictions.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-[#666666] font-serif text-sm">
                    No individual predictions logged for this session.
                  </td>
                </tr>
              ) : (
                sessionPredictions.map((pred) => {
                  const emoKey = pred.emotion.toLowerCase() as EmotionType;
                  const meta = EMOTIONS[emoKey];
                  const color = meta?.color || "#38bdf8";

                  return (
                    <tr key={`${pred.prediction_id}-${pred.face_id}`} className="hover:bg-[#141414] transition-colors">
                      <td className="py-3 px-4 text-[#999999] whitespace-nowrap">
                        {new Date(pred.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="py-3 px-4 whitespace-nowrap">
                        <div className="flex items-center gap-2">
                          <span className="text-base">{meta?.emoji}</span>
                          <span className="font-bold tracking-wide" style={{ color }}>
                            {meta ? meta.label : pred.emotion}
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2.5">
                          <div className="w-12 bg-[#1a1a1a] h-1.5 overflow-hidden">
                            <div
                              className="h-full"
                              style={{
                                width: `${pred.confidence * 100}%`,
                                backgroundColor: color,
                              }}
                            />
                          </div>
                          <span className="font-medium text-white">
                            {(pred.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        {pred.is_uncertain ? (
                          <span className="text-amber-400 text-[10px]">
                            UNCERTAIN
                          </span>
                        ) : (
                          <span className="text-emerald-400 text-[10px]">
                            CONFIDENT
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-[#666666] whitespace-nowrap">
                        [{pred.bbox.x}, {pred.bbox.y}, {pred.bbox.width}×{pred.bbox.height}]
                      </td>
                      <td className="py-3 px-4 text-right">
                        <button
                          type="button"
                          onClick={() => {
                            setSelectedPrediction(pred);
                            setIsModalOpen(true);
                          }}
                          className="px-2.5 py-1 border border-[#3a3a3a] text-white hover:border-white text-[10px] uppercase tracking-wider transition-colors cursor-pointer inline-flex items-center gap-1.5"
                        >
                          <Eye className="w-3 h-3 text-[#c3d9f3]" />
                          <span>INSPECT</span>
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Inspect Modal */}
      <PredictionInspectModal
        prediction={selectedPrediction}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </div>
  );
}
