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
import { EmotionDistributionChart } from "@/components/analytics/emotion-distribution-chart";
import { ConfidenceChart } from "@/components/analytics/confidence-chart";
import { EmotionTimelineChart } from "@/components/analytics/emotion-timeline-chart";
import { PredictionInspectModal } from "@/components/history/prediction-inspect-modal";
import { EMOTIONS, PredictionEmotion } from "@/types/emotion";
import {
  ArrowLeft,
  Calendar,
  Clock,
  Activity,
  Layers,
  Sparkles,
  Eye,
  AlertCircle,
  RefreshCw,
  Zap,
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

  const dominantKey = (analytics?.dominant_expression?.toLowerCase() || "neutral") as PredictionEmotion;
  const dominantMeta = EMOTIONS[dominantKey] || EMOTIONS.neutral;

  const formatDuration = (totalSeconds: number) => {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = Math.floor(totalSeconds % 60);
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  };

  if (loading) {
    return (
      <div className="py-20 text-center space-y-3">
        <RefreshCw className="w-8 h-8 animate-spin mx-auto text-indigo-400" />
        <h3 className="text-base font-semibold text-zinc-300">Loading Session Analytics...</h3>
        <p className="text-xs text-zinc-500 font-mono">Session ID: {sessionId}</p>
      </div>
    );
  }

  if (error || !analytics) {
    return (
      <div className="max-w-2xl mx-auto py-12 space-y-4 text-center">
        <div className="w-12 h-12 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-400 flex items-center justify-center mx-auto">
          <AlertCircle className="w-6 h-6" />
        </div>
        <h2 className="text-xl font-bold text-zinc-100">Session Not Found</h2>
        <p className="text-sm text-zinc-400">
          {error || "Could not retrieve analytics for this session."}
        </p>
        <div className="pt-2 flex items-center justify-center gap-3">
          <Link
            href="/history"
            className="px-4 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-sm font-medium text-zinc-200 transition-colors"
          >
            Back to Sessions
          </Link>
          <button
            onClick={fetchSessionData}
            className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-sm font-medium text-white transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Navigation Breadcrumb */}
      <div className="flex items-center gap-2 text-xs text-zinc-400">
        <Link href="/history" className="hover:text-indigo-400 transition-colors flex items-center gap-1">
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Sessions</span>
        </Link>
        <span>/</span>
        <span className="text-zinc-200 font-mono">{analytics.name || sessionId.slice(0, 8)}</span>
      </div>

      {/* Header Banner */}
      <div className="p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-zinc-100 tracking-tight">
              {analytics.name || `Session #${sessionId.slice(0, 8)}`}
            </h1>
            <span
              className={`text-xs px-2.5 py-0.5 rounded-full font-semibold uppercase tracking-wider border ${
                analytics.status === "active"
                  ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
                  : "bg-zinc-800 text-zinc-400 border-zinc-700"
              }`}
            >
              {analytics.status}
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-4 text-xs text-zinc-400 mt-2 font-mono">
            <span className="flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5 text-zinc-500" />
              {new Date(analytics.started_at).toLocaleString()}
            </span>
            <span>•</span>
            <span className="flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-zinc-500" />
              Duration: {formatDuration(analytics.duration_seconds)}
            </span>
            <span>•</span>
            <span className="text-zinc-500">ID: {sessionId}</span>
          </div>
        </div>

        <button
          onClick={fetchSessionData}
          className="p-2 self-start sm:self-auto rounded-xl bg-zinc-950 border border-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition-colors"
          title="Refresh Session"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        {/* Total Predictions */}
        <div className="p-4 rounded-2xl bg-zinc-900/60 border border-zinc-800/80">
          <div className="flex items-center justify-between text-zinc-500 mb-2">
            <span className="text-[11px] font-bold uppercase tracking-wider">Predictions</span>
            <Layers className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-bold text-zinc-100 font-mono">
            {analytics.total_predictions.toLocaleString()}
          </div>
          <span className="text-[11px] text-zinc-500 mt-1 block">Inference events</span>
        </div>

        {/* Faces Detected */}
        <div className="p-4 rounded-2xl bg-zinc-900/60 border border-zinc-800/80">
          <div className="flex items-center justify-between text-zinc-500 mb-2">
            <span className="text-[11px] font-bold uppercase tracking-wider">Faces</span>
            <Sparkles className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-bold text-zinc-100 font-mono">
            {analytics.total_faces.toLocaleString()}
          </div>
          <span className="text-[11px] text-zinc-500 mt-1 block">Classified faces</span>
        </div>

        {/* Dominant Expression */}
        <div className="p-4 rounded-2xl bg-zinc-900/60 border border-zinc-800/80">
          <div className="flex items-center justify-between text-zinc-500 mb-2">
            <span className="text-[11px] font-bold uppercase tracking-wider">Dominant</span>
            <span className="text-base">{dominantMeta.emoji}</span>
          </div>
          <div className="text-2xl font-bold text-zinc-100 capitalize">
            {analytics.dominant_expression ? dominantMeta.label : "None"}
          </div>
          <span className="text-[11px] text-zinc-500 mt-1 block">Most frequent</span>
        </div>

        {/* Average Confidence */}
        <div className="p-4 rounded-2xl bg-zinc-900/60 border border-zinc-800/80">
          <div className="flex items-center justify-between text-zinc-500 mb-2">
            <span className="text-[11px] font-bold uppercase tracking-wider">Confidence</span>
            <Activity className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400 font-mono">
            {(analytics.confidence_analytics.average_confidence * 100).toFixed(1)}%
          </div>
          <span className="text-[11px] text-zinc-500 mt-1 block">Mean score</span>
        </div>

        {/* Prediction Rate */}
        <div className="p-4 rounded-2xl bg-zinc-900/60 border border-zinc-800/80">
          <div className="flex items-center justify-between text-zinc-500 mb-2">
            <span className="text-[11px] font-bold uppercase tracking-wider">Rate</span>
            <Zap className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-zinc-100 font-mono">
            {analytics.prediction_rate_per_minute.toFixed(1)}/m
          </div>
          <span className="text-[11px] text-zinc-500 mt-1 block">Inference velocity</span>
        </div>
      </div>

      {/* Expression Timeline */}
      {timeline && (
        <EmotionTimelineChart
          timeline={timeline}
          title="Session Expression Progression"
        />
      )}

      {/* Distribution & Confidence Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <EmotionDistributionChart
          distribution={analytics.expression_distribution}
          title="Session Expression Breakdown"
        />
        <ConfidenceChart
          confidenceAnalytics={analytics.confidence_analytics}
          title="Session Confidence Distribution"
        />
      </div>

      {/* Session Predictions Table */}
      <div className="overflow-hidden rounded-2xl bg-zinc-900/60 border border-zinc-800/80">
        <div className="p-5 border-b border-zinc-800 flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-zinc-100">Session Prediction Log</h3>
            <p className="text-xs text-zinc-400 mt-0.5">
              Individual classified frames recorded in this session.
            </p>
          </div>
          <span className="text-xs font-mono text-zinc-500">
            {sessionPredictions.length} displayed
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-zinc-950/60 border-b border-zinc-800 text-xs font-semibold text-zinc-400 uppercase tracking-wider">
              <tr>
                <th className="py-3 px-4">Time</th>
                <th className="py-3 px-4">Predicted Expression</th>
                <th className="py-3 px-4">Confidence</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4">Face Box</th>
                <th className="py-3 px-4 text-right">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/50">
              {sessionPredictions.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-zinc-500 text-xs">
                    No individual predictions recorded for this session.
                  </td>
                </tr>
              ) : (
                sessionPredictions.map((pred) => {
                  const emoKey = (pred.emotion.toLowerCase() in EMOTIONS
                    ? pred.emotion.toLowerCase()
                    : "uncertain") as PredictionEmotion;
                  const meta = EMOTIONS[emoKey] || EMOTIONS.neutral;

                  return (
                    <tr key={`${pred.prediction_id}-${pred.face_id}`} className="hover:bg-zinc-800/25 transition-colors">
                      <td className="py-3 px-4 text-xs font-mono text-zinc-400 whitespace-nowrap">
                        {new Date(pred.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <span className="text-base">{meta.emoji}</span>
                          <span className="font-medium text-zinc-200 capitalize">
                            {meta.label}
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className="text-xs font-mono font-medium text-indigo-400">
                          {(pred.confidence * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        {pred.is_uncertain ? (
                          <span className="text-[10px] px-2 py-0.5 rounded-full font-semibold bg-amber-500/15 text-amber-300 border border-amber-500/30">
                            Uncertain
                          </span>
                        ) : (
                          <span className="text-[10px] px-2 py-0.5 rounded-full font-semibold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                            Calibrated
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-xs font-mono text-zinc-400 whitespace-nowrap">
                        [{pred.bbox.x}, {pred.bbox.y}, {pred.bbox.width}×{pred.bbox.height}]
                      </td>
                      <td className="py-3 px-4 text-right">
                        <button
                          onClick={() => {
                            setSelectedPrediction(pred);
                            setIsModalOpen(true);
                          }}
                          className="px-2.5 py-1 text-xs font-medium rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors inline-flex items-center gap-1.5"
                        >
                          <Eye className="w-3.5 h-3.5 text-indigo-400" />
                          <span>Inspect</span>
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
