"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { getSessionsAnalytics, getPredictionHistory } from "@/lib/api/endpoints";
import {
  SessionSummary,
  HistoricalPrediction,
  PredictionFilters,
} from "@/types/analytics";
import { EMOTIONS, SUPPORTED_EMOTIONS, PredictionEmotion } from "@/types/emotion";
import { PredictionInspectModal } from "@/components/history/prediction-inspect-modal";
import {
  History as HistoryIcon,
  RefreshCw,
  Clock,
  Layers,
  ChevronLeft,
  ChevronRight,
  Eye,
  AlertCircle,
} from "lucide-react";

export default function HistoryPage() {
  const [activeTab, setActiveTab] = useState<"sessions" | "predictions">("sessions");

  // Sessions state
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionTotal, setSessionTotal] = useState(0);
  const [sessionPage, setSessionPage] = useState(0);
  const sessionLimit = 15;

  // Predictions state
  const [predictions, setPredictions] = useState<HistoricalPrediction[]>([]);
  const [predictionTotal, setPredictionTotal] = useState(0);
  const [predPage, setPredPage] = useState(0);
  const predLimit = 20;

  // Filter state
  const [selectedEmotion, setSelectedEmotion] = useState<string>("all");
  const [uncertaintyFilter, setUncertaintyFilter] = useState<"all" | "uncertain" | "certain">("all");
  const [sortBy, setSortBy] = useState<"created_at" | "confidence">("created_at");
  const [sortOrder, setSortOrder] = useState<"desc" | "asc">("desc");

  // Inspect Modal State
  const [selectedPrediction, setSelectedPrediction] = useState<HistoricalPrediction | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Loading & Error States
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch Sessions
  const fetchSessions = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getSessionsAnalytics(sessionLimit, sessionPage * sessionLimit);
      setSessions(res.sessions);
      setSessionTotal(res.total);
    } catch (err: unknown) {
      console.error("Failed to load sessions:", err);
      const msg = err instanceof Error ? err.message : "Failed to load session history.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [sessionPage]);

  // Fetch Predictions
  const fetchPredictions = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const filters: PredictionFilters = {
        limit: predLimit,
        offset: predPage * predLimit,
        sort_by: sortBy,
        order: sortOrder,
      };

      if (selectedEmotion !== "all") {
        filters.emotion = selectedEmotion;
      }
      if (uncertaintyFilter === "uncertain") {
        filters.is_uncertain = true;
      } else if (uncertaintyFilter === "certain") {
        filters.is_uncertain = false;
      }

      const res = await getPredictionHistory(filters);
      setPredictions(res.items);
      setPredictionTotal(res.total);
    } catch (err: unknown) {
      console.error("Failed to load prediction history:", err);
      const msg = err instanceof Error ? err.message : "Failed to load prediction history.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [predPage, selectedEmotion, uncertaintyFilter, sortBy, sortOrder]);

  useEffect(() => {
    if (activeTab === "sessions") {
      fetchSessions();
    } else {
      fetchPredictions();
    }
  }, [activeTab, fetchSessions, fetchPredictions]);

  const handleInspect = (prediction: HistoricalPrediction) => {
    setSelectedPrediction(prediction);
    setIsModalOpen(true);
  };

  const formatDuration = (totalSeconds: number) => {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = Math.floor(totalSeconds % 60);
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="flex items-center justify-center w-8 h-8 rounded-xl bg-purple-500/15 border border-purple-500/30 text-purple-400">
              <HistoryIcon className="w-4 h-4" />
            </div>
            <h1 className="text-2xl font-bold text-zinc-100 tracking-tight">History & Logs</h1>
          </div>
          <p className="text-sm text-zinc-400 mt-1">
            Searchable historical archive of detection sessions and individual classified frames.
          </p>
        </div>

        {/* Tab Toggle */}
        <div className="flex items-center gap-1 p-1 rounded-xl bg-zinc-900 border border-zinc-800 self-start sm:self-auto">
          <button
            onClick={() => {
              setActiveTab("sessions");
              setSessionPage(0);
            }}
            className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-colors ${
              activeTab === "sessions"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <Clock className="w-3.5 h-3.5" />
            <span>Sessions ({sessionTotal})</span>
          </button>
          <button
            onClick={() => {
              setActiveTab("predictions");
              setPredPage(0);
            }}
            className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-colors ${
              activeTab === "predictions"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Prediction Log ({predictionTotal})</span>
          </button>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-start gap-3 text-rose-300 text-sm">
          <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="font-semibold">Query Error</div>
            <div className="text-xs text-rose-400/90 mt-0.5">{error}</div>
          </div>
          <button
            onClick={() => (activeTab === "sessions" ? fetchSessions() : fetchPredictions())}
            className="px-3 py-1 text-xs font-medium rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 transition-colors"
          >
            Retry
          </button>
        </div>
      )}

      {/* SESSIONS TAB */}
      {activeTab === "sessions" && (
        <div className="space-y-4">
          <div className="overflow-hidden rounded-2xl bg-zinc-900/60 border border-zinc-800/80">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="bg-zinc-950/60 border-b border-zinc-800 text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                  <tr>
                    <th className="py-3.5 px-4">Session Name / ID</th>
                    <th className="py-3.5 px-4">Date & Time</th>
                    <th className="py-3.5 px-4">Duration</th>
                    <th className="py-3.5 px-4">Predictions</th>
                    <th className="py-3.5 px-4">Dominant</th>
                    <th className="py-3.5 px-4">Avg Confidence</th>
                    <th className="py-3.5 px-4">Status</th>
                    <th className="py-3.5 px-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/50">
                  {loading && sessions.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="py-12 text-center text-zinc-500">
                        <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2 text-indigo-400" />
                        Loading sessions...
                      </td>
                    </tr>
                  ) : sessions.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="py-12 text-center text-zinc-500">
                        No sessions found. Start a live webcam detection session to record history.
                      </td>
                    </tr>
                  ) : (
                    sessions.map((sess) => {
                      const domKey = (sess.dominant_expression?.toLowerCase() || "neutral") as PredictionEmotion;
                      const domMeta = EMOTIONS[domKey] || EMOTIONS.neutral;

                      return (
                        <tr key={sess.id} className="hover:bg-zinc-800/25 transition-colors">
                          <td className="py-3.5 px-4">
                            <Link
                              href={`/sessions/${sess.id}`}
                              className="font-medium text-zinc-200 hover:text-indigo-400 transition-colors block"
                            >
                              {sess.name || `Session #${sess.id.slice(0, 8)}`}
                            </Link>
                            <span className="text-[11px] text-zinc-500 font-mono block">
                              {sess.id}
                            </span>
                          </td>
                          <td className="py-3.5 px-4 text-xs text-zinc-400 font-mono whitespace-nowrap">
                            {new Date(sess.started_at).toLocaleString()}
                          </td>
                          <td className="py-3.5 px-4 text-xs text-zinc-300 font-mono whitespace-nowrap">
                            {formatDuration(sess.duration_seconds)}
                          </td>
                          <td className="py-3.5 px-4 text-xs text-zinc-300 font-mono">
                            {sess.prediction_count.toLocaleString()}
                          </td>
                          <td className="py-3.5 px-4">
                            {sess.dominant_expression ? (
                              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-zinc-800 border border-zinc-700 text-zinc-200">
                                <span>{domMeta.emoji}</span>
                                <span className="capitalize">{domMeta.label}</span>
                              </span>
                            ) : (
                              <span className="text-xs text-zinc-500">—</span>
                            )}
                          </td>
                          <td className="py-3.5 px-4 text-xs font-mono font-medium text-indigo-400">
                            {(sess.average_confidence * 100).toFixed(1)}%
                          </td>
                          <td className="py-3.5 px-4">
                            <span
                              className={`text-[10px] px-2 py-0.5 rounded-full font-semibold uppercase tracking-wider border ${
                                sess.status === "active"
                                  ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
                                  : "bg-zinc-800 text-zinc-400 border-zinc-700"
                              }`}
                            >
                              {sess.status}
                            </span>
                          </td>
                          <td className="py-3.5 px-4 text-right">
                            <Link
                              href={`/sessions/${sess.id}`}
                              className="px-3 py-1 text-xs font-medium rounded-lg bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 transition-colors"
                            >
                              Analytics
                            </Link>
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            {sessionTotal > sessionLimit && (
              <div className="flex items-center justify-between p-4 border-t border-zinc-800 bg-zinc-950/40 text-xs text-zinc-400">
                <div>
                  Showing {sessionPage * sessionLimit + 1} to{" "}
                  {Math.min((sessionPage + 1) * sessionLimit, sessionTotal)} of {sessionTotal} sessions
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setSessionPage((p) => Math.max(0, p - 1))}
                    disabled={sessionPage === 0}
                    className="p-1.5 rounded-lg border border-zinc-800 bg-zinc-900 text-zinc-300 hover:bg-zinc-800 disabled:opacity-40 disabled:hover:bg-zinc-900"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <span className="font-mono px-2 font-medium text-zinc-200">
                    Page {sessionPage + 1} of {Math.ceil(sessionTotal / sessionLimit)}
                  </span>
                  <button
                    onClick={() => setSessionPage((p) => p + 1)}
                    disabled={(sessionPage + 1) * sessionLimit >= sessionTotal}
                    className="p-1.5 rounded-lg border border-zinc-800 bg-zinc-900 text-zinc-300 hover:bg-zinc-800 disabled:opacity-40 disabled:hover:bg-zinc-900"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* PREDICTIONS TAB */}
      {activeTab === "predictions" && (
        <div className="space-y-4">
          {/* Filter Bar */}
          <div className="p-4 rounded-2xl bg-zinc-900/60 border border-zinc-800/80 flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-3">
              {/* Emotion Filter */}
              <div className="flex items-center gap-2 text-xs">
                <span className="text-zinc-400 font-medium">Emotion:</span>
                <select
                  value={selectedEmotion}
                  onChange={(e) => {
                    setSelectedEmotion(e.target.value);
                    setPredPage(0);
                  }}
                  className="px-3 py-1.5 rounded-xl bg-zinc-950 border border-zinc-800 text-zinc-200 text-xs focus:outline-none focus:border-indigo-500"
                >
                  <option value="all">All Emotions</option>
                  {SUPPORTED_EMOTIONS.map((emo) => (
                    <option key={emo} value={emo}>
                      {EMOTIONS[emo as PredictionEmotion]?.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Uncertainty Filter */}
              <div className="flex items-center gap-2 text-xs">
                <span className="text-zinc-400 font-medium">Confidence:</span>
                <select
                  value={uncertaintyFilter}
                  onChange={(e) => {
                    setUncertaintyFilter(e.target.value as "all" | "uncertain" | "certain");
                    setPredPage(0);
                  }}
                  className="px-3 py-1.5 rounded-xl bg-zinc-950 border border-zinc-800 text-zinc-200 text-xs focus:outline-none focus:border-indigo-500"
                >
                  <option value="all">All Levels</option>
                  <option value="certain">High / Confident (≥60%)</option>
                  <option value="uncertain">Low / Uncertain (&lt;60%)</option>
                </select>
              </div>

              {/* Sort By */}
              <div className="flex items-center gap-2 text-xs">
                <span className="text-zinc-400 font-medium">Sort:</span>
                <select
                  value={`${sortBy}-${sortOrder}`}
                  onChange={(e) => {
                    const [field, ord] = e.target.value.split("-") as [
                      "created_at" | "confidence",
                      "desc" | "asc",
                    ];
                    setSortBy(field);
                    setSortOrder(ord);
                    setPredPage(0);
                  }}
                  className="px-3 py-1.5 rounded-xl bg-zinc-950 border border-zinc-800 text-zinc-200 text-xs focus:outline-none focus:border-indigo-500"
                >
                  <option value="created_at-desc">Newest First</option>
                  <option value="created_at-asc">Oldest First</option>
                  <option value="confidence-desc">Highest Confidence</option>
                  <option value="confidence-asc">Lowest Confidence</option>
                </select>
              </div>
            </div>

            <button
              onClick={fetchPredictions}
              disabled={loading}
              className="p-1.5 px-3 rounded-xl bg-zinc-950 border border-zinc-800 text-xs font-medium text-zinc-300 hover:text-white flex items-center gap-1.5"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
              <span>Refresh</span>
            </button>
          </div>

          {/* Predictions Table */}
          <div className="overflow-hidden rounded-2xl bg-zinc-900/60 border border-zinc-800/80">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="bg-zinc-950/60 border-b border-zinc-800 text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                  <tr>
                    <th className="py-3.5 px-4">Timestamp</th>
                    <th className="py-3.5 px-4">Predicted Expression</th>
                    <th className="py-3.5 px-4">Confidence</th>
                    <th className="py-3.5 px-4">Status</th>
                    <th className="py-3.5 px-4">Latency</th>
                    <th className="py-3.5 px-4">Session</th>
                    <th className="py-3.5 px-4 text-right">Inspect</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/50">
                  {loading && predictions.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="py-12 text-center text-zinc-500">
                        <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2 text-indigo-400" />
                        Loading predictions...
                      </td>
                    </tr>
                  ) : predictions.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="py-12 text-center text-zinc-500">
                        No predictions match the selected filter criteria.
                      </td>
                    </tr>
                  ) : (
                    predictions.map((pred) => {
                      const emoKey = (pred.emotion.toLowerCase() in EMOTIONS
                        ? pred.emotion.toLowerCase()
                        : "uncertain") as PredictionEmotion;
                      const meta = EMOTIONS[emoKey] || EMOTIONS.neutral;

                      return (
                        <tr key={`${pred.prediction_id}-${pred.face_id}`} className="hover:bg-zinc-800/25 transition-colors">
                          <td className="py-3.5 px-4 text-xs font-mono text-zinc-400 whitespace-nowrap">
                            {new Date(pred.timestamp).toLocaleTimeString()}{" "}
                            <span className="text-zinc-600 block text-[10px]">
                              {new Date(pred.timestamp).toLocaleDateString()}
                            </span>
                          </td>
                          <td className="py-3.5 px-4">
                            <div className="flex items-center gap-2">
                              <span className="text-lg">{meta.emoji}</span>
                              <span className="font-medium text-zinc-200 capitalize">
                                {meta.label}
                              </span>
                            </div>
                          </td>
                          <td className="py-3.5 px-4">
                            <div className="flex items-center gap-2.5">
                              <div className="w-16 bg-zinc-800 rounded-full h-1.5 overflow-hidden">
                                <div
                                  className="h-full rounded-full"
                                  style={{
                                    width: `${pred.confidence * 100}%`,
                                    backgroundColor: meta.color,
                                  }}
                                />
                              </div>
                              <span className="text-xs font-mono font-medium text-zinc-300">
                                {(pred.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          </td>
                          <td className="py-3.5 px-4">
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
                          <td className="py-3.5 px-4 text-xs font-mono text-zinc-400 whitespace-nowrap">
                            {pred.processing_time_ms} ms
                          </td>
                          <td className="py-3.5 px-4 text-xs font-mono text-zinc-400">
                            {pred.session_id ? (
                              <Link
                                href={`/sessions/${pred.session_id}`}
                                className="hover:text-indigo-400 transition-colors"
                              >
                                #{pred.session_id.slice(0, 8)}
                              </Link>
                            ) : (
                              <span className="text-zinc-600">Single image</span>
                            )}
                          </td>
                          <td className="py-3.5 px-4 text-right">
                            <button
                              onClick={() => handleInspect(pred)}
                              className="px-3 py-1 text-xs font-medium rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors flex items-center gap-1.5 ml-auto"
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

            {/* Pagination Controls */}
            {predictionTotal > predLimit && (
              <div className="flex items-center justify-between p-4 border-t border-zinc-800 bg-zinc-950/40 text-xs text-zinc-400">
                <div>
                  Showing {predPage * predLimit + 1} to{" "}
                  {Math.min((predPage + 1) * predLimit, predictionTotal)} of {predictionTotal} predictions
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setPredPage((p) => Math.max(0, p - 1))}
                    disabled={predPage === 0}
                    className="p-1.5 rounded-lg border border-zinc-800 bg-zinc-900 text-zinc-300 hover:bg-zinc-800 disabled:opacity-40 disabled:hover:bg-zinc-900"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <span className="font-mono px-2 font-medium text-zinc-200">
                    Page {predPage + 1} of {Math.ceil(predictionTotal / predLimit)}
                  </span>
                  <button
                    onClick={() => setPredPage((p) => p + 1)}
                    disabled={(predPage + 1) * predLimit >= predictionTotal}
                    className="p-1.5 rounded-lg border border-zinc-800 bg-zinc-900 text-zinc-300 hover:bg-zinc-800 disabled:opacity-40 disabled:hover:bg-zinc-900"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Inspect Modal */}
      <PredictionInspectModal
        prediction={selectedPrediction}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </div>
  );
}
