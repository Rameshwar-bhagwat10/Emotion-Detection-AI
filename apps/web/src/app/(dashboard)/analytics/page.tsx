"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { getGlobalAnalytics, getSessionsAnalytics } from "@/lib/api/endpoints";
import { GlobalAnalytics, SessionSummary } from "@/types/analytics";
import { EmotionDistributionChart } from "@/components/analytics/emotion-distribution-chart";
import { ConfidenceChart } from "@/components/analytics/confidence-chart";
import { EMOTIONS, PredictionEmotion } from "@/types/emotion";
import {
  Activity,
  Clock,
  Eye,
  LineChart,
  RefreshCw,
  Sparkles,
  Users,
  AlertCircle,
  ArrowUpRight,
} from "lucide-react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<GlobalAnalytics | null>(null);
  const [recentSessions, setRecentSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeFilter, setTimeFilter] = useState<"all" | "7d" | "30d">("all");

  const fetchAnalytics = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      let startDate: string | undefined = undefined;
      if (timeFilter === "7d") {
        const d = new Date();
        d.setDate(d.getDate() - 7);
        startDate = d.toISOString();
      } else if (timeFilter === "30d") {
        const d = new Date();
        d.setDate(d.getDate() - 30);
        startDate = d.toISOString();
      }

      const [globalData, sessionsData] = await Promise.all([
        getGlobalAnalytics(startDate ? { start_date: startDate } : undefined),
        getSessionsAnalytics(5, 0),
      ]);

      setAnalytics(globalData);
      setRecentSessions(sessionsData.sessions);
    } catch (err: unknown) {
      console.error("Failed to load analytics:", err);
      const msg = err instanceof Error ? err.message : "Failed to load analytics overview. Please ensure backend is running.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [timeFilter]);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  const dominantKey = (analytics?.dominant_expression?.toLowerCase() || "neutral") as PredictionEmotion;
  const dominantMeta = EMOTIONS[dominantKey] || EMOTIONS.neutral;

  const formatDuration = (totalSeconds: number) => {
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = Math.floor(totalSeconds % 60);
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${seconds}s`;
    return `${seconds}s`;
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Page Title & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="flex items-center justify-center w-8 h-8 rounded-xl bg-indigo-500/15 border border-indigo-500/30 text-indigo-400">
              <LineChart className="w-4 h-4" />
            </div>
            <h1 className="text-2xl font-bold text-zinc-100 tracking-tight">Global Analytics</h1>
          </div>
          <p className="text-sm text-zinc-400 mt-1">
            Aggregated facial-expression classification telemetry, model confidence, and historical trends.
          </p>
        </div>

        {/* Filter controls */}
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <div className="flex items-center p-1 rounded-xl bg-zinc-900 border border-zinc-800">
            <button
              onClick={() => setTimeFilter("all")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                timeFilter === "all"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              All Time
            </button>
            <button
              onClick={() => setTimeFilter("30d")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                timeFilter === "30d"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              30 Days
            </button>
            <button
              onClick={() => setTimeFilter("7d")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                timeFilter === "7d"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              7 Days
            </button>
          </div>

          <button
            onClick={fetchAnalytics}
            disabled={loading}
            className="p-2 rounded-xl bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition-colors disabled:opacity-50"
            title="Refresh Analytics"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-start gap-3 text-rose-300 text-sm">
          <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="font-semibold">Unable to Load Analytics</div>
            <div className="text-xs text-rose-400/90 mt-0.5">{error}</div>
          </div>
          <button
            onClick={fetchAnalytics}
            className="px-3 py-1 text-xs font-medium rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 transition-colors"
          >
            Retry
          </button>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && !analytics && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="h-28 rounded-2xl bg-zinc-900/40 border border-zinc-800/40 animate-pulse" />
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="h-80 rounded-2xl bg-zinc-900/40 border border-zinc-800/40 animate-pulse" />
            <div className="h-80 rounded-2xl bg-zinc-900/40 border border-zinc-800/40 animate-pulse" />
          </div>
        </div>
      )}

      {/* Main Content */}
      {analytics && (
        <div className="space-y-8">
          {/* Top KPI Metric Cards */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {/* Total Sessions */}
            <div className="p-4 rounded-2xl bg-zinc-900/60 border border-zinc-800/80">
              <div className="flex items-center justify-between text-zinc-500 mb-2">
                <span className="text-[11px] font-bold uppercase tracking-wider">Sessions</span>
                <Users className="w-4 h-4 text-indigo-400" />
              </div>
              <div className="text-2xl font-bold text-zinc-100 font-mono">
                {analytics.total_sessions.toLocaleString()}
              </div>
              <span className="text-[11px] text-zinc-500 mt-1 block">Live & analysis</span>
            </div>

            {/* Total Predictions */}
            <div className="p-4 rounded-2xl bg-zinc-900/60 border border-zinc-800/80">
              <div className="flex items-center justify-between text-zinc-500 mb-2">
                <span className="text-[11px] font-bold uppercase tracking-wider">Predictions</span>
                <Eye className="w-4 h-4 text-purple-400" />
              </div>
              <div className="text-2xl font-bold text-zinc-100 font-mono">
                {analytics.total_predictions.toLocaleString()}
              </div>
              <span className="text-[11px] text-zinc-500 mt-1 block">Inference events</span>
            </div>

            {/* Detected Faces */}
            <div className="p-4 rounded-2xl bg-zinc-900/60 border border-zinc-800/80">
              <div className="flex items-center justify-between text-zinc-500 mb-2">
                <span className="text-[11px] font-bold uppercase tracking-wider">Faces</span>
                <Sparkles className="w-4 h-4 text-pink-400" />
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
              <span className="text-[11px] text-zinc-500 mt-1 block">Highest frequency</span>
            </div>

            {/* Average Confidence */}
            <div className="p-4 rounded-2xl bg-zinc-900/60 border border-zinc-800/80">
              <div className="flex items-center justify-between text-zinc-500 mb-2">
                <span className="text-[11px] font-bold uppercase tracking-wider">Confidence</span>
                <Activity className="w-4 h-4 text-emerald-400" />
              </div>
              <div className="text-2xl font-bold text-emerald-400 font-mono">
                {(analytics.average_confidence * 100).toFixed(1)}%
              </div>
              <span className="text-[11px] text-zinc-500 mt-1 block">Mean certainty</span>
            </div>

            {/* Detection Duration */}
            <div className="p-4 rounded-2xl bg-zinc-900/60 border border-zinc-800/80">
              <div className="flex items-center justify-between text-zinc-500 mb-2">
                <span className="text-[11px] font-bold uppercase tracking-wider">Duration</span>
                <Clock className="w-4 h-4 text-amber-400" />
              </div>
              <div className="text-2xl font-bold text-zinc-100 font-mono">
                {formatDuration(analytics.total_duration_seconds)}
              </div>
              <span className="text-[11px] text-zinc-500 mt-1 block">Total tracking time</span>
            </div>
          </div>

          {/* Primary Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <EmotionDistributionChart
              distribution={analytics.expression_distribution}
              title="Global Expression Distribution"
            />
            <ConfidenceChart
              confidenceAnalytics={analytics.confidence_analytics}
              title="Global Confidence Distribution"
            />
          </div>

          {/* Historical Activity Trend */}
          {analytics.recent_trends && analytics.recent_trends.length > 0 && (
            <div className="p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800/80">
              <div className="flex items-center justify-between pb-4 border-b border-zinc-800/60">
                <div>
                  <h3 className="text-base font-semibold text-zinc-100">Daily Activity Trend</h3>
                  <p className="text-xs text-zinc-400 mt-0.5">
                    Prediction and session throughput aggregated across recent active days.
                  </p>
                </div>
                <div className="flex items-center gap-4 text-xs font-medium text-zinc-400">
                  <div className="flex items-center gap-1.5">
                    <span className="w-3 h-3 rounded-full bg-indigo-500" />
                    <span>Predictions</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-3 h-3 rounded-full bg-pink-500" />
                    <span>Sessions</span>
                  </div>
                </div>
              </div>

              <div className="h-64 w-full pt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={analytics.recent_trends} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="trendPredGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.5} />
                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                      </linearGradient>
                      <linearGradient id="trendSessGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ec4899" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#ec4899" stopOpacity={0.0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                    <XAxis
                      dataKey="date"
                      stroke="#71717a"
                      fontSize={11}
                      tickLine={false}
                      axisLine={{ stroke: "#3f3f46" }}
                    />
                    <YAxis
                      stroke="#71717a"
                      fontSize={11}
                      tickLine={false}
                      axisLine={{ stroke: "#3f3f46" }}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const item = payload[0].payload;
                          return (
                            <div className="p-3 rounded-xl bg-zinc-950/95 border border-zinc-800 shadow-xl text-xs space-y-1">
                              <div className="font-semibold text-zinc-200 font-mono">{item.date}</div>
                              <div className="text-indigo-300">
                                Predictions: <span className="font-mono font-medium text-zinc-100">{item.predictions_count}</span>
                              </div>
                              <div className="text-pink-300">
                                Sessions: <span className="font-mono font-medium text-zinc-100">{item.sessions_count}</span>
                              </div>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="predictions_count"
                      stroke="#818cf8"
                      strokeWidth={2}
                      fill="url(#trendPredGrad)"
                    />
                    <Area
                      type="monotone"
                      dataKey="sessions_count"
                      stroke="#f472b6"
                      strokeWidth={2}
                      fill="url(#trendSessGrad)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Recent Sessions Quick Access */}
          <div className="p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800/80">
            <div className="flex items-center justify-between pb-4 border-b border-zinc-800/60">
              <div>
                <h3 className="text-base font-semibold text-zinc-100">Recent Sessions</h3>
                <p className="text-xs text-zinc-400 mt-0.5">
                  Direct links to inspect session timelines and detailed analytics.
                </p>
              </div>
              <Link
                href="/history"
                className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1 group transition-colors"
              >
                <span>View All History</span>
                <ArrowUpRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
              </Link>
            </div>

            {recentSessions.length === 0 ? (
              <div className="py-8 text-center text-sm text-zinc-500">
                No recent sessions found. Start a webcam session to record historical data.
              </div>
            ) : (
              <div className="divide-y divide-zinc-800/50 mt-2">
                {recentSessions.map((session) => {
                  const sDominantKey = (session.dominant_expression?.toLowerCase() || "neutral") as PredictionEmotion;
                  const sMeta = EMOTIONS[sDominantKey] || EMOTIONS.neutral;

                  return (
                    <div
                      key={session.id}
                      className="py-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 hover:bg-zinc-800/20 px-2 rounded-xl transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                        <div>
                          <Link
                            href={`/sessions/${session.id}`}
                            className="text-sm font-medium text-zinc-200 hover:text-indigo-400 transition-colors"
                          >
                            {session.name || `Session ${session.id.slice(0, 8)}`}
                          </Link>
                          <div className="text-xs text-zinc-500 font-mono mt-0.5">
                            {new Date(session.started_at).toLocaleString()} • {formatDuration(session.duration_seconds)}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-4 self-end sm:self-auto text-xs">
                        {session.dominant_expression && (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-zinc-800/80 border border-zinc-700/60 text-zinc-300">
                            <span>{sMeta.emoji}</span>
                            <span className="capitalize">{sMeta.label}</span>
                          </span>
                        )}
                        <span className="font-mono text-zinc-400">
                          {session.prediction_count.toLocaleString()} predictions
                        </span>
                        <Link
                          href={`/sessions/${session.id}`}
                          className="px-3 py-1 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 font-medium transition-colors"
                        >
                          View
                        </Link>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
