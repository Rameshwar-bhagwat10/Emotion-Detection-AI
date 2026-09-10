"use client";

import React, { useState } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import { TimelineAnalytics } from "@/types/analytics";
import { EMOTIONS, SUPPORTED_EMOTIONS, PredictionEmotion } from "@/types/emotion";
import { Clock, Activity, Layers } from "lucide-react";

export interface EmotionTimelineChartProps {
  timeline: TimelineAnalytics;
  className?: string;
  title?: string;
}

export function EmotionTimelineChart({
  timeline,
  className = "",
  title = "Expression Progression Timeline",
}: EmotionTimelineChartProps) {
  const [activeMetric, setActiveMetric] = useState<"stacked" | "confidence">("stacked");

  const { buckets = [], bucket_seconds, total_buckets } = timeline;

  const chartData = buckets.map((b) => {
    const min = Math.floor(b.relative_seconds / 60);
    const sec = Math.floor(b.relative_seconds % 60);
    const timeLabel = `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;

    const emotionFields: Record<string, number> = {};
    for (const emo of SUPPORTED_EMOTIONS) {
      emotionFields[emo] = b.emotion_counts[emo] || 0;
    }

    const domKey = (b.dominant_emotion?.toLowerCase() || "neutral") as PredictionEmotion;
    const domMeta = EMOTIONS[domKey] || EMOTIONS.neutral;

    return {
      timeLabel,
      relativeSeconds: b.relative_seconds,
      count: b.prediction_count,
      confidencePercent: Math.round(b.average_confidence * 100),
      dominantLabel: domMeta.label,
      dominantEmoji: domMeta.emoji,
      dominantColor: domMeta.color,
      ...emotionFields,
    };
  });

  if (buckets.length === 0) {
    return (
      <div
        className={`p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800/80 flex flex-col items-center justify-center min-h-[300px] text-center ${className}`}
      >
        <div className="w-12 h-12 rounded-xl bg-zinc-800/50 flex items-center justify-center text-zinc-500 mb-3">
          <Clock className="w-6 h-6" />
        </div>
        <h3 className="text-base font-semibold text-zinc-300">No Timeline Data</h3>
        <p className="text-sm text-zinc-500 max-w-sm mt-1">
          Predictions recorded over multiple time intervals will form a chronological progression chart.
        </p>
      </div>
    );
  }

  return (
    <div className={`p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800/80 ${className}`}>
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-zinc-800/60">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-zinc-100">{title}</h3>
            <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-zinc-800 text-zinc-300 border border-zinc-700">
              {bucket_seconds}s intervals
            </span>
          </div>
          <p className="text-xs text-zinc-400 mt-1">
            Chronological classification flow across {total_buckets} temporal window{total_buckets !== 1 ? "s" : ""}.
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 rounded-xl bg-zinc-950 border border-zinc-800 self-start sm:self-auto">
          <button
            onClick={() => setActiveMetric("stacked")}
            className={`p-1.5 px-3 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors ${
              activeMetric === "stacked"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Expressions</span>
          </button>
          <button
            onClick={() => setActiveMetric("confidence")}
            className={`p-1.5 px-3 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors ${
              activeMetric === "confidence"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Confidence Curve</span>
          </button>
        </div>
      </div>

      {/* Chart Canvas */}
      <div className="h-64 w-full pt-4">
        <ResponsiveContainer width="100%" height="100%">
          {activeMetric === "stacked" ? (
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
              <XAxis
                dataKey="timeLabel"
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
                    const data = payload[0].payload;
                    return (
                      <div className="p-3 rounded-xl bg-zinc-950/95 border border-zinc-800 shadow-xl text-xs space-y-2">
                        <div className="flex items-center justify-between gap-4 font-semibold text-zinc-200">
                          <span className="font-mono text-zinc-400">+{data.timeLabel}</span>
                          <div className="flex items-center gap-1 text-indigo-300">
                            <span>{data.dominantEmoji}</span>
                            <span>{data.dominantLabel}</span>
                          </div>
                        </div>
                        <div className="pt-1 border-t border-zinc-800/80 space-y-1 text-zinc-400">
                          <div>Window Predictions: <span className="font-mono text-zinc-200">{data.count}</span></div>
                          <div>Avg Confidence: <span className="font-mono text-zinc-200">{data.confidencePercent}%</span></div>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              {SUPPORTED_EMOTIONS.map((emo) => {
                const meta = EMOTIONS[emo as PredictionEmotion];
                return (
                  <Area
                    key={emo}
                    type="monotone"
                    dataKey={emo}
                    stackId="1"
                    stroke={meta.color}
                    fill={meta.color}
                    fillOpacity={0.4}
                  />
                );
              })}
            </AreaChart>
          ) : (
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="confidenceGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.6} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
              <XAxis
                dataKey="timeLabel"
                stroke="#71717a"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: "#3f3f46" }}
              />
              <YAxis
                stroke="#71717a"
                fontSize={11}
                domain={[0, 100]}
                tickFormatter={(val) => `${val}%`}
                tickLine={false}
                axisLine={{ stroke: "#3f3f46" }}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="p-3 rounded-xl bg-zinc-950/95 border border-zinc-800 shadow-xl text-xs space-y-1.5">
                        <div className="font-semibold text-zinc-200 font-mono">
                          Time: +{data.timeLabel}
                        </div>
                        <div className="text-indigo-400 font-medium">
                          Confidence: <span className="font-mono text-zinc-100">{data.confidencePercent}%</span>
                        </div>
                        <div className="text-zinc-400">
                          Dominant: <span className="text-zinc-200">{data.dominantEmoji} {data.dominantLabel}</span>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Area
                type="monotone"
                dataKey="confidencePercent"
                stroke="#818cf8"
                strokeWidth={2}
                fill="url(#confidenceGrad)"
              />
            </AreaChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Mini Legend */}
      <div className="flex flex-wrap items-center gap-3 pt-4 mt-2 border-t border-zinc-800/40 text-[11px] text-zinc-400">
        {SUPPORTED_EMOTIONS.map((emo) => {
          const meta = EMOTIONS[emo as PredictionEmotion];
          return (
            <div key={emo} className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: meta.color }} />
              <span>{meta.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default EmotionTimelineChart;
