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
import { SUPPORTED_EMOTIONS, EMOTIONS, EmotionType } from "@/types/emotion";

export interface EmotionTimelineChartProps {
  timeline: TimelineAnalytics;
  className?: string;
  title?: string;
}

export function EmotionTimelineChart({
  timeline,
  className = "",
  title = "Emotion Timeline",
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

    const domKey = (b.dominant_emotion || "neutral").toLowerCase() as EmotionType;
    const domMeta = EMOTIONS[domKey];

    return {
      timeLabel,
      relativeSeconds: b.relative_seconds,
      count: b.prediction_count,
      confidencePercent: Math.round(b.average_confidence * 100),
      dominantLabel: domMeta ? `${domMeta.emoji} ${domMeta.label}` : (b.dominant_emotion || "NEUTRAL").toUpperCase(),
      dominantColor: domMeta?.color || "#38bdf8",
      ...emotionFields,
    };
  });

  if (buckets.length === 0) {
    return (
      <div
        className={`p-6 bg-[#0d0d0d] border border-[#262626] flex flex-col items-center justify-center min-h-[300px] text-center ${className}`}
      >
        <div className="font-mono text-xs text-[#666666] mb-2">[NO DATA]</div>
        <h3 className="font-display text-lg uppercase tracking-[2px] text-white">No Timeline Data</h3>
        <p className="font-serif text-sm text-[#999999] max-w-sm mt-1">
          Predictions recorded over multiple seconds will construct this timeline graph.
        </p>
      </div>
    );
  }

  return (
    <div className={`p-6 bg-[#0d0d0d] border border-[#262626] ${className}`}>
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-[#262626]">
        <div>
          <div className="flex items-center gap-3">
            <h3 className="font-display text-lg uppercase tracking-[2px] text-white">{title}</h3>
            <span className="font-mono text-[10px] uppercase tracking-[1.5px] text-[#c3d9f3] bg-[#141414] px-2 py-0.5 border border-[#3a3a3a]">
              {bucket_seconds}s intervals
            </span>
          </div>
          <p className="font-serif text-xs text-[#999999] mt-1">
            Emotion progression across {total_buckets} time window{total_buckets !== 1 ? "s" : ""}.
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 bg-[#141414] border border-[#262626] self-start sm:self-auto font-mono text-[10px]">
          <button
            type="button"
            onClick={() => setActiveMetric("stacked")}
            className={`px-3 py-1 transition-colors cursor-pointer ${
              activeMetric === "stacked"
                ? "bg-[#1f1f1f] text-white border border-white"
                : "text-[#666666] hover:text-white"
            }`}
          >
            Emotion Flow
          </button>
          <button
            type="button"
            onClick={() => setActiveMetric("confidence")}
            className={`px-3 py-1 transition-colors cursor-pointer ${
              activeMetric === "confidence"
                ? "bg-[#1f1f1f] text-white border border-white"
                : "text-[#666666] hover:text-white"
            }`}
          >
            Confidence Curve
          </button>
        </div>
      </div>

      {/* Chart Canvas */}
      <div className="h-64 w-full pt-4">
        <ResponsiveContainer width="100%" height="100%">
          {activeMetric === "stacked" ? (
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="2 2" stroke="#262626" vertical={false} />
              <XAxis
                dataKey="timeLabel"
                stroke="#666666"
                fontSize={10}
                fontFamily="monospace"
                tickLine={false}
                axisLine={{ stroke: "#262626" }}
              />
              <YAxis
                stroke="#666666"
                fontSize={10}
                fontFamily="monospace"
                tickLine={false}
                axisLine={{ stroke: "#262626" }}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="p-3 bg-black border border-[#3a3a3a] text-xs font-mono space-y-1">
                        <div className="flex items-center justify-between gap-4 font-bold text-white">
                          <span>+{data.timeLabel}</span>
                          <span className="text-[#c3d9f3]">{data.dominantLabel}</span>
                        </div>
                        <div className="pt-1 border-t border-[#262626] text-[#999999] space-y-0.5">
                          <div>EVALUATIONS: <span className="text-white">{data.count}</span></div>
                          <div>AVG CERTAINTY: <span className="text-white">{data.confidencePercent}%</span></div>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              {SUPPORTED_EMOTIONS.map((emo) => {
                const meta = EMOTIONS[emo as EmotionType];
                const color = meta?.color || "rgb(56, 189, 248)";
                return (
                  <Area
                    key={emo}
                    type="monotone"
                    dataKey={emo}
                    name={meta ? `${meta.emoji} ${meta.label}` : emo}
                    stackId="1"
                    stroke={color}
                    fill={color}
                    fillOpacity={0.45}
                  />
                );
              })}
            </AreaChart>
          ) : (
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="confidenceGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#c3d9f3" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#c3d9f3" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="2 2" stroke="#262626" vertical={false} />
              <XAxis
                dataKey="timeLabel"
                stroke="#666666"
                fontSize={10}
                fontFamily="monospace"
                tickLine={false}
                axisLine={{ stroke: "#262626" }}
              />
              <YAxis
                stroke="#666666"
                fontSize={10}
                fontFamily="monospace"
                domain={[0, 100]}
                tickFormatter={(val) => `${val}%`}
                tickLine={false}
                axisLine={{ stroke: "#262626" }}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="p-3 bg-black border border-[#3a3a3a] text-xs font-mono space-y-1">
                        <div className="font-bold text-white tracking-wider">
                          +{data.timeLabel}
                        </div>
                        <div className="text-emerald-400">
                          Confidence: <span className="text-white">{data.confidencePercent}%</span>
                        </div>
                        <div className="text-[#999999]">
                          Dominant: <span className="text-white font-medium">{data.dominantLabel}</span>
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
                stroke="#c3d9f3"
                strokeWidth={1.5}
                fill="url(#confidenceGrad)"
              />
            </AreaChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Mini Legend */}
      <div className="flex flex-wrap items-center gap-4 pt-3 border-t border-[#262626] font-mono text-[10px] text-[#999999] uppercase">
        {SUPPORTED_EMOTIONS.map((emo) => {
          const meta = EMOTIONS[emo as EmotionType];
          const color = meta?.color || "rgb(56, 189, 248)";
          return (
            <div key={emo} className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
              <span>{meta?.emoji} {meta?.label || emo}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default EmotionTimelineChart;
