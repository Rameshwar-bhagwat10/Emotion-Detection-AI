"use client";

import React, { useState } from "react";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import { EMOTIONS, PredictionEmotion } from "@/types/emotion";
import { ExpressionDistribution } from "@/types/analytics";
import { PieChart as PieIcon, BarChart2 } from "lucide-react";

export interface EmotionDistributionChartProps {
  distribution: ExpressionDistribution;
  className?: string;
  title?: string;
}

export function EmotionDistributionChart({
  distribution,
  className = "",
  title = "Predicted Expression Distribution",
}: EmotionDistributionChartProps) {
  const [viewMode, setViewMode] = useState<"donut" | "bar">("donut");

  const total = distribution.total_predictions;
  const items = distribution.items || [];

  const chartData = items.map((item) => {
    const emotionKey = (item.emotion.toLowerCase() in EMOTIONS
      ? item.emotion.toLowerCase()
      : "uncertain") as PredictionEmotion;
    const meta = EMOTIONS[emotionKey];
    return {
      name: meta.label,
      emotionKey: item.emotion,
      count: item.count,
      percentage: item.percentage,
      color: meta.color,
      emoji: meta.emoji,
    };
  });

  const dominantKey = (distribution.dominant_emotion?.toLowerCase() || "neutral") as PredictionEmotion;
  const dominantMeta = EMOTIONS[dominantKey] || EMOTIONS.neutral;

  if (total === 0) {
    return (
      <div
        className={`p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800/80 flex flex-col items-center justify-center min-h-[320px] text-center ${className}`}
      >
        <div className="w-12 h-12 rounded-xl bg-zinc-800/50 flex items-center justify-center text-zinc-500 mb-3">
          <PieIcon className="w-6 h-6" />
        </div>
        <h3 className="text-base font-semibold text-zinc-300">No Predictions Available</h3>
        <p className="text-sm text-zinc-500 max-w-sm mt-1">
          Perform image analysis or start a live webcam detection session to generate expression distribution data.
        </p>
      </div>
    );
  }

  return (
    <div className={`p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800/80 ${className}`}>
      {/* Header & View Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-5 border-b border-zinc-800/60">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-zinc-100">{title}</h3>
            {distribution.dominant_emotion && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
                <span>Dominant:</span>
                <span>{dominantMeta.emoji}</span>
                <span className="capitalize">{dominantMeta.label}</span>
              </span>
            )}
          </div>
          <p className="text-xs text-zinc-400 mt-1">
            Proportional frequency across {total.toLocaleString()} valid face detection events.
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 rounded-xl bg-zinc-950 border border-zinc-800 self-start sm:self-auto">
          <button
            onClick={() => setViewMode("donut")}
            className={`p-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors ${
              viewMode === "donut"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
            title="Donut Chart"
          >
            <PieIcon className="w-4 h-4" />
            <span className="hidden md:inline">Donut</span>
          </button>
          <button
            onClick={() => setViewMode("bar")}
            className={`p-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors ${
              viewMode === "bar"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
            title="Bar Chart"
          >
            <BarChart2 className="w-4 h-4" />
            <span className="hidden md:inline">Bar</span>
          </button>
        </div>
      </div>

      {/* Chart Canvas & Legend */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center pt-6">
        <div className="lg:col-span-7 h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            {viewMode === "donut" ? (
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={95}
                  paddingAngle={3}
                  dataKey="count"
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} stroke="rgba(24, 24, 27, 0.8)" strokeWidth={2} />
                  ))}
                </Pie>
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="p-3 rounded-xl bg-zinc-950/95 border border-zinc-800 shadow-xl text-xs">
                          <div className="flex items-center gap-2 font-semibold text-zinc-200">
                            <span>{data.emoji}</span>
                            <span>{data.name}</span>
                          </div>
                          <div className="mt-1.5 space-y-0.5 text-zinc-400">
                            <div>Count: <span className="text-zinc-200 font-mono font-medium">{data.count.toLocaleString()}</span></div>
                            <div>Share: <span className="text-zinc-200 font-mono font-medium">{data.percentage}%</span></div>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
              </PieChart>
            ) : (
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                <XAxis
                  dataKey="name"
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
                        <div className="p-3 rounded-xl bg-zinc-950/95 border border-zinc-800 shadow-xl text-xs">
                          <div className="flex items-center gap-2 font-semibold text-zinc-200">
                            <span>{data.emoji}</span>
                            <span>{data.name}</span>
                          </div>
                          <div className="mt-1.5 space-y-0.5 text-zinc-400">
                            <div>Count: <span className="text-zinc-200 font-mono font-medium">{data.count.toLocaleString()}</span></div>
                            <div>Share: <span className="text-zinc-200 font-mono font-medium">{data.percentage}%</span></div>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell key={`bar-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            )}
          </ResponsiveContainer>
        </div>

        {/* Breakdown List */}
        <div className="lg:col-span-5 space-y-2 max-h-64 overflow-y-auto pr-1">
          {chartData.map((entry) => (
            <div
              key={entry.emotionKey}
              className="flex items-center justify-between p-2 rounded-xl bg-zinc-950/40 border border-zinc-800/40 hover:border-zinc-700 transition-colors"
            >
              <div className="flex items-center gap-2.5">
                <span className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }} />
                <span className="text-base">{entry.emoji}</span>
                <span className="text-xs font-medium text-zinc-300">{entry.name}</span>
              </div>
              <div className="text-right flex items-center gap-3">
                <span className="text-xs text-zinc-400 font-mono">
                  {entry.count.toLocaleString()}
                </span>
                <span className="text-xs font-semibold text-zinc-200 font-mono w-12 text-right">
                  {entry.percentage.toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default EmotionDistributionChart;
