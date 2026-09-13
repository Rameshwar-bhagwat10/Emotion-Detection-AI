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
import { ExpressionDistribution } from "@/types/analytics";
import { EMOTIONS, EmotionType } from "@/types/emotion";

export interface EmotionDistributionChartProps {
  distribution: ExpressionDistribution;
  className?: string;
  title?: string;
}

export function EmotionDistributionChart({
  distribution,
  className = "",
  title = "Emotion Distribution",
}: EmotionDistributionChartProps) {
  const [viewMode, setViewMode] = useState<"donut" | "bar">("donut");

  const total = distribution.total_predictions;
  const items = distribution.items || [];

  const chartData = items.map((item) => {
    const emoKey = item.emotion.toLowerCase() as EmotionType;
    const meta = EMOTIONS[emoKey];
    return {
      name: meta ? `${meta.emoji} ${meta.label}` : item.emotion.toUpperCase(),
      label: meta ? meta.label : item.emotion,
      emoji: meta?.emoji || "",
      emotionKey: item.emotion,
      count: item.count,
      percentage: item.percentage,
      color: meta?.color || "rgb(56, 189, 248)",
    };
  });

  if (total === 0) {
    return (
      <div
        className={`p-6 bg-[#0d0d0d] border border-[#262626] flex flex-col items-center justify-center min-h-[320px] text-center ${className}`}
      >
        <div className="font-mono text-xs text-[#666666] mb-2">[NO DATA]</div>
        <h3 className="font-display text-lg uppercase tracking-[2px] text-white">No Predictions Yet</h3>
        <p className="font-serif text-sm text-[#999999] max-w-sm mt-1">
          Start a live camera session or analyze an image/video to see the breakdown of detected emotions.
        </p>
      </div>
    );
  }

  return (
    <div className={`p-6 bg-[#0d0d0d] border border-[#262626] ${className}`}>
      {/* Header & View Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-[#262626]">
        <div>
          <div className="flex items-center gap-3">
            <h3 className="font-display text-lg uppercase tracking-[2px] text-white">{title}</h3>
            {distribution.dominant_emotion && (
              <span className="font-mono text-[10px] uppercase tracking-[1.5px] text-emerald-400 bg-[#141414] px-2 py-0.5 border border-emerald-500/30">
                Dominant: {EMOTIONS[distribution.dominant_emotion.toLowerCase() as EmotionType]?.emoji || ""} {distribution.dominant_emotion.toUpperCase()}
              </span>
            )}
          </div>
          <p className="font-serif text-xs text-[#999999] mt-1">
            Breakdown across {total.toLocaleString()} detected emotion events.
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 bg-[#141414] border border-[#262626] self-start sm:self-auto font-mono text-[10px]">
          <button
            type="button"
            onClick={() => setViewMode("donut")}
            className={`px-3 py-1 transition-colors cursor-pointer ${
              viewMode === "donut"
                ? "bg-[#1f1f1f] text-white border border-white"
                : "text-[#666666] hover:text-white"
            }`}
          >
            Donut
          </button>
          <button
            type="button"
            onClick={() => setViewMode("bar")}
            className={`px-3 py-1 transition-colors cursor-pointer ${
              viewMode === "bar"
                ? "bg-[#1f1f1f] text-white border border-white"
                : "text-[#666666] hover:text-white"
            }`}
          >
            Bar
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
                  outerRadius={90}
                  paddingAngle={2}
                  dataKey="count"
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} stroke="#000000" strokeWidth={1.5} />
                  ))}
                </Pie>
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="p-3 bg-black border border-[#3a3a3a] text-xs font-mono space-y-1">
                          <div className="text-white tracking-wider font-bold">{data.name}</div>
                          <div className="text-[#999999]">COUNT: <span className="text-white">{data.count.toLocaleString()}</span></div>
                          <div className="text-[#c3d9f3]">SHARE: <span className="text-white">{data.percentage}%</span></div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
              </PieChart>
            ) : (
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="2 2" stroke="#262626" vertical={false} />
                <XAxis
                  dataKey="name"
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
                          <div className="text-white tracking-wider font-bold">{data.name}</div>
                          <div className="text-[#999999]">COUNT: <span className="text-white">{data.count.toLocaleString()}</span></div>
                          <div className="text-[#c3d9f3]">SHARE: <span className="text-white">{data.percentage}%</span></div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="count">
                  {chartData.map((entry, index) => (
                    <Cell key={`bar-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            )}
          </ResponsiveContainer>
        </div>

        {/* Breakdown List */}
        <div className="lg:col-span-5 space-y-1.5 max-h-64 overflow-y-auto pr-1">
          {chartData.map((entry) => (
            <div
              key={entry.emotionKey}
              className="flex items-center justify-between p-2 bg-[#141414] border border-[#262626] font-mono text-xs"
            >
              <div className="flex items-center gap-2.5">
                <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: entry.color }} />
                <span className="text-white font-medium">{entry.name}</span>
              </div>
              <div className="text-right flex items-center gap-3">
                <span className="text-[#666666]">
                  {entry.count.toLocaleString()}
                </span>
                <span className="font-bold w-12 text-right" style={{ color: entry.color }}>
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
