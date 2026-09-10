"use client";

import React, { useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from "recharts";
import { ConfidenceAnalytics } from "@/types/analytics";
import { EMOTIONS, PredictionEmotion } from "@/types/emotion";
import { Gauge, ShieldAlert, ShieldCheck, BarChart3 } from "lucide-react";

export interface ConfidenceChartProps {
  confidenceAnalytics: ConfidenceAnalytics;
  className?: string;
  title?: string;
}

export function ConfidenceChart({
  confidenceAnalytics,
  className = "",
  title = "Model Confidence Analysis",
}: ConfidenceChartProps) {
  const [activeTab, setActiveTab] = useState<"distribution" | "classes">("distribution");

  const {
    average_confidence,
    min_confidence,
    max_confidence,
    distribution = [],
    class_confidences = {},
    low_confidence_count,
    high_confidence_count,
  } = confidenceAnalytics;

  // Histogram data
  const histogramData = distribution.map((b) => ({
    name: b.range,
    count: b.count,
    percentage: b.percentage,
  }));

  // Class confidences data
  const classData = Object.entries(class_confidences).map(([emo, avg]) => {
    const meta = EMOTIONS[emo.toLowerCase() as PredictionEmotion] || EMOTIONS.neutral;
    return {
      name: meta.label,
      emotion: emo,
      avgPercent: Math.round(avg * 100),
      avgScore: avg,
      color: meta.color,
      emoji: meta.emoji,
    };
  });

  const totalEvaluated = distribution.reduce((sum, b) => sum + b.count, 0);

  if (totalEvaluated === 0) {
    return (
      <div
        className={`p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800/80 flex flex-col items-center justify-center min-h-[320px] text-center ${className}`}
      >
        <div className="w-12 h-12 rounded-xl bg-zinc-800/50 flex items-center justify-center text-zinc-500 mb-3">
          <Gauge className="w-6 h-6" />
        </div>
        <h3 className="text-base font-semibold text-zinc-300">No Confidence Telemetry</h3>
        <p className="text-sm text-zinc-500 max-w-sm mt-1">
          Confidence distributions will populate once predictions are recorded in the database.
        </p>
      </div>
    );
  }

  return (
    <div className={`p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800/80 ${className}`}>
      {/* Header & Tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-5 border-b border-zinc-800/60">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-zinc-100">{title}</h3>
            <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
              Avg: {(average_confidence * 100).toFixed(1)}%
            </span>
          </div>
          <p className="text-xs text-zinc-400 mt-1">
            Softmax certainty intervals across {totalEvaluated.toLocaleString()} predictions.
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 rounded-xl bg-zinc-950 border border-zinc-800 self-start sm:self-auto">
          <button
            onClick={() => setActiveTab("distribution")}
            className={`p-1.5 px-3 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors ${
              activeTab === "distribution"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Histogram</span>
          </button>
          <button
            onClick={() => setActiveTab("classes")}
            className={`p-1.5 px-3 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors ${
              activeTab === "classes"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            <Gauge className="w-3.5 h-3.5" />
            <span>Per Class</span>
          </button>
        </div>
      </div>

      {/* KPI Highlights Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 my-5">
        <div className="p-3 rounded-xl bg-zinc-950/60 border border-zinc-800/60">
          <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider block">
            Average Score
          </span>
          <span className="text-lg font-bold text-zinc-100 font-mono mt-0.5 block">
            {(average_confidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="p-3 rounded-xl bg-zinc-950/60 border border-zinc-800/60">
          <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider block">
            Score Range
          </span>
          <span className="text-lg font-bold text-zinc-100 font-mono mt-0.5 block">
            {(min_confidence * 100).toFixed(0)}% – {(max_confidence * 100).toFixed(0)}%
          </span>
        </div>
        <div className="p-3 rounded-xl bg-zinc-950/60 border border-zinc-800/60">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">
              High (≥80%)
            </span>
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <span className="text-lg font-bold text-emerald-400 font-mono mt-0.5 block">
            {high_confidence_count.toLocaleString()}
          </span>
        </div>
        <div className="p-3 rounded-xl bg-zinc-950/60 border border-zinc-800/60">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">
              Low (&lt;60%)
            </span>
            <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
          </div>
          <span className="text-lg font-bold text-amber-400 font-mono mt-0.5 block">
            {low_confidence_count.toLocaleString()}
          </span>
        </div>
      </div>

      {/* Chart Canvas */}
      <div className="h-60 w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          {activeTab === "distribution" ? (
            <BarChart data={histogramData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
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
                        <div className="font-semibold text-zinc-200">
                          Range: <span className="font-mono text-indigo-400">{data.name}</span>
                        </div>
                        <div className="mt-1.5 space-y-0.5 text-zinc-400">
                          <div>Predictions: <span className="text-zinc-200 font-mono font-medium">{data.count.toLocaleString()}</span></div>
                          <div>Proportion: <span className="text-zinc-200 font-mono font-medium">{data.percentage}%</span></div>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                {histogramData.map((entry, index) => {
                  const colors = ["#ef4444", "#f97316", "#eab308", "#3b82f6", "#10b981"];
                  return <Cell key={`hist-${index}`} fill={colors[index % colors.length]} />;
                })}
              </Bar>
            </BarChart>
          ) : (
            <BarChart data={classData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
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
                      <div className="p-3 rounded-xl bg-zinc-950/95 border border-zinc-800 shadow-xl text-xs">
                        <div className="flex items-center gap-2 font-semibold text-zinc-200">
                          <span>{data.emoji}</span>
                          <span>{data.name}</span>
                        </div>
                        <div className="mt-1.5 space-y-0.5 text-zinc-400">
                          <div>Avg Confidence: <span className="text-zinc-200 font-mono font-medium">{data.avgPercent}%</span></div>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="avgPercent" radius={[6, 6, 0, 0]}>
                {classData.map((entry, index) => (
                  <Cell key={`cls-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default ConfidenceChart;
