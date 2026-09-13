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

export interface ConfidenceChartProps {
  confidenceAnalytics: ConfidenceAnalytics;
  className?: string;
  title?: string;
}

const MONO_SHADES = ["#444444", "#666666", "#999999", "#c3d9f3", "#ffffff"];

export function ConfidenceChart({
  confidenceAnalytics,
  className = "",
  title = "MODEL CONFIDENCE INTERVALS",
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

  const histogramData = distribution.map((b) => ({
    name: b.range,
    count: b.count,
    percentage: b.percentage,
  }));

  const classData = Object.entries(class_confidences).map(([emo, avg]) => {
    return {
      name: emo.toUpperCase(),
      emotion: emo,
      avgPercent: Math.round(avg * 100),
      avgScore: avg,
    };
  });

  const totalEvaluated = distribution.reduce((sum, b) => sum + b.count, 0);

  if (totalEvaluated === 0) {
    return (
      <div
        className={`p-6 bg-[#0d0d0d] border border-[#262626] flex flex-col items-center justify-center min-h-[320px] text-center ${className}`}
      >
        <div className="font-mono text-xs text-[#666666] mb-2">[NO TELEMETRY]</div>
        <h3 className="font-display text-lg uppercase tracking-[2px] text-white">NO CONFIDENCE DATA</h3>
        <p className="font-serif text-sm text-[#999999] max-w-sm mt-1">
          Confidence distributions will populate once inference events are recorded into the database.
        </p>
      </div>
    );
  }

  return (
    <div className={`p-6 bg-[#0d0d0d] border border-[#262626] ${className}`}>
      {/* Header & Tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-[#262626]">
        <div>
          <div className="flex items-center gap-3">
            <h3 className="font-display text-lg uppercase tracking-[2px] text-white">{title}</h3>
            <span className="font-mono text-[10px] uppercase tracking-[1.5px] text-[#c3d9f3] bg-[#141414] px-2 py-0.5 border border-[#3a3a3a]">
              MEAN: {(average_confidence * 100).toFixed(1)}%
            </span>
          </div>
          <p className="font-serif text-xs text-[#999999] mt-1">
            Softmax certainty intervals across {totalEvaluated.toLocaleString()} evaluations.
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 bg-[#141414] border border-[#262626] self-start sm:self-auto font-mono text-[10px]">
          <button
            type="button"
            onClick={() => setActiveTab("distribution")}
            className={`px-3 py-1 transition-colors cursor-pointer ${
              activeTab === "distribution"
                ? "bg-[#1f1f1f] text-white border border-white"
                : "text-[#666666] hover:text-white"
            }`}
          >
            HISTOGRAM
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("classes")}
            className={`px-3 py-1 transition-colors cursor-pointer ${
              activeTab === "classes"
                ? "bg-[#1f1f1f] text-white border border-white"
                : "text-[#666666] hover:text-white"
            }`}
          >
            PER CLASS
          </button>
        </div>
      </div>

      {/* KPI Highlights Bar (4 Spec Cells in Hairline Grid) */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-px bg-[#262626] border border-[#262626] my-5 font-mono">
        <div className="p-3 bg-[#141414]">
          <span className="text-[10px] text-[#666666] uppercase tracking-[1.5px] block">
            MEAN SCORE
          </span>
          <span className="text-xl text-white mt-1 block">
            {(average_confidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="p-3 bg-[#141414]">
          <span className="text-[10px] text-[#666666] uppercase tracking-[1.5px] block">
            RANGE BOUNDS
          </span>
          <span className="text-xl text-white mt-1 block">
            {(min_confidence * 100).toFixed(0)}% – {(max_confidence * 100).toFixed(0)}%
          </span>
        </div>
        <div className="p-3 bg-[#141414]">
          <span className="text-[10px] text-[#666666] uppercase tracking-[1.5px] block">
            HIGH (&gt;=80%)
          </span>
          <span className="text-xl text-[#c3d9f3] mt-1 block">
            {high_confidence_count.toLocaleString()}
          </span>
        </div>
        <div className="p-3 bg-[#141414]">
          <span className="text-[10px] text-[#666666] uppercase tracking-[1.5px] block">
            LOW (&lt;60%)
          </span>
          <span className="text-xl text-[#999999] mt-1 block">
            {low_confidence_count.toLocaleString()}
          </span>
        </div>
      </div>

      {/* Chart Canvas */}
      <div className="h-60 w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          {activeTab === "distribution" ? (
            <BarChart data={histogramData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
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
                        <div className="text-white tracking-wider font-bold">INTERVAL: {data.name}</div>
                        <div className="text-[#999999]">COUNT: <span className="text-white">{data.count.toLocaleString()}</span></div>
                        <div className="text-[#c3d9f3]">SHARE: <span className="text-white">{data.percentage}%</span></div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="count">
                {histogramData.map((_, index) => {
                  return <Cell key={`hist-${index}`} fill={MONO_SHADES[index % MONO_SHADES.length]} />;
                })}
              </Bar>
            </BarChart>
          ) : (
            <BarChart data={classData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
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
                        <div className="text-white tracking-wider font-bold">{data.name}</div>
                        <div className="text-[#c3d9f3]">MEAN CONFIDENCE: <span className="text-white">{data.avgPercent}%</span></div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="avgPercent">
                {classData.map((_, index) => (
                  <Cell key={`cls-${index}`} fill={index % 2 === 0 ? "#ffffff" : "#c3d9f3"} />
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
