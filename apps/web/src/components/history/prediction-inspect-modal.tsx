"use client";

import React from "react";
import { HistoricalPrediction } from "@/types/analytics";
import { EMOTIONS, PredictionEmotion } from "@/types/emotion";
import {
  X,
  Cpu,
  Clock,
  Layers,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  Maximize2,
} from "lucide-react";

export interface PredictionInspectModalProps {
  prediction: HistoricalPrediction | null;
  isOpen: boolean;
  onClose: () => void;
}

export function PredictionInspectModal({
  prediction,
  isOpen,
  onClose,
}: PredictionInspectModalProps) {
  if (!isOpen || !prediction) return null;

  const probabilities = prediction.probabilities || {};
  // Sort probabilities to generate Top-K explainability ranking
  const rankedProbabilities = Object.entries(probabilities)
    .map(([emo, prob]) => ({
      emotion: emo,
      probability: Number(prob),
      percentage: (Number(prob) * 100).toFixed(1),
    }))
    .sort((a, b) => b.probability - a.probability);

  const topEmotionKey = (prediction.emotion.toLowerCase() in EMOTIONS
    ? prediction.emotion.toLowerCase()
    : "uncertain") as PredictionEmotion;
  const topMeta = EMOTIONS[topEmotionKey] || EMOTIONS.neutral;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-200">
      <div
        onClick={(e) => e.stopPropagation()}
        className="relative w-full max-w-2xl bg-zinc-950 border border-zinc-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 bg-zinc-900/50">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-500/15 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-zinc-100">
                Prediction Details & Explainability
              </h3>
              <p className="text-xs text-zinc-400 font-mono">
                ID: {prediction.prediction_id}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-xl text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Top Prediction Summary Card */}
          <div className="p-4 rounded-2xl bg-zinc-900/60 border border-zinc-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <span className="text-4xl">{topMeta.emoji}</span>
              <div>
                <span className="text-xs text-zinc-400 font-medium uppercase tracking-wider block">
                  Classified Expression
                </span>
                <span className="text-xl font-bold text-zinc-100 capitalize">
                  {topMeta.label}
                </span>
                <div className="flex items-center gap-2 mt-1">
                  {prediction.is_uncertain ? (
                    <span className="inline-flex items-center gap-1 text-xs text-amber-400 font-medium">
                      <AlertTriangle className="w-3.5 h-3.5" />
                      Low Confidence / Uncertain
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-xs text-emerald-400 font-medium">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      Calibrated Confidence
                    </span>
                  )}
                </div>
              </div>
            </div>

            <div className="sm:text-right border-t sm:border-t-0 pt-3 sm:pt-0 border-zinc-800">
              <span className="text-xs text-zinc-400 font-medium uppercase tracking-wider block">
                Confidence
              </span>
              <span className="text-2xl font-bold text-indigo-400 font-mono">
                {(prediction.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Top-K Explainability Ranking */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-400" />
                <span>Softmax Probability Distribution (Top-K)</span>
              </h4>
              <span className="text-xs text-zinc-500">Ranked by likelihood</span>
            </div>

            <div className="space-y-2.5">
              {rankedProbabilities.map((item, idx) => {
                const emoKey = (item.emotion.toLowerCase() in EMOTIONS
                  ? item.emotion.toLowerCase()
                  : "uncertain") as PredictionEmotion;
                const meta = EMOTIONS[emoKey] || EMOTIONS.neutral;
                const isTop = idx === 0;

                return (
                  <div
                    key={item.emotion}
                    className={`p-2.5 rounded-xl border transition-colors ${
                      isTop
                        ? "bg-indigo-950/20 border-indigo-500/40"
                        : "bg-zinc-900/40 border-zinc-800/60"
                    }`}
                  >
                    <div className="flex items-center justify-between text-xs mb-1.5">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-zinc-500 w-4">#{idx + 1}</span>
                        <span className="text-sm">{meta.emoji}</span>
                        <span className={`font-medium capitalize ${isTop ? "text-indigo-300" : "text-zinc-300"}`}>
                          {meta.label}
                        </span>
                      </div>
                      <span className="font-mono font-medium text-zinc-200">
                        {item.percentage}%
                      </span>
                    </div>

                    <div className="w-full bg-zinc-800/80 rounded-full h-1.5 overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${item.percentage}%`,
                          backgroundColor: meta.color,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Model Traceability & Technical Telemetry */}
          <div>
            <h4 className="text-sm font-semibold text-zinc-200 mb-3 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-purple-400" />
              <span>Model Traceability & Metadata</span>
            </h4>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-3 rounded-xl bg-zinc-900/40 border border-zinc-800/60 space-y-1">
                <span className="text-zinc-500 block">Model Version</span>
                <span className="font-mono font-medium text-zinc-200">{prediction.model_version}</span>
              </div>
              <div className="p-3 rounded-xl bg-zinc-900/40 border border-zinc-800/60 space-y-1">
                <span className="text-zinc-500 block">Pipeline Latency</span>
                <span className="font-mono font-medium text-zinc-200 flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-zinc-400" />
                  {prediction.processing_time_ms} ms
                </span>
              </div>
              <div className="p-3 rounded-xl bg-zinc-900/40 border border-zinc-800/60 space-y-1">
                <span className="text-zinc-500 block">Face Bounding Box</span>
                <span className="font-mono font-medium text-zinc-200 flex items-center gap-1.5">
                  <Maximize2 className="w-3.5 h-3.5 text-zinc-400" />
                  [{prediction.bbox.x}, {prediction.bbox.y}, {prediction.bbox.width}×{prediction.bbox.height}]
                </span>
              </div>
              <div className="p-3 rounded-xl bg-zinc-900/40 border border-zinc-800/60 space-y-1">
                <span className="text-zinc-500 block">Timestamp</span>
                <span className="font-mono font-medium text-zinc-200">
                  {new Date(prediction.timestamp).toLocaleString()}
                </span>
              </div>
            </div>
          </div>

          {/* Ethical AI & Explainability Notice */}
          <div className="p-3 rounded-xl bg-zinc-900/30 border border-zinc-800/50 text-[11px] text-zinc-500 leading-relaxed">
            <span className="font-semibold text-zinc-400 block mb-0.5">Ethical AI Notice</span>
            This inspection presents statistical softmax likelihoods produced by visual facial geometry classification.
            Scores reflect model feature attribution and should not be construed as clinical diagnostic or definitive subjective emotional determinations.
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-zinc-800 bg-zinc-900/30 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-sm font-medium bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export default PredictionInspectModal;
