"use client";

import React from "react";
import { HistoricalPrediction } from "@/types/analytics";
import { X } from "lucide-react";

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
  const rankedProbabilities = Object.entries(probabilities)
    .map(([emo, prob]) => ({
      emotion: emo,
      probability: Number(prob),
      percentage: (Number(prob) * 100).toFixed(1),
    }))
    .sort((a, b) => b.probability - a.probability);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-150">
      <div
        onClick={(e) => e.stopPropagation()}
        className="relative w-full max-w-2xl bg-[#0d0d0d] border border-[#262626] shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#262626] bg-[#000000]">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-[2px] text-[#999999]">
              INFERENCE TELEMETRY DOSSIER
            </div>
            <h3 className="font-display text-xl uppercase tracking-[2px] text-white">
              PREDICTION EXPLAINABILITY
            </h3>
            <p className="font-mono text-[10px] text-[#666666]">
              ID: {prediction.prediction_id}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 border border-[#3a3a3a] text-[#999999] hover:text-white hover:border-white transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Top Prediction Summary Card */}
          <div className="p-4 bg-[#141414] border border-[#262626] flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <span className="font-mono text-[10px] uppercase tracking-[2px] text-[#666666] block">
                CLASSIFIED AFFECT
              </span>
              <span className="font-display text-3xl uppercase tracking-[2.5px] text-white mt-0.5 block">
                {prediction.emotion}
              </span>
              <div className="font-mono text-xs mt-1">
                {prediction.is_uncertain ? (
                  <span className="text-amber-400">[!] LOW CERTAINTY THRESHOLD</span>
                ) : (
                  <span className="text-[#c3d9f3]">[CALIBRATED CONFIDENCE]</span>
                )}
              </div>
            </div>

            <div className="sm:text-right border-t sm:border-t-0 pt-3 sm:pt-0 border-[#262626]">
              <span className="font-mono text-[10px] uppercase tracking-[2px] text-[#666666] block">
                CONFIDENCE SCORE
              </span>
              <span className="font-mono text-3xl text-white">
                {(prediction.confidence * 100).toFixed(1)}
                <span className="text-sm text-[#666666]">%</span>
              </span>
            </div>
          </div>

          {/* Top-K Explainability Ranking */}
          <div className="space-y-3">
            <div className="flex items-center justify-between font-mono text-xs uppercase tracking-[2px] text-[#999999] pb-2 border-b border-[#262626]">
              <span>SOFTMAX PROBABILITY DENSITY</span>
              <span className="text-[#666666]">RANKED</span>
            </div>

            <div className="space-y-2">
              {rankedProbabilities.map((item, idx) => {
                const isTop = idx === 0;

                return (
                  <div
                    key={item.emotion}
                    className={`p-3 border transition-colors ${
                      isTop
                        ? "bg-[#1f1f1f] border-[#c3d9f3]"
                        : "bg-[#141414] border-[#262626]"
                    }`}
                  >
                    <div className="flex items-center justify-between font-mono text-xs uppercase tracking-[1.5px] mb-1.5">
                      <div className="flex items-center gap-2">
                        <span className="text-[#666666]">#{idx + 1}</span>
                        <span className={isTop ? "text-white" : "text-[#999999]"}>
                          {item.emotion}
                        </span>
                      </div>
                      <span className={isTop ? "text-[#c3d9f3]" : "text-[#999999]"}>
                        {item.percentage}%
                      </span>
                    </div>

                    <div className="w-full bg-[#1a1a1a] h-1 overflow-hidden">
                      <div
                        className={`h-full transition-all duration-300 ${
                          isTop ? "bg-[#c3d9f3]" : "bg-[#3a3a3a]"
                        }`}
                        style={{ width: `${item.percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Model Traceability & Technical Telemetry */}
          <div className="space-y-3">
            <div className="font-mono text-xs uppercase tracking-[2px] text-[#999999] pb-2 border-b border-[#262626]">
              TRACEABILITY & SPECIFICATION
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-px bg-[#262626] border border-[#262626] font-mono text-xs">
              <div className="p-3 bg-[#141414]">
                <span className="text-[10px] text-[#666666] uppercase tracking-[1.5px] block">
                  ARCHITECTURE
                </span>
                <span className="text-white">{prediction.model_version}</span>
              </div>
              <div className="p-3 bg-[#141414]">
                <span className="text-[10px] text-[#666666] uppercase tracking-[1.5px] block">
                  LATENCY
                </span>
                <span className="text-white">{prediction.processing_time_ms} MS</span>
              </div>
              <div className="p-3 bg-[#141414]">
                <span className="text-[10px] text-[#666666] uppercase tracking-[1.5px] block">
                  RETICLE COORDS
                </span>
                <span className="text-white">
                  [{prediction.bbox.x}, {prediction.bbox.y}, {prediction.bbox.width}×{prediction.bbox.height}]
                </span>
              </div>
              <div className="p-3 bg-[#141414]">
                <span className="text-[10px] text-[#666666] uppercase tracking-[1.5px] block">
                  TIMESTAMP
                </span>
                <span className="text-white truncate block">
                  {new Date(prediction.timestamp).toISOString()}
                </span>
              </div>
            </div>
          </div>

          {/* Ethical AI Notice */}
          <div className="p-3 bg-[#000000] border border-[#262626] font-serif text-xs text-[#666666] leading-relaxed">
            <span className="font-mono text-[10px] uppercase tracking-[1.5px] text-[#999999] block mb-1">
              ENGINEERING SPECIFICATION NOTICE
            </span>
            Likelihood values represent mathematical activations produced by deep neural geometry isolation. Values reflect statistical feature weights rather than subjective emotional reality.
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-[#262626] bg-[#000000] flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-5 py-1.5 rounded-full border border-white bg-transparent text-white font-mono text-xs uppercase tracking-[2px] hover:bg-white hover:text-black transition-all cursor-pointer"
          >
            DISMISS
          </button>
        </div>
      </div>
    </div>
  );
}

export default PredictionInspectModal;
