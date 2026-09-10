"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  Camera,
  Image as ImageIcon,
  Layers,
  Sparkles,
} from "lucide-react";
import { getHealth, getReadiness } from "@/lib/api/endpoints";

export default function HomePage() {
  const [isBackendOnline, setIsBackendOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const probe = async () => {
      try {
        const [health, ready] = await Promise.allSettled([getHealth(), getReadiness()]);
        setIsBackendOnline(
          health.status === "fulfilled" &&
            (health.value.status === "ok" || health.value.status === "degraded")
        );
      } catch {
        setIsBackendOnline(false);
      }
    };
    probe();
  }, []);

  return (
    <main className="min-h-screen bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950 text-zinc-100 flex flex-col justify-between p-6 md:p-12 relative overflow-hidden">
      {/* Background Glows */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-indigo-500/10 rounded-full blur-[140px] pointer-events-none" />
      <div className="absolute bottom-10 right-10 w-[400px] h-[400px] bg-purple-500/10 rounded-full blur-[120px] pointer-events-none" />

      {/* Top Navigation Bar */}
      <header className="max-w-6xl w-full mx-auto flex items-center justify-between py-4 relative z-10">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 via-purple-500 to-pink-500 shadow-lg shadow-indigo-500/25">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="font-extrabold text-lg text-white tracking-tight">
              Emotion<span className="text-indigo-400">AI</span>
            </span>
            <span className="block text-[10px] uppercase font-semibold text-zinc-500 tracking-wider">
              Phase 12 Production
            </span>
          </div>
        </div>

        {/* System Status Pill */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-zinc-900/80 border border-zinc-800 text-xs font-medium">
          <span
            className={`w-2 h-2 rounded-full ${
              isBackendOnline === true
                ? "bg-emerald-400 animate-pulse"
                : isBackendOnline === false
                ? "bg-rose-500"
                : "bg-amber-400 animate-ping"
            }`}
          />
          <span className="text-zinc-300">
            {isBackendOnline === true
              ? "Backend Online"
              : isBackendOnline === false
              ? "Backend Offline"
              : "Checking System..."}
          </span>
        </div>
      </header>

      {/* Hero Section */}
      <div className="max-w-4xl w-full mx-auto my-auto text-center space-y-8 relative z-10 py-12">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-xs font-semibold text-indigo-300 shadow-sm">
          <Layers className="w-3.5 h-3.5 text-indigo-400" />
          <span>ResNet-18 Champion + YuNet Neural Face Detector</span>
        </div>

        <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-white max-w-3xl mx-auto leading-tight">
          Real-Time Facial Expression Classification
        </h1>

        <p className="text-base sm:text-lg text-zinc-400 max-w-2xl mx-auto leading-relaxed">
          High-performance, production-ready AI application for detecting and classifying human facial expressions in images and live webcam video with calibrated probabilities across 7 emotion categories.
        </p>

        {/* Primary Action Launcher Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 max-w-2xl mx-auto pt-6 text-left">
          {/* Live Camera Option */}
          <Link
            href="/live"
            className="group relative p-6 rounded-2xl bg-zinc-900/80 hover:bg-zinc-900 border border-zinc-800 hover:border-indigo-500/50 transition-all duration-200 shadow-xl hover:shadow-indigo-500/10"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 group-hover:scale-105 transition-transform">
                <Camera className="w-6 h-6" />
              </div>
              <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                Webcam Feed
              </span>
            </div>
            <h3 className="text-lg font-bold text-zinc-100 group-hover:text-indigo-300 transition-colors">
              Live Webcam Detection
            </h3>
            <p className="text-xs text-zinc-400 mt-1.5 leading-relaxed">
              Stream live video through low-latency WebSockets with bounding box overlays and session tracking.
            </p>
            <div className="flex items-center gap-2 text-xs font-semibold text-indigo-400 mt-4 group-hover:translate-x-1 transition-transform">
              <span>Start Camera</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </div>
          </Link>

          {/* Image Analysis Option */}
          <Link
            href="/image-analysis"
            className="group relative p-6 rounded-2xl bg-zinc-900/80 hover:bg-zinc-900 border border-zinc-800 hover:border-purple-500/50 transition-all duration-200 shadow-xl hover:shadow-purple-500/10"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400 group-hover:scale-105 transition-transform">
                <ImageIcon className="w-6 h-6" />
              </div>
              <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30">
                File Upload
              </span>
            </div>
            <h3 className="text-lg font-bold text-zinc-100 group-hover:text-purple-300 transition-colors">
              Image Expression Analysis
            </h3>
            <p className="text-xs text-zinc-400 mt-1.5 leading-relaxed">
              Upload photographs to detect multiple faces, inspect probability distributions, and latency telemetry.
            </p>
            <div className="flex items-center gap-2 text-xs font-semibold text-purple-400 mt-4 group-hover:translate-x-1 transition-transform">
              <span>Analyze Image</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </div>
          </Link>
        </div>

        {/* Dashboard Link */}
        <div className="pt-2">
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-2 text-xs font-semibold text-zinc-400 hover:text-zinc-200 transition-colors"
          >
            <span>Open System Dashboard & Telemetry</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>

      {/* Footer Info */}
      <footer className="max-w-6xl w-full mx-auto pt-6 border-t border-zinc-800/80 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-zinc-500 relative z-10">
        <div className="flex items-center gap-4">
          <span>Architecture: FastAPI + Next.js + PyTorch</span>
          <span>•</span>
          <span>Model: ResNet-18 (30% Pruning)</span>
        </div>
        <div>
          <span>Phase 12 — Complete Application Integration</span>
        </div>
      </footer>
    </main>
  );
}
