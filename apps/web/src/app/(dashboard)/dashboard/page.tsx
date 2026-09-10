"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  Camera,
  Cpu,
  Database,
  Image as ImageIcon,
  RefreshCw,
  Server,
  ShieldAlert,
  Sparkles,
  Zap,
} from "lucide-react";
import { getHealth, getReadiness, listSessions } from "@/lib/api/endpoints";
import { HealthResponse, ReadinessResponse, SessionResponse } from "@/types/api";
import { SUPPORTED_EMOTIONS, EMOTIONS } from "@/types/emotion";

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [sessions, setSessions] = useState<SessionResponse[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  const loadData = async () => {
    setIsRefreshing(true);
    setError(null);
    try {
      const [healthData, readyData, sessionData] = await Promise.allSettled([
        getHealth(),
        getReadiness(),
        listSessions(5, 0),
      ]);

      if (healthData.status === "fulfilled") {
        setHealth(healthData.value);
      }
      if (readyData.status === "fulfilled") {
        setReadiness(readyData.value);
      }
      if (sessionData.status === "fulfilled") {
        setSessions(sessionData.value.sessions);
      }

      if (healthData.status === "rejected" && readyData.status === "rejected") {
        setError("Backend API is currently unreachable. Make sure FastAPI is running on port 8000.");
      }
    } catch (err: unknown) {
      const errorObj = err as Error;
      setError(errorObj.message || "Failed to load dashboard metrics.");
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 20000);
    return () => clearInterval(interval);
  }, []);

  const isModelReady = readiness?.model === "ready";
  const isDbReady = readiness?.database === "ready" || health?.dependencies?.database?.status === "healthy";
  const isSystemOnline = health?.status === "ok" || readiness?.status === "ready";

  return (
    <div className="space-y-8">
      {/* Top Banner / System Status Hero */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-zinc-900 via-zinc-900/90 to-zinc-950 border border-zinc-800/80 p-6 md:p-8 shadow-xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl -mr-20 -mt-20 pointer-events-none" />
        
        <div className="relative flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/15 border border-indigo-500/30 text-xs font-semibold text-indigo-300">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Phase 12 Integrated Production System</span>
            </div>
            <h2 className="text-2xl md:text-3xl font-extrabold text-white tracking-tight">
              Facial Expression Emotion Detection AI
            </h2>
            <p className="text-sm text-zinc-400 max-w-2xl leading-relaxed">
              Real-time deep learning inference platform powered by ResNet-18 Champion and YuNet neural face detection.
              Trained across 7 human facial expression categories.
            </p>
          </div>

          <button
            onClick={loadData}
            disabled={isRefreshing}
            className="self-start md:self-auto flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-xl bg-zinc-800/80 hover:bg-zinc-700/80 text-zinc-200 border border-zinc-700/50 transition-all cursor-pointer disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? "animate-spin" : ""}`} />
            <span>{isRefreshing ? "Syncing..." : "Sync Status"}</span>
          </button>
        </div>

        {/* Live Status Cards Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-8">
          {/* Backend Status */}
          <div className="p-4 rounded-xl bg-zinc-950/60 border border-zinc-800/80 backdrop-blur-sm">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-zinc-400">FastAPI Backend</span>
              <Server className="w-4 h-4 text-zinc-500" />
            </div>
            <div className="flex items-center gap-2">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  isSystemOnline ? "bg-emerald-400 animate-pulse" : "bg-rose-500"
                }`}
              />
              <span className="text-lg font-bold text-zinc-100">
                {isSystemOnline ? "System Online" : "Service Offline"}
              </span>
            </div>
            <div className="text-[11px] text-zinc-500 mt-1 font-mono">
              v{health?.version || "1.0.0"} ({health?.environment || "development"})
            </div>
          </div>

          {/* Model Inference Status */}
          <div className="p-4 rounded-xl bg-zinc-950/60 border border-zinc-800/80 backdrop-blur-sm">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-zinc-400">ML Inference Engine</span>
              <Cpu className="w-4 h-4 text-zinc-500" />
            </div>
            <div className="flex items-center gap-2">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  isModelReady ? "bg-emerald-400" : "bg-amber-400 animate-ping"
                }`}
              />
              <span className="text-lg font-bold text-zinc-100">
                {isModelReady ? "Inference Ready" : "Warming Up..."}
              </span>
            </div>
            <div className="text-[11px] text-zinc-500 mt-1 font-mono">
              {readiness?.details?.model_version || "champion-pruning-30"} ({readiness?.details?.device || "cpu"})
            </div>
          </div>

          {/* Database Status */}
          <div className="p-4 rounded-xl bg-zinc-950/60 border border-zinc-800/80 backdrop-blur-sm">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-zinc-400">Session Database</span>
              <Database className="w-4 h-4 text-zinc-500" />
            </div>
            <div className="flex items-center gap-2">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  isDbReady ? "bg-emerald-400" : "bg-amber-400"
                }`}
              />
              <span className="text-lg font-bold text-zinc-100">
                {isDbReady ? "Connected" : "Disconnected"}
              </span>
            </div>
            <div className="text-[11px] text-zinc-500 mt-1 font-mono capitalize">
              {health?.dependencies?.database?.database || "Active"} ({health?.dependencies?.database?.host || "local"})
            </div>
          </div>

          {/* Real-Time WebSocket */}
          <div className="p-4 rounded-xl bg-zinc-950/60 border border-zinc-800/80 backdrop-blur-sm">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-zinc-400">WebSocket Stream</span>
              <Zap className="w-4 h-4 text-zinc-500" />
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
              <span className="text-lg font-bold text-zinc-100">Bidirectional</span>
            </div>
            <div className="text-[11px] text-zinc-500 mt-1 font-mono">
              /api/v1/realtime/emotion
            </div>
          </div>
        </div>
      </div>

      {/* Error Alert if API is down */}
      {error && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-rose-500/10 border border-rose-500/25 text-rose-400 text-sm">
          <ShieldAlert className="w-5 h-5 flex-shrink-0" />
          <div className="flex-1">
            <strong className="font-semibold">Connection Notice: </strong>
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Quick Actions Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Live Camera Action */}
        <Link
          href="/live"
          className="group relative flex flex-col justify-between p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800 hover:border-indigo-500/50 transition-all hover:shadow-xl hover:shadow-indigo-500/10"
        >
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 group-hover:scale-105 transition-transform">
                <Camera className="w-6 h-6" />
              </div>
              <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                Real-Time Streaming
              </span>
            </div>
            <h3 className="text-xl font-bold text-zinc-100 group-hover:text-indigo-300 transition-colors">
              Live Webcam Emotion Detection
            </h3>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Launch interactive webcam detection with real-time face bounding boxes, instantaneous and smoothed confidence metrics, and session recording.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs font-semibold text-indigo-400 mt-6 group-hover:translate-x-1 transition-transform">
            <span>Launch Live Feed</span>
            <ArrowRight className="w-4 h-4" />
          </div>
        </Link>

        {/* Image Upload Action */}
        <Link
          href="/image-analysis"
          className="group relative flex flex-col justify-between p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800 hover:border-purple-500/50 transition-all hover:shadow-xl hover:shadow-purple-500/10"
        >
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="p-3 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400 group-hover:scale-105 transition-transform">
                <ImageIcon className="w-6 h-6" />
              </div>
              <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-purple-500/10 text-purple-300 border border-purple-500/20">
                Single / Multi-Face
              </span>
            </div>
            <h3 className="text-xl font-bold text-zinc-100 group-hover:text-purple-300 transition-colors">
              Image Expression Analysis
            </h3>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Upload photographs to detect facial expressions, view complete 7-class probability distributions, face bounding coordinates, and inference latency telemetry.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs font-semibold text-purple-400 mt-6 group-hover:translate-x-1 transition-transform">
            <span>Analyze Image</span>
            <ArrowRight className="w-4 h-4" />
          </div>
        </Link>
      </div>

      {/* Model Specifications & Emotion Palette */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Supported Emotion Classes (7 cols) */}
        <div className="lg:col-span-7 p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-base font-bold text-zinc-200">Recognized Facial Expression Classes</h3>
            <span className="text-xs text-zinc-500 font-mono">7 Classes</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
            {SUPPORTED_EMOTIONS.map((emotion) => {
              const meta = EMOTIONS[emotion];
              return (
                <div
                  key={emotion}
                  className="p-3 rounded-xl bg-zinc-950/60 border border-zinc-800/80 flex flex-col justify-between"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xl">{meta.emoji}</span>
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: meta.color }}
                    />
                  </div>
                  <div className="font-semibold text-xs text-zinc-200">{meta.label}</div>
                  <div className="text-[10px] text-zinc-500 line-clamp-1 mt-0.5">
                    {meta.description}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Model Architecture Specs (5 cols) */}
        <div className="lg:col-span-5 p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-zinc-200">Model Architecture</h3>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Champion
            </span>
          </div>
          <div className="space-y-2.5 text-xs text-zinc-400">
            <div className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
              <span className="text-zinc-500">Backbone</span>
              <span className="font-medium text-zinc-200">ResNet-18 + CBAM Attention</span>
            </div>
            <div className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
              <span className="text-zinc-500">Face Detection</span>
              <span className="font-medium text-zinc-200">YuNet ONNX (640x480)</span>
            </div>
            <div className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
              <span className="text-zinc-500">Input Resolution</span>
              <span className="font-medium text-zinc-200">48 × 48 px (Grayscale)</span>
            </div>
            <div className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
              <span className="text-zinc-500">Optimization</span>
              <span className="font-medium text-zinc-200">30% Structured Pruning</span>
            </div>
            <div className="flex items-center justify-between py-1.5">
              <span className="text-zinc-500">Uncertainty Threshold</span>
              <span className="font-medium text-zinc-200">40.0% Confidence Cutoff</span>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Sessions List */}
      <div className="p-6 rounded-2xl bg-zinc-900/60 border border-zinc-800">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-bold text-zinc-200">Recent Analysis Sessions</h3>
            <p className="text-xs text-zinc-500 mt-0.5">Persisted session records from database</p>
          </div>
          <Link
            href="/live"
            className="text-xs font-medium text-indigo-400 hover:text-indigo-300 transition-colors"
          >
            Start New Session →
          </Link>
        </div>

        {sessions.length > 0 ? (
          <div className="divide-y divide-zinc-800/60 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-zinc-500 uppercase tracking-wider font-semibold border-b border-zinc-800/80">
                  <th className="py-2.5 px-3">Session Name / ID</th>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3">Started At</th>
                  <th className="py-2.5 px-3">Ended At</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/40 font-mono">
                {sessions.map((session) => (
                  <tr key={session.id} className="hover:bg-zinc-800/20 transition-colors">
                    <td className="py-3 px-3">
                      <span className="font-sans font-medium text-zinc-200 block">
                        {session.name || "Detection Session"}
                      </span>
                      <span className="text-[11px] text-zinc-500">{session.id}</span>
                    </td>
                    <td className="py-3 px-3">
                      <span
                        className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold uppercase ${
                          session.status === "active"
                            ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                            : "bg-zinc-800 text-zinc-400"
                        }`}
                      >
                        {session.status}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-zinc-400">
                      {new Date(session.started_at).toLocaleString()}
                    </td>
                    <td className="py-3 px-3 text-zinc-500">
                      {session.ended_at ? new Date(session.ended_at).toLocaleString() : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-8 text-center bg-zinc-950/40 rounded-xl border border-zinc-800/50">
            <Activity className="w-8 h-8 text-zinc-600 mx-auto mb-2" />
            <p className="text-xs text-zinc-400 font-medium">No recorded sessions yet.</p>
            <p className="text-[11px] text-zinc-500 mt-1">
              Start a live detection session or upload an image to begin tracking predictions.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
