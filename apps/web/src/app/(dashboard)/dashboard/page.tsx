"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Camera,
  Cpu,
  Database,
  Image as ImageIcon,
  RefreshCw,
  Server,
  ShieldAlert,
  Video,
  Zap,
  ArrowRight,
  Trash2,
} from "lucide-react";
import { getHealth, getReadiness, listSessions } from "@/lib/api/endpoints";
import {
  getOrCreateUserId,
  getUserSessions,
  getUserSessionIds,
  deleteUserSession,
} from "@/lib/storage/user-storage";
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
      const userId = getOrCreateUserId();
      const localSessions = getUserSessions();
      const localSessionIds = getUserSessionIds();

      const [healthData, readyData, sessionData] = await Promise.allSettled([
        getHealth(),
        getReadiness(),
        listSessions(10, 0, userId),
      ]);

      if (healthData.status === "fulfilled") {
        setHealth(healthData.value);
      }
      if (readyData.status === "fulfilled") {
        setReadiness(readyData.value);
      }

      if (sessionData.status === "fulfilled") {
        // Filter strictly by this user's local session IDs or user_id matching
        const userFiltered = sessionData.value.sessions.filter(
          (s) => localSessionIds.has(s.id) || (s.user_id && s.user_id === userId)
        );

        if (userFiltered.length > 0) {
          setSessions(userFiltered);
        } else if (localSessions.length > 0) {
          // Fallback to local storage records if backend SQLite has no records yet
          setSessions(
            localSessions.map((ls) => ({
              id: ls.id,
              name: ls.name,
              user_id: userId,
              status: ls.status,
              started_at: ls.started_at,
              ended_at: ls.ended_at || null,
              created_at: ls.started_at,
            }))
          );
        } else {
          setSessions([]);
        }
      } else {
        // If backend listSessions fails (or in production without session DB), fallback to localStorage
        if (localSessions.length > 0) {
          setSessions(
            localSessions.map((ls) => ({
              id: ls.id,
              name: ls.name,
              user_id: userId,
              status: ls.status,
              started_at: ls.started_at,
              ended_at: ls.ended_at || null,
              created_at: ls.started_at,
            }))
          );
        } else {
          setSessions([]);
        }
      }

      if (healthData.status === "rejected" && readyData.status === "rejected") {
        setError("Backend API is currently unreachable. Confirm FastAPI is operational on port 8000.");
      }
    } catch (err: unknown) {
      const errorObj = err as Error;
      setError(errorObj.message || "Failed to load telemetry metrics.");
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleDeleteSession = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    deleteUserSession(sessionId);
    setSessions((prev) => prev.filter((s) => s.id !== sessionId));
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
    <div className="space-y-8 bg-black text-[#cccccc]">
      {/* 1. Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-[#222222]">
        <div>
          <h1 className="font-display text-2xl sm:text-3xl uppercase tracking-[2px] text-white">
            Overview
          </h1>
          <p className="font-mono text-xs text-[#888888] tracking-[1px] mt-1">
            Real-time emotion AI metrics, pipeline status, and detection services
          </p>
        </div>

        <button
          onClick={loadData}
          disabled={isRefreshing}
          className="inline-flex items-center gap-2 h-9 px-4 rounded-full border border-[#262626] hover:border-[#444444] bg-[#111111] hover:bg-[#1a1a1a] text-[#cccccc] hover:text-white font-mono text-[11px] uppercase tracking-[1.5px] transition-all cursor-pointer self-start sm:self-auto disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? "animate-spin text-emerald-400" : "text-[#888888]"}`} />
          <span>{isRefreshing ? "Syncing..." : "Refresh"}</span>
        </button>
      </div>

      {/* 2. Error Notice */}
      {error && (
        <div className="p-4 bg-[#141414] border border-red-500/40 text-red-400 font-mono text-xs uppercase tracking-[1px] rounded-none flex items-center gap-3">
          <ShieldAlert className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* 3. Core Status Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Backend Node */}
        <div className="p-5 bg-[#0e0e0e] border border-[#222222] hover:border-[#333333] rounded-none space-y-3 transition-colors">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[10px] uppercase tracking-[2px] text-[#777777]">
              API SERVER
            </span>
            <div className="w-7 h-7 rounded-none bg-[#161616] border border-[#262626] flex items-center justify-center text-[#888888]">
              <Server className="w-3.5 h-3.5" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isSystemOnline ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]" : "bg-red-500"}`} />
              <span className={`font-display text-xl uppercase tracking-[1px] ${isSystemOnline ? "text-emerald-400" : "text-red-400"}`}>
                {isSystemOnline ? "ONLINE" : "OFFLINE"}
              </span>
            </div>
            <div className="font-mono text-[10px] uppercase tracking-[1px] text-[#666666] mt-1">
              v{health?.version || "1.0.0"} · FASTAPI
            </div>
          </div>
        </div>

        {/* Inference Core */}
        <div className="p-5 bg-[#0e0e0e] border border-[#222222] hover:border-[#333333] rounded-none space-y-3 transition-colors">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[10px] uppercase tracking-[2px] text-[#777777]">
              AI MODEL
            </span>
            <div className="w-7 h-7 rounded-none bg-[#161616] border border-[#262626] flex items-center justify-center text-[#888888]">
              <Cpu className="w-3.5 h-3.5" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isModelReady ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]" : "bg-amber-400 animate-pulse"}`} />
              <span className={`font-display text-xl uppercase tracking-[1px] ${isModelReady ? "text-white" : "text-amber-400"}`}>
                {isModelReady ? "READY & ACTIVE" : "WARMING UP"}
              </span>
            </div>
            <div className="font-mono text-[10px] uppercase tracking-[1px] text-[#888888] mt-1">
              RESNET-18 CBAM · YUNET
            </div>
          </div>
        </div>

        {/* Database */}
        <div className="p-5 bg-[#0e0e0e] border border-[#222222] hover:border-[#333333] rounded-none space-y-3 transition-colors">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[10px] uppercase tracking-[2px] text-[#777777]">
              DATABASE
            </span>
            <div className="w-7 h-7 rounded-none bg-[#161616] border border-[#262626] flex items-center justify-center text-[#888888]">
              <Database className="w-3.5 h-3.5" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isDbReady ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]" : "bg-amber-400"}`} />
              <span className={`font-display text-xl uppercase tracking-[1px] ${isDbReady ? "text-white" : "text-amber-400"}`}>
                {isDbReady ? "CONNECTED" : "DISCONNECTED"}
              </span>
            </div>
            <div className="font-mono text-[10px] uppercase tracking-[1px] text-[#666666] mt-1">
              SQLITE WAL MODE
            </div>
          </div>
        </div>

        {/* Real-time Stream */}
        <div className="p-5 bg-[#0e0e0e] border border-[#222222] hover:border-[#333333] rounded-none space-y-3 transition-colors">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[10px] uppercase tracking-[2px] text-[#777777]">
              STREAMING
            </span>
            <div className="w-7 h-7 rounded-none bg-[#161616] border border-[#262626] flex items-center justify-center text-[#888888]">
              <Zap className="w-3.5 h-3.5" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]" />
              <span className="font-display text-xl uppercase tracking-[1px] text-white">
                WEBSOCKET
              </span>
            </div>
            <div className="font-mono text-[10px] uppercase tracking-[1px] text-emerald-400 mt-1">
              REAL-TIME DUPLEX
            </div>
          </div>
        </div>
      </div>

      {/* 4. Action Feature Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Live Camera Card */}
        <div className="bg-[#0e0e0e] border border-[#222222] hover:border-[#333333] rounded-none p-6 flex flex-col justify-between group transition-all">
          <div className="space-y-4 mb-6">
            <div className="flex items-center justify-between">
              <div className="w-9 h-9 rounded-none bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                <Camera className="w-4 h-4" />
              </div>
              <span className="font-mono text-[10px] uppercase tracking-[1px] text-emerald-400 px-2 py-0.5 border border-emerald-500/30 bg-emerald-950/20 rounded-none">
                REAL-TIME
              </span>
            </div>
            <div>
              <h2 className="font-display text-xl uppercase tracking-[1.5px] text-white">
                Live Webcam Detection
              </h2>
              <p className="font-sans text-xs text-[#888888] leading-relaxed mt-2">
                Detect and follow emotions through your camera in real time with smooth face tracking, emotion color tags, and live confidence meters.
              </p>
            </div>
          </div>
          <Link
            href="/live"
            className="btn-valence self-start"
          >
            <span>Start Live Camera</span>
            <ArrowRight className="w-3.5 h-3.5 ml-2 inline" />
          </Link>
        </div>

        {/* Image Analysis Card */}
        <div className="bg-[#0e0e0e] border border-[#222222] hover:border-[#333333] rounded-none p-6 flex flex-col justify-between group transition-all">
          <div className="space-y-4 mb-6">
            <div className="flex items-center justify-between">
              <div className="w-9 h-9 rounded-none bg-[#c3d9f3]/10 border border-[#c3d9f3]/20 flex items-center justify-center text-[#c3d9f3]">
                <ImageIcon className="w-4 h-4" />
              </div>
              <span className="font-mono text-[10px] uppercase tracking-[1px] text-[#cccccc] px-2 py-0.5 border border-[#2a2a2a] bg-[#161616] rounded-none">
                UPLOAD
              </span>
            </div>
            <div>
              <h2 className="font-display text-xl uppercase tracking-[1.5px] text-white">
                Image Emotion Analysis
              </h2>
              <p className="font-sans text-xs text-[#888888] leading-relaxed mt-2">
                Upload photos to detect all faces with colored bounding boxes and inspect full 7-class probability distributions for each person.
              </p>
            </div>
          </div>
          <Link
            href="/image-analysis"
            className="btn-valence-secondary self-start"
          >
            <span>Analyze Image</span>
            <ArrowRight className="w-3.5 h-3.5 ml-2 inline" />
          </Link>
        </div>

        {/* Video Analysis Card */}
        <div className="bg-[#0e0e0e] border border-[#222222] hover:border-[#333333] rounded-none p-6 flex flex-col justify-between group transition-all">
          <div className="space-y-4 mb-6">
            <div className="flex items-center justify-between">
              <div className="w-9 h-9 rounded-none bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
                <Video className="w-4 h-4" />
              </div>
              <span className="font-mono text-[10px] uppercase tracking-[1px] text-amber-400 px-2 py-0.5 border border-amber-500/30 bg-amber-950/20 rounded-none">
                TIMELINE
              </span>
            </div>
            <div>
              <h2 className="font-display text-xl uppercase tracking-[1.5px] text-white">
                Video File Analysis
              </h2>
              <p className="font-sans text-xs text-[#888888] leading-relaxed mt-2">
                Upload video files to track facial expressions over time with an interactive colorful timeline scrubber and emotion shift summary.
              </p>
            </div>
          </div>
          <Link
            href="/video-analysis"
            className="btn-valence-secondary self-start"
          >
            <span>Analyze Video</span>
            <ArrowRight className="w-3.5 h-3.5 ml-2 inline" />
          </Link>
        </div>
      </div>

      {/* 5. Supported Emotion Classes Matrix */}
      <div className="bg-[#0c0c0c] border border-[#222222] rounded-none p-6 sm:p-7">
        <div className="flex items-center justify-between mb-6 pb-4 border-b border-[#1f1f1f]">
          <div className="flex items-center gap-2.5">
            <div className="w-2 h-2 bg-emerald-400" />
            <h2 className="font-display text-lg uppercase tracking-[2px] text-white">
              Supported Emotions
            </h2>
          </div>
          <span className="font-mono text-[10px] uppercase tracking-[1.5px] text-[#777777]">
            7 Target Classes
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
          {SUPPORTED_EMOTIONS.map((emotion) => {
            const meta = EMOTIONS[emotion];
            return (
              <div
                key={emotion}
                className="p-4 bg-[#121212] border border-[#222222] hover:border-[#333333] rounded-none flex flex-col justify-between space-y-3 transition-all duration-200 relative overflow-hidden group"
              >
                {/* Subtle top color hairline */}
                <div
                  className="absolute top-0 inset-x-0 h-0.5 transition-all group-hover:h-1"
                  style={{ backgroundColor: meta.color }}
                />

                {/* Emoji badge with sharp tinted container */}
                <div
                  className="w-10 h-10 rounded-none flex items-center justify-center text-2xl transition-transform group-hover:scale-110 select-none"
                  style={{
                    backgroundColor: `${meta.color}15`,
                    border: `1px solid ${meta.color}30`,
                  }}
                >
                  <span>{meta.emoji}</span>
                </div>

                <div>
                  <div className="font-display text-sm uppercase tracking-[1px] text-white">
                    {meta.label}
                  </div>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <span
                      className="w-1.5 h-1.5"
                      style={{ backgroundColor: meta.color }}
                    />
                    <span
                      className="font-mono text-[10px] uppercase tracking-[1px]"
                      style={{ color: meta.color }}
                    >
                      {emotion}
                    </span>
                  </div>
                </div>

                <div className="font-sans text-[11px] text-[#777777] leading-relaxed line-clamp-2">
                  {meta.description}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 6. Recent Sessions Table */}
      <div className="bg-[#0c0c0c] border border-[#222222] rounded-none p-6 sm:p-7">
        <div className="flex items-center justify-between mb-6 pb-4 border-b border-[#1f1f1f]">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-display text-lg uppercase tracking-[2px] text-white">
                Recent Sessions
              </h2>
              <span className="px-1.5 py-0.5 border border-[#333333] bg-[#141414] text-[9px] uppercase tracking-[1px] text-[#888888] font-mono">
                USER ISOLATED
              </span>
            </div>
            <div className="font-mono text-[10px] uppercase tracking-[1.5px] text-[#777777] mt-0.5">
              Live Detection Sessions & Recordings
            </div>
          </div>
        </div>

        {sessions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead>
                <tr className="border-b border-[#222222] text-[#666666] uppercase text-[10px] tracking-[1.5px]">
                  <th className="py-2.5 px-4">Session Name</th>
                  <th className="py-2.5 px-4">Status</th>
                  <th className="py-2.5 px-4">Started At</th>
                  <th className="py-2.5 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1a1a1a]">
                {sessions.map((session) => (
                  <tr key={session.id} className="hover:bg-[#141414] transition-colors">
                    <td className="py-3 px-4 text-white">
                      <div className="font-medium">{session.name || "Unnamed Session"}</div>
                      <div className="text-[10px] text-[#666666] font-mono">{session.id}</div>
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`inline-flex items-center gap-1.5 px-2 py-0.5 text-[10px] uppercase tracking-[0.5px] border rounded-none ${
                          session.status === "active"
                            ? "border-emerald-500/30 text-emerald-400 bg-emerald-950/20"
                            : "border-[#2a2a2a] text-[#888888] bg-[#141414]"
                        }`}
                      >
                        <span
                          className={`w-1.5 h-1.5 ${
                            session.status === "active" ? "bg-emerald-400" : "bg-[#666666]"
                          }`}
                        />
                        {session.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-[#888888]">
                      {new Date(session.started_at).toLocaleString()}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <Link
                          href={`/sessions/${session.id}`}
                          className="inline-flex items-center h-7 px-3 border border-[#333333] hover:border-white text-[#cccccc] hover:text-white font-mono text-[10px] uppercase tracking-[1px] transition-all rounded-none"
                        >
                          Details
                        </Link>
                        <button
                          type="button"
                          onClick={(e) => handleDeleteSession(session.id, e)}
                          className="p-1.5 border border-[#262626] hover:border-red-500/50 bg-[#111111] hover:bg-red-950/20 text-[#666666] hover:text-red-400 transition-colors cursor-pointer rounded-none"
                          title="Remove session record"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-10 text-center text-[#666666] font-mono text-xs uppercase tracking-[2px] flex flex-col items-center justify-center gap-2">
            <Camera className="w-5 h-5 text-[#444444]" />
            <span>No sessions recorded for your account yet</span>
            <span className="text-[10px] lowercase text-[#555555] tracking-normal font-sans">
              start a live camera session to record your detection history
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
