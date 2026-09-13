"use client";

import React, { useState, useMemo } from "react";
import { Play, Clock, Sparkles, Layers } from "lucide-react";
import { ExpressionEvent, ExpressionSegment, VideoTrack } from "@/types/video-analysis";
import { EMOTIONS, PredictionEmotion } from "@/types/emotion";

export interface EmotionTimelineProps {
  durationSeconds: number;
  segments: ExpressionSegment[];
  events?: ExpressionEvent[];
  tracks: VideoTrack[];
  currentTime: number;
  onSeek: (seconds: number) => void;
  className?: string;
}

const EMOTION_COLORS: Record<string, string> = {
  happy: "rgb(16, 185, 129)",    // Emerald
  neutral: "rgb(56, 189, 248)",  // Sky Blue
  surprise: "rgb(245, 158, 11)", // Amber
  sad: "rgb(99, 102, 241)",      // Indigo
  fear: "rgb(168, 85, 247)",     // Purple
  angry: "rgb(244, 63, 94)",     // Rose / Red
  disgust: "rgb(20, 184, 166)",  // Teal
  uncertain: "rgb(161, 161, 170)",
};

export function EmotionTimeline({
  durationSeconds,
  segments,
  tracks,
  currentTime,
  onSeek,
  className = "",
}: EmotionTimelineProps) {
  const [selectedTrackId, setSelectedTrackId] = useState<number | "all">("all");

  const duration = Math.max(1.0, durationSeconds);

  // Filter segments by selected track
  const filteredSegments = useMemo(() => {
    if (selectedTrackId === "all") return segments;
    return segments.filter((s) => s.track_id === selectedTrackId);
  }, [segments, selectedTrackId]);

  // Group segments by track for multi-track rendering
  const tracksGrouped = useMemo(() => {
    const map: Record<number, ExpressionSegment[]> = {};
    segments.forEach((s) => {
      map[s.track_id] = map[s.track_id] || [];
      map[s.track_id].push(s);
    });
    return map;
  }, [segments]);

  // Identify currently active segment at currentTime
  const activeSegment = useMemo(() => {
    if (filteredSegments.length === 0) return null;
    return (
      filteredSegments.find(
        (s) => currentTime >= s.start_time && currentTime <= s.end_time
      ) || filteredSegments[0]
    );
  }, [filteredSegments, currentTime]);

  const activeMeta = activeSegment
    ? EMOTIONS[activeSegment.emotion.toLowerCase() as PredictionEmotion] || EMOTIONS.neutral
    : EMOTIONS.neutral;

  const formatTimestamp = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 10);
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}.${ms}`;
  };

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickPercent = Math.max(0, Math.min(1, clickX / rect.width));
    const seekTarget = clickPercent * duration;
    onSeek(seekTarget);
  };

  // Generate ruler time ticks
  const ticks = useMemo(() => {
    const tickCount = Math.min(8, Math.max(3, Math.floor(duration / 3)));
    const step = duration / tickCount;
    return Array.from({ length: tickCount + 1 }, (_, i) => i * step);
  }, [duration]);

  return (
    <div className={`p-6 bg-[#0d0d0d] border border-[#262626] rounded-none space-y-6 flex flex-col justify-between h-full ${className}`}>
      {/* Header & Track Selector Tabs */}
      <div>
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-[#262626]">
          <div>
            <div className="font-mono text-xs uppercase tracking-[2px] text-[#999999] mb-1 flex items-center gap-2">
              <Clock className="w-3.5 h-3.5 text-[#c3d9f3]" />
              <span>VIDEO TIMELINE</span>
              <span className="text-[#3a3a3a]">/</span>
              <span className="text-[#c3d9f3]">EMOTION SHIFTS</span>
            </div>
            <h3 className="font-display text-2xl uppercase tracking-[2px] text-white">
              CHRONOLOGICAL EMOTION TIMELINE
            </h3>
            <p className="font-sans text-xs text-[#999999] mt-1">
              Click any color segment or timestamp to jump to that moment in the video.
            </p>
          </div>

          {/* Multi-Face Track Tabs */}
          {tracks && tracks.length > 1 && (
            <div className="flex items-center gap-1 bg-[#141414] p-1 border border-[#262626] rounded-none">
              <button
                type="button"
                onClick={() => setSelectedTrackId("all")}
                className={`px-3 py-1 font-mono text-[10px] uppercase tracking-[1.5px] transition-all cursor-pointer rounded-none ${
                  selectedTrackId === "all"
                    ? "bg-[#1f1f1f] text-white border border-white"
                    : "text-[#666666] hover:text-white"
                }`}
              >
                ALL FACES ({tracks.length})
              </button>
              {tracks.map((t) => (
                <button
                  key={t.track_id}
                  type="button"
                  onClick={() => setSelectedTrackId(t.track_id)}
                  className={`px-3 py-1 font-mono text-[10px] uppercase tracking-[1.5px] transition-all cursor-pointer rounded-none ${
                    selectedTrackId === t.track_id
                      ? "bg-[#1f1f1f] text-[#c3d9f3] border border-[#c3d9f3]"
                      : "text-[#666666] hover:text-white"
                  }`}
                >
                  FACE {t.track_id}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Timeline Visualization Container */}
        <div className="space-y-2 mt-5">
          {/* Time Ruler */}
          <div className="relative h-4 w-full font-mono text-[10px] text-[#666666] select-none">
            {ticks.map((t, idx) => {
              const leftPct = (t / duration) * 100;
              return (
                <span
                  key={idx}
                  style={{ left: `${Math.min(95, Math.max(0, leftPct))}%` }}
                  className="absolute -translate-x-1/2"
                >
                  {formatTimestamp(t)}
                </span>
              );
            })}
          </div>

          {/* Track Bar(s) */}
          <div className="space-y-3">
            {selectedTrackId === "all" ? (
              Object.entries(tracksGrouped).map(([trackStr, segs]) => {
                const trackId = parseInt(trackStr);
                return (
                  <div key={trackId} className="space-y-1">
                    <div className="flex items-center justify-between font-mono text-[10px] text-[#999999] uppercase tracking-wider">
                      <span className="flex items-center gap-1.5">
                        <Layers className="w-3 h-3 text-[#c3d9f3]" />
                        <span>FACE #{trackId}</span>
                      </span>
                      <span>{segs.length} EMOTION SEGMENTS</span>
                    </div>

                    {/* Render segmented bar for this track */}
                    <div
                      onClick={handleTimelineClick}
                      className="relative h-9 w-full bg-[#141414] border border-[#262626] cursor-pointer hover:border-[#3a3a3a] transition-colors rounded-none overflow-hidden"
                    >
                      {segs.map((seg) => {
                        const widthPct = Math.max(0.5, (seg.duration / duration) * 100);
                        const leftPct = (seg.start_time / duration) * 100;
                        const isSegActive = currentTime >= seg.start_time && currentTime <= seg.end_time;
                        const emotionKey = seg.emotion.toLowerCase();
                        const meta = EMOTIONS[emotionKey as PredictionEmotion] || EMOTIONS.neutral;
                        const bg = EMOTION_COLORS[emotionKey] || "rgb(56, 189, 248)";

                        return (
                          <div
                            key={seg.id}
                            style={{
                              left: `${leftPct}%`,
                              width: `${widthPct}%`,
                              backgroundColor: bg,
                            }}
                            onClick={(e) => {
                              e.stopPropagation();
                              onSeek(seg.start_time);
                            }}
                            className={`absolute top-0 bottom-0 flex items-center justify-center px-1 text-black font-mono text-[11px] font-bold tracking-wider overflow-hidden transition-all select-none rounded-none ${
                              isSegActive ? "ring-2 ring-white z-10 brightness-110" : "opacity-90 hover:opacity-100 hover:brightness-110"
                            }`}
                            title={`${meta.emoji} ${meta.label}: ${formatTimestamp(seg.start_time)} - ${formatTimestamp(seg.end_time)} (${Math.round(seg.average_confidence * 100)}% confidence)`}
                          >
                            {widthPct > 8 && (
                              <span className="truncate drop-shadow-sm flex items-center gap-1 text-black">
                                <span>{meta.emoji}</span>
                                <span className="capitalize hidden sm:inline">{meta.label}</span>
                              </span>
                            )}
                          </div>
                        );
                      })}

                      {/* Playhead line */}
                      <div
                        style={{ left: `${Math.max(0, Math.min(100, (currentTime / duration) * 100))}%` }}
                        className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_10px_rgba(255,255,255,1)] pointer-events-none z-20"
                      />
                    </div>
                  </div>
                );
              })
            ) : (
              /* Single Selected Track */
              <div
                onClick={handleTimelineClick}
                className="relative h-10 w-full bg-[#141414] border border-[#262626] cursor-pointer hover:border-[#3a3a3a] transition-colors rounded-none overflow-hidden"
              >
                {filteredSegments.map((seg) => {
                  const widthPct = Math.max(0.5, (seg.duration / duration) * 100);
                  const leftPct = (seg.start_time / duration) * 100;
                  const isSegActive = currentTime >= seg.start_time && currentTime <= seg.end_time;
                  const emotionKey = seg.emotion.toLowerCase();
                  const meta = EMOTIONS[emotionKey as PredictionEmotion] || EMOTIONS.neutral;
                  const bg = EMOTION_COLORS[emotionKey] || "rgb(56, 189, 248)";

                  return (
                    <div
                      key={seg.id}
                      style={{
                        left: `${leftPct}%`,
                        width: `${widthPct}%`,
                        backgroundColor: bg,
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        onSeek(seg.start_time);
                      }}
                      className={`absolute top-0 bottom-0 flex items-center justify-center px-1 text-black font-mono text-[11px] font-bold tracking-wider overflow-hidden transition-all select-none rounded-none ${
                        isSegActive ? "ring-2 ring-white z-10 brightness-110" : "opacity-90 hover:opacity-100 hover:brightness-110"
                      }`}
                      title={`${meta.emoji} ${meta.label}: ${formatTimestamp(seg.start_time)} - ${formatTimestamp(seg.end_time)}`}
                    >
                      {widthPct > 6 && (
                        <span className="truncate drop-shadow-sm flex items-center gap-1 text-black">
                          <span>{meta.emoji}</span>
                          <span className="capitalize">{meta.label}</span>
                        </span>
                      )}
                    </div>
                  );
                })}

                {/* Playhead line */}
                <div
                  style={{ left: `${Math.max(0, Math.min(100, (currentTime / duration) * 100))}%` }}
                  className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_10px_rgba(255,255,255,1)] pointer-events-none z-20"
                />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom Section: Color Legend & Current Moment Card */}
      <div className="space-y-4 pt-3">
        {/* Emotion Color Legend */}
        <div className="pt-3 border-t border-[#262626]">
          <div className="font-mono text-[10px] uppercase tracking-[1.5px] text-[#666666] mb-2.5 flex items-center gap-1.5">
            <Sparkles className="w-3 h-3 text-[#c3d9f3]" />
            <span>EMOTION COLOR REFERENCE</span>
          </div>
          <div className="flex flex-wrap gap-2.5 sm:gap-4">
            {Object.entries(EMOTION_COLORS).filter(([key]) => key !== "uncertain").map(([emotionKey, color]) => {
              const meta = EMOTIONS[emotionKey as PredictionEmotion] || EMOTIONS.neutral;
              return (
                <div key={emotionKey} className="flex items-center gap-1.5 text-xs">
                  <span
                    className="w-2 h-2 rounded-none inline-block"
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-sm select-none">{meta.emoji}</span>
                  <span className="text-[#cccccc] font-mono text-[11px] capitalize">{meta.label}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Active / Inspected Segment Detail Card */}
        {activeSegment && (
          <div
            className="bg-[#141414] border p-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 rounded-none"
            style={{ borderColor: activeMeta.color }}
          >
            <div className="flex items-center gap-3">
              <span className="text-3xl select-none" role="img" aria-label={activeMeta.label}>
                {activeMeta.emoji}
              </span>
              <div>
                <span className="font-mono text-[10px] uppercase tracking-[1.5px] text-[#999999] block">
                  CURRENT MOMENT EMOTION
                </span>
                <div className="flex items-center gap-2 mt-0.5">
                  <span
                    className="font-display text-2xl uppercase tracking-[1.5px] font-normal"
                    style={{ color: activeMeta.color }}
                  >
                    {activeMeta.label}
                  </span>
                  <span className="font-mono text-xs text-[#c3d9f3] bg-[#1f1f1f] px-2 py-0.5 border border-[#3a3a3a] rounded-none">
                    FACE #{activeSegment.track_id}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-4 font-mono text-xs text-[#999999]">
              <div>
                <span className="block text-[9px] uppercase text-[#666666] tracking-wider">
                  TIME RANGE
                </span>
                <span className="text-white">
                  {formatTimestamp(activeSegment.start_time)} – {formatTimestamp(activeSegment.end_time)}
                </span>
              </div>
              <div>
                <span className="block text-[9px] uppercase text-[#666666] tracking-wider">
                  DURATION
                </span>
                <span className="text-white">{activeSegment.duration.toFixed(1)}s</span>
              </div>
              <div>
                <span className="block text-[9px] uppercase text-[#666666] tracking-wider">
                  CONFIDENCE
                </span>
                <span style={{ color: activeMeta.color }} className="font-bold">
                  {(activeSegment.average_confidence * 100).toFixed(1)}%
                </span>
              </div>

              <button
                type="button"
                onClick={() => onSeek(activeSegment.start_time)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 border border-white bg-white text-black font-mono text-[11px] uppercase tracking-[1.5px] hover:bg-[#eaeaea] transition-all cursor-pointer rounded-none"
              >
                <Play className="w-3 h-3 fill-current" />
                <span>JUMP TO START</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default EmotionTimeline;
