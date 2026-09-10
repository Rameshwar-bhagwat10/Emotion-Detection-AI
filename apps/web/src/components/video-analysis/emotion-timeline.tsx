"use client";

import React, { useState, useMemo } from "react";
import { Play, Sparkles, User, BarChart2 } from "lucide-react";
import { EMOTIONS } from "@/types/emotion";
import { ExpressionEvent, ExpressionSegment, VideoTrack } from "@/types/video-analysis";

export interface EmotionTimelineProps {
  durationSeconds: number;
  segments: ExpressionSegment[];
  events?: ExpressionEvent[];
  tracks: VideoTrack[];
  currentTime: number;
  onSeek: (seconds: number) => void;
  className?: string;
}

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
    const tickCount = Math.min(10, Math.max(3, Math.floor(duration / 2)));
    const step = duration / tickCount;
    return Array.from({ length: tickCount + 1 }, (_, i) => i * step);
  }, [duration]);

  return (
    <div className={`bg-card/70 backdrop-blur-md border border-border/80 rounded-2xl p-6 shadow-xl space-y-6 ${className}`}>
      {/* Header & Track Selector Tabs */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-border/60">
        <div>
          <h3 className="text-lg font-bold text-foreground flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-primary/10 text-primary">
              <Sparkles className="w-4 h-4" />
            </span>
            Synchronized Expression Timeline
          </h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Interactive temporal progression. Click any expression segment or timestamp to seek playback.
          </p>
        </div>

        {/* Multi-Face Track Tabs */}
        {tracks && tracks.length > 1 && (
          <div className="flex items-center gap-1 bg-muted/60 p-1 rounded-xl border border-border/50">
            <button
              type="button"
              onClick={() => setSelectedTrackId("all")}
              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                selectedTrackId === "all"
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              All Faces ({tracks.length})
            </button>
            {tracks.map((t) => (
              <button
                key={t.track_id}
                type="button"
                onClick={() => setSelectedTrackId(t.track_id)}
                className={`flex items-center gap-1 px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                  selectedTrackId === t.track_id
                    ? "bg-card text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <User className="w-3 h-3 text-primary" />
                Track {t.track_id}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Timeline Visualization Container */}
      <div className="space-y-2">
        {/* Time Ruler */}
        <div className="relative h-4 w-full text-[10px] font-mono text-muted-foreground select-none">
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
                  <div className="flex items-center justify-between text-[11px] font-semibold text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <User className="w-3 h-3 text-primary" />
                      Face Track {trackId}
                    </span>
                    <span>{segs.length} segments</span>
                  </div>

                  {/* Render segmented bar for this track */}
                  <div
                    onClick={handleTimelineClick}
                    className="relative h-10 w-full bg-muted/40 rounded-xl overflow-hidden cursor-pointer border border-border/60 hover:border-primary/50 transition-colors flex"
                  >
                    {segs.map((seg) => {
                      const meta = EMOTIONS[seg.emotion] || EMOTIONS.neutral;
                      const widthPct = Math.max(0.5, (seg.duration / duration) * 100);
                      const leftPct = (seg.start_time / duration) * 100;
                      const isSegActive = currentTime >= seg.start_time && currentTime <= seg.end_time;

                      return (
                        <div
                          key={seg.id}
                          style={{
                            left: `${leftPct}%`,
                            width: `${widthPct}%`,
                            backgroundColor: meta.color,
                          }}
                          onClick={(e) => {
                            e.stopPropagation();
                            onSeek(seg.start_time);
                          }}
                          className={`absolute top-0 bottom-0 flex items-center justify-center px-1 text-white text-[11px] font-bold overflow-hidden transition-all hover:brightness-110 hover:z-10 group ${
                            isSegActive ? "ring-2 ring-white shadow-lg z-10" : "opacity-90"
                          }`}
                          title={`${meta.label}: ${formatTimestamp(seg.start_time)} - ${formatTimestamp(seg.end_time)} (${Math.round(seg.average_confidence * 100)}% conf)`}
                        >
                          {widthPct > 6 && (
                            <span className="truncate drop-shadow-sm flex items-center gap-1">
                              <span>{meta.emoji}</span>
                              {widthPct > 12 && <span className="capitalize">{seg.emotion}</span>}
                            </span>
                          )}
                        </div>
                      );
                    })}

                    {/* Playhead line */}
                    <div
                      style={{ left: `${Math.max(0, Math.min(100, (currentTime / duration) * 100))}%` }}
                      className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_8px_rgba(255,255,255,0.9)] pointer-events-none z-20"
                    >
                      <div className="absolute -top-1 -left-1 w-2.5 h-2.5 rounded-full bg-white shadow-md" />
                    </div>
                  </div>
                </div>
              );
            })
          ) : (
            /* Single Selected Track */
            <div
              onClick={handleTimelineClick}
              className="relative h-12 w-full bg-muted/40 rounded-xl overflow-hidden cursor-pointer border border-border/60 hover:border-primary/50 transition-colors flex"
            >
              {filteredSegments.map((seg) => {
                const meta = EMOTIONS[seg.emotion] || EMOTIONS.neutral;
                const widthPct = Math.max(0.5, (seg.duration / duration) * 100);
                const leftPct = (seg.start_time / duration) * 100;
                const isSegActive = currentTime >= seg.start_time && currentTime <= seg.end_time;

                return (
                  <div
                    key={seg.id}
                    style={{
                      left: `${leftPct}%`,
                      width: `${widthPct}%`,
                      backgroundColor: meta.color,
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSeek(seg.start_time);
                    }}
                    className={`absolute top-0 bottom-0 flex items-center justify-center px-1 text-white text-xs font-bold overflow-hidden transition-all hover:brightness-110 hover:z-10 ${
                      isSegActive ? "ring-2 ring-white shadow-lg z-10" : "opacity-90"
                    }`}
                    title={`${meta.label}: ${formatTimestamp(seg.start_time)} - ${formatTimestamp(seg.end_time)}`}
                  >
                    {widthPct > 5 && (
                      <span className="truncate drop-shadow-sm flex items-center gap-1.5">
                        <span>{meta.emoji}</span>
                        {widthPct > 10 && <span className="capitalize">{seg.emotion}</span>}
                        {widthPct > 18 && (
                          <span className="opacity-90 font-mono text-[10px]">
                            {Math.round(seg.average_confidence * 100)}%
                          </span>
                        )}
                      </span>
                    )}
                  </div>
                );
              })}

              {/* Playhead line */}
              <div
                style={{ left: `${Math.max(0, Math.min(100, (currentTime / duration) * 100))}%` }}
                className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_8px_rgba(255,255,255,0.9)] pointer-events-none z-20"
              >
                <div className="absolute -top-1 -left-1 w-2.5 h-2.5 rounded-full bg-white shadow-md" />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Active / Inspected Segment Detail Card */}
      {activeSegment && (
        <div className="bg-muted/30 border border-border/60 rounded-xl p-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 animate-in fade-in">
          <div className="flex items-center gap-3.5">
            <div
              style={{ backgroundColor: EMOTIONS[activeSegment.emotion]?.color || "gray" }}
              className="w-10 h-10 rounded-xl flex items-center justify-center text-white text-xl shadow-md shrink-0"
            >
              {EMOTIONS[activeSegment.emotion]?.emoji || "😐"}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold text-foreground capitalize">
                  {activeSegment.emotion}
                </span>
                <span className="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-primary/10 text-primary border border-primary/20">
                  Track {activeSegment.track_id}
                </span>
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">
                {EMOTIONS[activeSegment.emotion]?.description}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-4 text-xs font-mono text-muted-foreground">
            <div>
              <span className="block text-[10px] uppercase text-muted-foreground/80 font-sans font-semibold">
                Interval
              </span>
              <span className="text-foreground font-semibold">
                {formatTimestamp(activeSegment.start_time)} – {formatTimestamp(activeSegment.end_time)}
              </span>
            </div>
            <div>
              <span className="block text-[10px] uppercase text-muted-foreground/80 font-sans font-semibold">
                Duration
              </span>
              <span className="text-foreground font-semibold">{activeSegment.duration.toFixed(1)}s</span>
            </div>
            <div>
              <span className="block text-[10px] uppercase text-muted-foreground/80 font-sans font-semibold">
                Avg Confidence
              </span>
              <span className="text-emerald-500 font-bold">
                {(activeSegment.average_confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span className="block text-[10px] uppercase text-muted-foreground/80 font-sans font-semibold">
                Predictions
              </span>
              <span className="text-foreground font-semibold">{activeSegment.prediction_count}</span>
            </div>

            <button
              type="button"
              onClick={() => onSeek(activeSegment.start_time)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground font-semibold hover:bg-primary/90 transition-all font-sans text-xs shadow-sm"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              Play Segment
            </button>
          </div>
        </div>
      )}

      {/* Emotion Legend */}
      <div className="flex flex-wrap items-center gap-3 pt-2">
        <span className="text-xs font-semibold text-muted-foreground flex items-center gap-1">
          <BarChart2 className="w-3.5 h-3.5" /> Legend:
        </span>
        {Object.entries(EMOTIONS).map(([key, meta]) => (
          <div key={key} className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <span
              style={{ backgroundColor: meta.color }}
              className="w-2.5 h-2.5 rounded-full inline-block"
            />
            <span className="capitalize">{meta.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default EmotionTimeline;
