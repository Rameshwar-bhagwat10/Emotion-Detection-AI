"use client";

import React, { useRef, useEffect, useState } from "react";
import { Play, Pause, RotateCcw, Volume2, VolumeX, Maximize2, Minimize2, Layers, AlertTriangle } from "lucide-react";
import { VideoMetadata, VideoPredictionItem } from "@/types/video-analysis";
import { EMOTIONS, PredictionEmotion } from "@/types/emotion";

export interface VideoPlayerProps {
  streamUrl: string;
  metadata?: VideoMetadata;
  predictions?: VideoPredictionItem[];
  currentTime: number;
  onTimeUpdate: (time: number) => void;
  seekToTime?: number | null;
  className?: string;
}

export function VideoPlayer({
  streamUrl,
  metadata,
  predictions = [],
  currentTime,
  onTimeUpdate,
  seekToTime,
  className = "",
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [duration, setDuration] = useState<number>(metadata?.duration_seconds || 0);
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [showOverlay, setShowOverlay] = useState<boolean>(true);
  const [playbackRate, setPlaybackRate] = useState<number>(1.0);
  const [hasPlaybackError, setHasPlaybackError] = useState<boolean>(false);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number }>({
    width: metadata?.width || 640,
    height: metadata?.height || 360,
  });

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  useEffect(() => {
    setHasPlaybackError(false);
  }, [streamUrl]);

  useEffect(() => {
    if (seekToTime !== undefined && seekToTime !== null && videoRef.current) {
      videoRef.current.currentTime = seekToTime;
      onTimeUpdate(seekToTime);
    }
  }, [seekToTime, onTimeUpdate]);

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration || metadata?.duration_seconds || 0);
      setVideoDimensions({
        width: videoRef.current.videoWidth || metadata?.width || 640,
        height: videoRef.current.videoHeight || metadata?.height || 360,
      });
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      onTimeUpdate(videoRef.current.currentTime);
    }
  };

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  const handleRestart = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      onTimeUpdate(0);
      videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const handleRateChange = () => {
    const rates = [0.5, 1.0, 1.5, 2.0];
    const nextIdx = (rates.indexOf(playbackRate) + 1) % rates.length;
    const nextRate = rates[nextIdx];
    setPlaybackRate(nextRate);
    if (videoRef.current) {
      videoRef.current.playbackRate = nextRate;
    }
  };

  const toggleFullscreen = () => {
    if (containerRef.current) {
      if (!document.fullscreenElement) {
        containerRef.current.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    }
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 10);
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}.${ms}`;
  };

  // Find predictions matching current timestamp (within 0.35s window)
  const currentFaces = React.useMemo(() => {
    if (!predictions || predictions.length === 0) return [];
    const windowSec = 0.35;
    const matching = predictions.filter(
      (p) => Math.abs(p.timestamp - currentTime) <= windowSec
    );

    const byTrack: Record<number, VideoPredictionItem> = {};
    matching.forEach((p) => {
      const existing = byTrack[p.track_id];
      if (!existing || Math.abs(p.timestamp - currentTime) < Math.abs(existing.timestamp - currentTime)) {
        byTrack[p.track_id] = p;
      }
    });

    return Object.values(byTrack);
  }, [predictions, currentTime]);

  const viewScale = Math.max(0.6, Math.min(2.0, videoDimensions.width / 720));

  return (
    <div
      ref={containerRef}
      className={`relative bg-[#000000] border border-[#262626] rounded-none overflow-hidden flex flex-col justify-between ${className}`}
    >
      {/* Video Element Viewport */}
      <div className="relative aspect-video flex items-center justify-center bg-black overflow-hidden">
        {/* Precision Corner Crosshairs */}
        <div className="absolute top-3 left-3 w-3 h-3 border-t border-l border-white/30 pointer-events-none z-20" />
        <div className="absolute top-3 right-3 w-3 h-3 border-t border-r border-white/30 pointer-events-none z-20" />
        <div className="absolute bottom-3 left-3 w-3 h-3 border-b border-l border-white/30 pointer-events-none z-20" />
        <div className="absolute bottom-3 right-3 w-3 h-3 border-b border-r border-white/30 pointer-events-none z-20" />

        <video
          ref={videoRef}
          src={streamUrl}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onError={() => setHasPlaybackError(true)}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onEnded={() => setIsPlaying(false)}
          playsInline
          className="w-full h-full object-contain cursor-pointer"
          onClick={togglePlay}
        />

        {/* Video Load Error Fallback */}
        {hasPlaybackError && (
          <div className="absolute inset-0 bg-black/90 flex flex-col items-center justify-center text-center p-6 space-y-3 z-30 font-mono text-xs">
            <div className="w-10 h-10 border border-amber-500/40 bg-amber-950/20 flex items-center justify-center rounded-none">
              <AlertTriangle className="w-5 h-5 text-amber-400" />
            </div>
            <span className="text-amber-400 font-bold uppercase tracking-wider">[STREAM BUFFER NOTICE]</span>
            <p className="font-sans text-xs text-[#999999] max-w-sm">
              The video stream requires buffer initialization or reload.
            </p>
            <button
              type="button"
              onClick={() => {
                setHasPlaybackError(false);
                if (videoRef.current) {
                  videoRef.current.load();
                }
              }}
              className="px-4 py-2 border border-white bg-white text-black font-mono text-xs uppercase tracking-[1.5px] hover:bg-[#eaeaea] transition-all cursor-pointer rounded-none"
            >
              RELOAD STREAM
            </button>
          </div>
        )}

        {/* High-Precision Cyber-Tactical SVG Face Overlay */}
        {showOverlay && videoDimensions.width > 0 && videoDimensions.height > 0 && currentFaces.length > 0 && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox={`0 0 ${videoDimensions.width} ${videoDimensions.height}`}
            preserveAspectRatio="xMidYMid meet"
          >
            <defs>
              {currentFaces.map((face) => {
                const meta =
                  EMOTIONS[face.smoothed_emotion.toLowerCase() as PredictionEmotion] ||
                  EMOTIONS.neutral;
                return (
                  <filter
                    key={`glow-${face.track_id}`}
                    id={`glow-v-${face.track_id}`}
                    x="-20%"
                    y="-20%"
                    width="140%"
                    height="140%"
                  >
                    <feDropShadow
                      dx="0"
                      dy="0"
                      stdDeviation={4 * viewScale}
                      floodColor={meta.color}
                      floodOpacity="0.85"
                    />
                  </filter>
                );
              })}
            </defs>

            {currentFaces.map((face) => {
              const meta =
                EMOTIONS[face.smoothed_emotion.toLowerCase() as PredictionEmotion] ||
                EMOTIONS.neutral;
              const color = meta.color || "rgb(56, 189, 248)";

              const { x, y, width: w, height: h } = face.bbox;
              const bLen = Math.max(14 * viewScale, Math.min(32 * viewScale, w * 0.22));
              const cx = x + w / 2;
              const cy = y + h / 2;
              const crossLen = 5 * viewScale;

              // Badge layout calculations
              const badgeH = 26 * viewScale;
              const fontSize = 10.5 * viewScale;
              const emojiSize = 12 * viewScale;
              const pctSize = 9 * viewScale;
              const labelText = (meta.label || "NEUTRAL").toUpperCase();
              const pctText = `${Math.round(face.smoothed_confidence * 100)}%`;

              const badgeW = Math.max(140 * viewScale, (labelText.length * 7.5 + 85) * viewScale);
              const badgeY = y > badgeH + 6 * viewScale ? y - badgeH - 6 * viewScale : y + 6 * viewScale;
              const badgeX = Math.max(2, Math.min(videoDimensions.width - badgeW - 2, x));

              return (
                <g key={face.track_id} className="transition-all duration-150">
                  {/* Subtle holographic face fill */}
                  <rect
                    x={x}
                    y={y}
                    width={w}
                    height={h}
                    fill={color}
                    fillOpacity="0.08"
                  />

                  {/* Hairline dashed perimeter */}
                  <rect
                    x={x}
                    y={y}
                    width={w}
                    height={h}
                    fill="none"
                    stroke={color}
                    strokeWidth="1"
                    strokeDasharray="4 3"
                    strokeOpacity="0.35"
                  />

                  {/* High-tech L-shaped corner brackets with glow filter */}
                  <path
                    d={`M ${x} ${y + bLen} L ${x} ${y} L ${x + bLen} ${y}`}
                    stroke={color}
                    strokeWidth={2.5 * viewScale}
                    fill="none"
                    strokeLinecap="square"
                    filter={`url(#glow-v-${face.track_id})`}
                  />
                  <path
                    d={`M ${x + w - bLen} ${y} L ${x + w} ${y} L ${x + w} ${y + bLen}`}
                    stroke={color}
                    strokeWidth={2.5 * viewScale}
                    fill="none"
                    strokeLinecap="square"
                    filter={`url(#glow-v-${face.track_id})`}
                  />
                  <path
                    d={`M ${x} ${y + h - bLen} L ${x} ${y + h} L ${x + bLen} ${y + h}`}
                    stroke={color}
                    strokeWidth={2.5 * viewScale}
                    fill="none"
                    strokeLinecap="square"
                    filter={`url(#glow-v-${face.track_id})`}
                  />
                  <path
                    d={`M ${x + w - bLen} ${y + h} L ${x + w} ${y + h} L ${x + w} ${y + h - bLen}`}
                    stroke={color}
                    strokeWidth={2.5 * viewScale}
                    fill="none"
                    strokeLinecap="square"
                    filter={`url(#glow-v-${face.track_id})`}
                  />

                  {/* 4 Corner micro-vertices */}
                  <rect x={x - 1.5} y={y - 1.5} width={3 * viewScale} height={3 * viewScale} fill="#ffffff" />
                  <rect x={x + w - 1.5} y={y - 1.5} width={3 * viewScale} height={3 * viewScale} fill="#ffffff" />
                  <rect x={x - 1.5} y={y + h - 1.5} width={3 * viewScale} height={3 * viewScale} fill="#ffffff" />
                  <rect x={x + w - 1.5} y={y + h - 1.5} width={3 * viewScale} height={3 * viewScale} fill="#ffffff" />

                  {/* Centroid precision crosshair [ + ] */}
                  <line
                    x1={cx - crossLen}
                    y1={cy}
                    x2={cx + crossLen}
                    y2={cy}
                    stroke={color}
                    strokeWidth="1"
                    strokeOpacity="0.6"
                  />
                  <line
                    x1={cx}
                    y1={cy - crossLen}
                    x2={cx}
                    y2={cy + crossLen}
                    stroke={color}
                    strokeWidth="1"
                    strokeOpacity="0.6"
                  />

                  {/* Floating Obsidian Emotion HUD Badge */}
                  <g>
                    {/* Solid Obsidian background */}
                    <rect
                      x={badgeX}
                      y={badgeY}
                      width={badgeW}
                      height={badgeH}
                      fill="#080808"
                      stroke={color}
                      strokeWidth="1"
                      fillOpacity="0.96"
                    />

                    {/* Left status accent strip */}
                    <rect
                      x={badgeX}
                      y={badgeY}
                      width={3.5 * viewScale}
                      height={badgeH}
                      fill={color}
                    />

                    {/* Canonical Emotion Emoji */}
                    <text
                      x={badgeX + 9 * viewScale}
                      y={badgeY + badgeH * 0.68}
                      fontSize={emojiSize}
                      dominantBaseline="middle"
                      style={{ userSelect: "none" }}
                    >
                      {meta.emoji}
                    </text>

                    {/* Emotion Label */}
                    <text
                      x={badgeX + 26 * viewScale}
                      y={badgeY + badgeH * 0.65}
                      fill="#ffffff"
                      fontSize={fontSize}
                      fontFamily="monospace"
                      fontWeight="bold"
                      dominantBaseline="middle"
                      letterSpacing="1px"
                    >
                      {labelText}
                    </text>

                    {/* Confidence Score Pill */}
                    <rect
                      x={badgeX + badgeW - 42 * viewScale}
                      y={badgeY + (badgeH - 16 * viewScale) / 2}
                      width={38 * viewScale}
                      height={16 * viewScale}
                      fill="#141414"
                      stroke="#2e2e2e"
                      strokeWidth="1"
                    />
                    <text
                      x={badgeX + badgeW - 23 * viewScale}
                      y={badgeY + badgeH * 0.66}
                      fill="#e0e0e0"
                      fontSize={pctSize}
                      fontFamily="monospace"
                      fontWeight="bold"
                      textAnchor="middle"
                      dominantBaseline="middle"
                    >
                      {pctText}
                    </text>

                    {/* Probability Progress Bar at Badge Bottom */}
                    <line
                      x1={badgeX}
                      y1={badgeY + badgeH}
                      x2={badgeX + badgeW * Math.max(0.1, face.smoothed_confidence)}
                      y2={badgeY + badgeH}
                      stroke={color}
                      strokeWidth={2 * viewScale}
                    />
                  </g>
                </g>
              );
            })}
          </svg>
        )}
      </div>

      {/* Controls Bar */}
      <div className="bg-[#0d0d0d] border-t border-[#262626] p-4 flex flex-col gap-3 rounded-none">
        {/* Scrubber Progress Bar */}
        <div className="relative flex items-center cursor-pointer">
          <input
            type="range"
            min={0}
            max={duration || 100}
            step={0.05}
            value={currentTime}
            onChange={(e) => {
              const t = parseFloat(e.target.value);
              if (videoRef.current) {
                videoRef.current.currentTime = t;
              }
              onTimeUpdate(t);
            }}
            className="w-full accent-white h-1 bg-[#262626] cursor-pointer rounded-none"
          />
        </div>

        {/* Action Buttons Row */}
        <div className="flex items-center justify-between gap-4 text-white text-xs font-mono">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={togglePlay}
              className="p-1.5 border border-[#3a3a3a] hover:border-white text-white transition-colors cursor-pointer rounded-none"
              title={isPlaying ? "Pause" : "Play"}
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5 fill-white" /> : <Play className="w-3.5 h-3.5 fill-white" />}
            </button>

            <button
              type="button"
              onClick={handleRestart}
              className="p-1.5 border border-[#262626] text-[#999999] hover:text-white hover:border-[#3a3a3a] transition-colors cursor-pointer rounded-none"
              title="Restart"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>

            <div className="tracking-wider text-[#999999] font-mono text-xs">
              <span className="text-white">{formatTime(currentTime)}</span>
              <span className="mx-1 text-[#3a3a3a]">/</span>
              <span>{formatTime(duration)}</span>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-3">
            {/* Toggle Overlay Reticle */}
            <button
              type="button"
              onClick={() => setShowOverlay(!showOverlay)}
              className={`flex items-center gap-1.5 px-2.5 py-1 border text-[10px] uppercase tracking-[1.5px] transition-colors cursor-pointer rounded-none ${
                showOverlay
                  ? "bg-[#1f1f1f] border-[#c3d9f3] text-[#c3d9f3]"
                  : "bg-[#141414] border-[#262626] text-[#666666] hover:text-white"
              }`}
              title="Toggle Face Tracking & Reticle Overlay"
            >
              <Layers className="w-3 h-3" />
              <span>RETICLE: {showOverlay ? "ON" : "OFF"}</span>
            </button>

            {/* Playback speed */}
            <button
              type="button"
              onClick={handleRateChange}
              className="px-2 py-1 border border-[#262626] bg-[#141414] text-[10px] font-mono text-[#999999] hover:text-white cursor-pointer rounded-none"
              title="Playback Speed"
            >
              {playbackRate}X
            </button>

            {/* Mute */}
            <button
              type="button"
              onClick={toggleMute}
              className="p-1.5 border border-[#262626] text-[#999999] hover:text-white cursor-pointer rounded-none"
            >
              {isMuted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
            </button>

            {/* Fullscreen */}
            <button
              type="button"
              onClick={toggleFullscreen}
              className="p-1.5 border border-[#262626] text-[#999999] hover:text-white cursor-pointer rounded-none"
              title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
            >
              {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default VideoPlayer;
