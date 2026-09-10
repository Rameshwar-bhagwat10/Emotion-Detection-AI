"use client";

import React, { useRef, useEffect, useState } from "react";
import { Play, Pause, RotateCcw, Volume2, VolumeX, Maximize2, Layers, AlertCircle, RefreshCw } from "lucide-react";
import { EMOTIONS, PredictionEmotion } from "@/types/emotion";
import { VideoMetadata, VideoPredictionItem } from "@/types/video-analysis";

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
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number }>({
    width: metadata?.width || 640,
    height: metadata?.height || 360,
  });

  // Reset error when streamUrl changes
  useEffect(() => {
    setHasPlaybackError(false);
  }, [streamUrl]);

  // Handle external seek requests
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

    // Group by track_id and pick closest
    const byTrack: Record<number, VideoPredictionItem> = {};
    matching.forEach((p) => {
      const existing = byTrack[p.track_id];
      if (!existing || Math.abs(p.timestamp - currentTime) < Math.abs(existing.timestamp - currentTime)) {
        byTrack[p.track_id] = p;
      }
    });

    return Object.values(byTrack);
  }, [predictions, currentTime]);

  return (
    <div
      ref={containerRef}
      className={`relative bg-black rounded-2xl overflow-hidden shadow-2xl border border-border/80 group ${className}`}
    >
      {/* Video Element */}
      <div className="relative aspect-video flex items-center justify-center bg-black/90">
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

        {/* Video Load Error Fallback Overlay */}
        {hasPlaybackError && (
          <div className="absolute inset-0 bg-black/85 flex flex-col items-center justify-center text-center p-6 space-y-3 z-30">
            <AlertCircle className="w-10 h-10 text-amber-500 animate-pulse" />
            <div>
              <p className="text-white text-sm font-semibold">Video Stream Playback Notice</p>
              <p className="text-white/70 text-xs mt-1 max-w-sm">
                The browser could not directly render this video container. Click reload to refresh the stream buffer.
              </p>
            </div>
            <button
              type="button"
              onClick={() => {
                setHasPlaybackError(false);
                if (videoRef.current) {
                  videoRef.current.load();
                }
              }}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:bg-primary/90 transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Reload Stream
            </button>
          </div>
        )}


        {/* Dynamic Face Bounding Box & Emotion Tag Overlay */}
        {showOverlay && videoDimensions.width > 0 && videoDimensions.height > 0 && (
          <div className="absolute inset-0 pointer-events-none overflow-hidden">
            {currentFaces.map((face) => {
              const meta = EMOTIONS[face.smoothed_emotion as PredictionEmotion] || EMOTIONS.neutral;
              // Bounding box percentages
              const left = (face.bbox.x / videoDimensions.width) * 100;
              const top = (face.bbox.y / videoDimensions.height) * 100;
              const width = (face.bbox.width / videoDimensions.width) * 100;
              const height = (face.bbox.height / videoDimensions.height) * 100;

              return (
                <div
                  key={face.track_id}
                  style={{
                    left: `${Math.max(0, left)}%`,
                    top: `${Math.max(0, top)}%`,
                    width: `${Math.min(100 - left, width)}%`,
                    height: `${Math.min(100 - top, height)}%`,
                  }}
                  className="absolute border-2 rounded-lg transition-all duration-100"
                  // Use CSS variable or style for border color
                >
                  <div
                    style={{ borderColor: meta.color }}
                    className="absolute inset-0 border-2 rounded-lg shadow-[0_0_12px_rgba(0,0,0,0.5)]"
                  />
                  {/* Face Tag Badge */}
                  <div
                    style={{ backgroundColor: meta.color }}
                    className="absolute -top-7 left-0 px-2 py-0.5 rounded text-[11px] font-bold text-white shadow-md flex items-center gap-1.5 whitespace-nowrap"
                  >
                    <span>{meta.emoji}</span>
                    <span>Track {face.track_id}:</span>
                    <span className="capitalize">{face.smoothed_emotion}</span>
                    <span className="opacity-90 font-mono text-[10px]">
                      {Math.round(face.smoothed_confidence * 100)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Floating Controls Bar */}
      <div className="bg-gradient-to-t from-black/95 via-black/80 to-transparent p-4 flex flex-col gap-3">
        {/* Scrubber Progress Bar */}
        <div className="relative flex items-center group/scrubber cursor-pointer">
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
            className="w-full accent-primary h-1.5 bg-white/20 rounded-lg cursor-pointer group-hover/scrubber:h-2.5 transition-all"
          />
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between gap-4 text-white text-sm">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={togglePlay}
              className="p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
              title={isPlaying ? "Pause" : "Play"}
            >
              {isPlaying ? <Pause className="w-5 h-5 fill-white" /> : <Play className="w-5 h-5 fill-white" />}
            </button>

            <button
              type="button"
              onClick={handleRestart}
              className="p-1.5 rounded-lg text-white/80 hover:text-white hover:bg-white/10 transition-colors"
              title="Restart"
            >
              <RotateCcw className="w-4 h-4" />
            </button>

            <div className="font-mono text-xs text-white/80">
              <span className="text-white font-semibold">{formatTime(currentTime)}</span>
              <span className="opacity-50 mx-1.5">/</span>
              <span>{formatTime(duration)}</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Toggle Overlay */}
            <button
              type="button"
              onClick={() => setShowOverlay(!showOverlay)}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium border transition-colors ${
                showOverlay
                  ? "bg-primary/20 border-primary/40 text-primary"
                  : "bg-white/5 border-white/10 text-white/60 hover:text-white"
              }`}
              title="Toggle Face Tracking & Emotion Overlay"
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Overlay</span>
            </button>

            {/* Playback speed */}
            <button
              type="button"
              onClick={handleRateChange}
              className="px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 text-xs font-mono text-white/90"
              title="Playback Speed"
            >
              {playbackRate}x
            </button>

            {/* Mute */}
            <button
              type="button"
              onClick={toggleMute}
              className="p-1.5 rounded-lg text-white/80 hover:text-white hover:bg-white/10 transition-colors"
            >
              {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </button>

            {/* Fullscreen */}
            <button
              type="button"
              onClick={toggleFullscreen}
              className="p-1.5 rounded-lg text-white/80 hover:text-white hover:bg-white/10 transition-colors"
              title="Fullscreen"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default VideoPlayer;
