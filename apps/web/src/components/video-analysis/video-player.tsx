import React from "react";

export interface VideoPlayerProps {
  className?: string;
}

export function VideoPlayer({ className }: VideoPlayerProps) {
  return (
    <div className={className}>
      <span>VideoPlayer placeholder</span>
    </div>
  );
}

export default VideoPlayer;
