import React from "react";

export interface VideoUploaderProps {
  className?: string;
}

export function VideoUploader({ className }: VideoUploaderProps) {
  return (
    <div className={className}>
      <span>VideoUploader placeholder</span>
    </div>
  );
}

export default VideoUploader;
