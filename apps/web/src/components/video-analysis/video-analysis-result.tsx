import React from "react";

export interface VideoAnalysisResultProps {
  className?: string;
}

export function VideoAnalysisResult({ className }: VideoAnalysisResultProps) {
  return (
    <div className={className}>
      <span>VideoAnalysisResult placeholder</span>
    </div>
  );
}

export default VideoAnalysisResult;
