import React from "react";

export interface ImageAnalysisResultProps {
  className?: string;
}

export function ImageAnalysisResult({ className }: ImageAnalysisResultProps) {
  return (
    <div className={className}>
      <span>ImageAnalysisResult placeholder</span>
    </div>
  );
}

export default ImageAnalysisResult;
