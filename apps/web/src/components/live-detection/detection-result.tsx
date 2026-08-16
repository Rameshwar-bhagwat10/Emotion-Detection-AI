import React from "react";

export interface DetectionResultProps {
  className?: string;
}

export function DetectionResult({ className }: DetectionResultProps) {
  return (
    <div className={className}>
      <span>DetectionResult placeholder</span>
    </div>
  );
}

export default DetectionResult;
