import React from "react";

export interface ProcessingProgressProps {
  className?: string;
}

export function ProcessingProgress({ className }: ProcessingProgressProps) {
  return (
    <div className={className}>
      <span>ProcessingProgress placeholder</span>
    </div>
  );
}

export default ProcessingProgress;
