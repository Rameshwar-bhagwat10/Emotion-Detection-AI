import React from "react";

export interface ModelComparisonProps {
  className?: string;
}

export function ModelComparison({ className }: ModelComparisonProps) {
  return (
    <div className={className}>
      <span>ModelComparison placeholder</span>
    </div>
  );
}

export default ModelComparison;
