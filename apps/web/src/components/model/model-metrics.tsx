import React from "react";

export interface ModelMetricsProps {
  className?: string;
}

export function ModelMetrics({ className }: ModelMetricsProps) {
  return (
    <div className={className}>
      <span>ModelMetrics placeholder</span>
    </div>
  );
}

export default ModelMetrics;
