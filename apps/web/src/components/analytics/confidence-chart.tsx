import React from "react";

export interface ConfidenceChartProps {
  className?: string;
}

export function ConfidenceChart({ className }: ConfidenceChartProps) {
  return (
    <div className={className}>
      <span>ConfidenceChart placeholder</span>
    </div>
  );
}

export default ConfidenceChart;
