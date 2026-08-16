import React from "react";

export interface EmotionDistributionChartProps {
  className?: string;
}

export function EmotionDistributionChart({ className }: EmotionDistributionChartProps) {
  return (
    <div className={className}>
      <span>EmotionDistributionChart placeholder</span>
    </div>
  );
}

export default EmotionDistributionChart;
