import React from "react";

export interface EmotionTimelineChartProps {
  className?: string;
}

export function EmotionTimelineChart({ className }: EmotionTimelineChartProps) {
  return (
    <div className={className}>
      <span>EmotionTimelineChart placeholder</span>
    </div>
  );
}

export default EmotionTimelineChart;
