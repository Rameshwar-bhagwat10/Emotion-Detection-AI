import React from "react";

export interface EmotionTimelineProps {
  className?: string;
}

export function EmotionTimeline({ className }: EmotionTimelineProps) {
  return (
    <div className={className}>
      <span>EmotionTimeline placeholder</span>
    </div>
  );
}

export default EmotionTimeline;
