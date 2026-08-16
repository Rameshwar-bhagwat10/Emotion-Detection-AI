import React from "react";

export interface EmotionOverviewProps {
  className?: string;
}

export function EmotionOverview({ className }: EmotionOverviewProps) {
  return (
    <div className={className}>
      <span>EmotionOverview placeholder</span>
    </div>
  );
}

export default EmotionOverview;
