import React from "react";

export interface EmotionBadgeProps {
  className?: string;
}

export function EmotionBadge({ className }: EmotionBadgeProps) {
  return (
    <div className={className}>
      <span>EmotionBadge placeholder</span>
    </div>
  );
}

export default EmotionBadge;
