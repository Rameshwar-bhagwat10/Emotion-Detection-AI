import React from "react";

export interface ConfidenceIndicatorProps {
  className?: string;
}

export function ConfidenceIndicator({ className }: ConfidenceIndicatorProps) {
  return (
    <div className={className}>
      <span>ConfidenceIndicator placeholder</span>
    </div>
  );
}

export default ConfidenceIndicator;
