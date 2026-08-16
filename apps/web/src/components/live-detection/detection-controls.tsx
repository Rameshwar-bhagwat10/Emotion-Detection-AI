import React from "react";

export interface DetectionControlsProps {
  className?: string;
}

export function DetectionControls({ className }: DetectionControlsProps) {
  return (
    <div className={className}>
      <span>DetectionControls placeholder</span>
    </div>
  );
}

export default DetectionControls;
