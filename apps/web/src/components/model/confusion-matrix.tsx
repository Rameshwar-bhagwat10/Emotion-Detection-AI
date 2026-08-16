import React from "react";

export interface ConfusionMatrixProps {
  className?: string;
}

export function ConfusionMatrix({ className }: ConfusionMatrixProps) {
  return (
    <div className={className}>
      <span>ConfusionMatrix placeholder</span>
    </div>
  );
}

export default ConfusionMatrix;
