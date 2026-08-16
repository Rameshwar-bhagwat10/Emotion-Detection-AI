import React from "react";

export interface ProgressProps {
  className?: string;
}

export function Progress({ className }: ProgressProps) {
  return (
    <div className={className}>
      <span>Progress placeholder</span>
    </div>
  );
}

export default Progress;
