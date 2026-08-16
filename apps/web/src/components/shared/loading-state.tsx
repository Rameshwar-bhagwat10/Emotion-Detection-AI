import React from "react";

export interface LoadingStateProps {
  className?: string;
}

export function LoadingState({ className }: LoadingStateProps) {
  return (
    <div className={className}>
      <span>LoadingState placeholder</span>
    </div>
  );
}

export default LoadingState;
