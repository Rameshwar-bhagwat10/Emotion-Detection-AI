import React from "react";

export interface ErrorStateProps {
  className?: string;
}

export function ErrorState({ className }: ErrorStateProps) {
  return (
    <div className={className}>
      <span>ErrorState placeholder</span>
    </div>
  );
}

export default ErrorState;
