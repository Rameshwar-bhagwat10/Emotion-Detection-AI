import React from "react";

export interface EmptyStateProps {
  className?: string;
}

export function EmptyState({ className }: EmptyStateProps) {
  return (
    <div className={className}>
      <span>EmptyState placeholder</span>
    </div>
  );
}

export default EmptyState;
