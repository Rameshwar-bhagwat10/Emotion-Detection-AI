import React from "react";

export interface QuickActionsProps {
  className?: string;
}

export function QuickActions({ className }: QuickActionsProps) {
  return (
    <div className={className}>
      <span>QuickActions placeholder</span>
    </div>
  );
}

export default QuickActions;
