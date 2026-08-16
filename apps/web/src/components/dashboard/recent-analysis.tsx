import React from "react";

export interface RecentAnalysisProps {
  className?: string;
}

export function RecentAnalysis({ className }: RecentAnalysisProps) {
  return (
    <div className={className}>
      <span>RecentAnalysis placeholder</span>
    </div>
  );
}

export default RecentAnalysis;
