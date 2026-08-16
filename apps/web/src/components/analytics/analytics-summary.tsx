import React from "react";

export interface AnalyticsSummaryProps {
  className?: string;
}

export function AnalyticsSummary({ className }: AnalyticsSummaryProps) {
  return (
    <div className={className}>
      <span>AnalyticsSummary placeholder</span>
    </div>
  );
}

export default AnalyticsSummary;
