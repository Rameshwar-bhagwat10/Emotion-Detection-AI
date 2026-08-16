import React from "react";

export interface AnalysisDetailsProps {
  className?: string;
}

export function AnalysisDetails({ className }: AnalysisDetailsProps) {
  return (
    <div className={className}>
      <span>AnalysisDetails placeholder</span>
    </div>
  );
}

export default AnalysisDetails;
