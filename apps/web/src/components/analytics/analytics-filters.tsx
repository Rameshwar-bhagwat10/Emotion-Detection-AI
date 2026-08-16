import React from "react";

export interface AnalyticsFiltersProps {
  className?: string;
}

export function AnalyticsFilters({ className }: AnalyticsFiltersProps) {
  return (
    <div className={className}>
      <span>AnalyticsFilters placeholder</span>
    </div>
  );
}

export default AnalyticsFilters;
