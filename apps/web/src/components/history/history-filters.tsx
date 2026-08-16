import React from "react";

export interface HistoryFiltersProps {
  className?: string;
}

export function HistoryFilters({ className }: HistoryFiltersProps) {
  return (
    <div className={className}>
      <span>HistoryFilters placeholder</span>
    </div>
  );
}

export default HistoryFilters;
