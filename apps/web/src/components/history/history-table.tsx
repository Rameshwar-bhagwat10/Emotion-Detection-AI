import React from "react";

export interface HistoryTableProps {
  className?: string;
}

export function HistoryTable({ className }: HistoryTableProps) {
  return (
    <div className={className}>
      <span>HistoryTable placeholder</span>
    </div>
  );
}

export default HistoryTable;
