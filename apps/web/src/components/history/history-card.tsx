import React from "react";

export interface HistoryCardProps {
  className?: string;
}

export function HistoryCard({ className }: HistoryCardProps) {
  return (
    <div className={className}>
      <span>HistoryCard placeholder</span>
    </div>
  );
}

export default HistoryCard;
