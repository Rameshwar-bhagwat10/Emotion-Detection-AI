import React from "react";

export interface StatCardProps {
  className?: string;
}

export function StatCard({ className }: StatCardProps) {
  return (
    <div className={className}>
      <span>StatCard placeholder</span>
    </div>
  );
}

export default StatCard;
