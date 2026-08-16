import React from "react";

export interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div className={className}>
      <span>Skeleton placeholder</span>
    </div>
  );
}

export default Skeleton;
