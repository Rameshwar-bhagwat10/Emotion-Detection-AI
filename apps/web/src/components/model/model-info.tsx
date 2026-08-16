import React from "react";

export interface ModelInfoProps {
  className?: string;
}

export function ModelInfo({ className }: ModelInfoProps) {
  return (
    <div className={className}>
      <span>ModelInfo placeholder</span>
    </div>
  );
}

export default ModelInfo;
