import React from "react";

export interface FaceOverlayProps {
  className?: string;
}

export function FaceOverlay({ className }: FaceOverlayProps) {
  return (
    <div className={className}>
      <span>FaceOverlay placeholder</span>
    </div>
  );
}

export default FaceOverlay;
