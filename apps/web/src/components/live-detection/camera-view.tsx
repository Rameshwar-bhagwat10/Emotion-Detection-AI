import React from "react";

export interface CameraViewProps {
  className?: string;
}

export function CameraView({ className }: CameraViewProps) {
  return (
    <div className={className}>
      <span>CameraView placeholder</span>
    </div>
  );
}

export default CameraView;
