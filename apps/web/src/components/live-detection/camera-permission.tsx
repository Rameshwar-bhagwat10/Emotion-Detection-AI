import React from "react";

export interface CameraPermissionProps {
  className?: string;
}

export function CameraPermission({ className }: CameraPermissionProps) {
  return (
    <div className={className}>
      <span>CameraPermission placeholder</span>
    </div>
  );
}

export default CameraPermission;
