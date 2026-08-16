import React from "react";

export interface FaceResultProps {
  className?: string;
}

export function FaceResult({ className }: FaceResultProps) {
  return (
    <div className={className}>
      <span>FaceResult placeholder</span>
    </div>
  );
}

export default FaceResult;
