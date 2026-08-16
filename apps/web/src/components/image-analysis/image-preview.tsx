import React from "react";

export interface ImagePreviewProps {
  className?: string;
}

export function ImagePreview({ className }: ImagePreviewProps) {
  return (
    <div className={className}>
      <span>ImagePreview placeholder</span>
    </div>
  );
}

export default ImagePreview;
