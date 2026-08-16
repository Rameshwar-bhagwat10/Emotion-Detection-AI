import React from "react";

export interface ImageUploaderProps {
  className?: string;
}

export function ImageUploader({ className }: ImageUploaderProps) {
  return (
    <div className={className}>
      <span>ImageUploader placeholder</span>
    </div>
  );
}

export default ImageUploader;
