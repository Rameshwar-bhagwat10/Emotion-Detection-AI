import React from "react";

export interface ToastProps {
  className?: string;
}

export function Toast({ className }: ToastProps) {
  return (
    <div className={className}>
      <span>Toast placeholder</span>
    </div>
  );
}

export default Toast;
