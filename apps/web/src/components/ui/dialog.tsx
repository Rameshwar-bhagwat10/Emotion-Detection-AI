import React from "react";

export interface DialogProps {
  className?: string;
}

export function Dialog({ className }: DialogProps) {
  return (
    <div className={className}>
      <span>Dialog placeholder</span>
    </div>
  );
}

export default Dialog;
