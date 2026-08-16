import React from "react";

export interface TooltipProps {
  className?: string;
}

export function Tooltip({ className }: TooltipProps) {
  return (
    <div className={className}>
      <span>Tooltip placeholder</span>
    </div>
  );
}

export default Tooltip;
