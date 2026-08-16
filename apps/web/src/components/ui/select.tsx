import React from "react";

export interface SelectProps {
  className?: string;
}

export function Select({ className }: SelectProps) {
  return (
    <div className={className}>
      <span>Select placeholder</span>
    </div>
  );
}

export default Select;
