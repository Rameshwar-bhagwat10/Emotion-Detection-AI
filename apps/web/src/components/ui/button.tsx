import React from "react";

export interface ButtonProps {
  className?: string;
}

export function Button({ className }: ButtonProps) {
  return (
    <div className={className}>
      <span>Button placeholder</span>
    </div>
  );
}

export default Button;
