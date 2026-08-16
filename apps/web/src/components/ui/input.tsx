import React from "react";

export interface InputProps {
  className?: string;
}

export function Input({ className }: InputProps) {
  return (
    <div className={className}>
      <span>Input placeholder</span>
    </div>
  );
}

export default Input;
