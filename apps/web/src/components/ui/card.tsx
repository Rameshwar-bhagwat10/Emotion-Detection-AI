import React from "react";

export interface CardProps {
  className?: string;
}

export function Card({ className }: CardProps) {
  return (
    <div className={className}>
      <span>Card placeholder</span>
    </div>
  );
}

export default Card;
