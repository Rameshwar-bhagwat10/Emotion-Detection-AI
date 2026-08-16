import React from "react";

export interface TableProps {
  className?: string;
}

export function Table({ className }: TableProps) {
  return (
    <div className={className}>
      <span>Table placeholder</span>
    </div>
  );
}

export default Table;
