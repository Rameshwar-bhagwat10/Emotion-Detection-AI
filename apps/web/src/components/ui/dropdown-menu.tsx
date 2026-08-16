import React from "react";

export interface DropdownMenuProps {
  className?: string;
}

export function DropdownMenu({ className }: DropdownMenuProps) {
  return (
    <div className={className}>
      <span>DropdownMenu placeholder</span>
    </div>
  );
}

export default DropdownMenu;
