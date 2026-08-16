import React from "react";

export interface UserMenuProps {
  className?: string;
}

export function UserMenu({ className }: UserMenuProps) {
  return (
    <div className={className}>
      <span>UserMenu placeholder</span>
    </div>
  );
}

export default UserMenu;
