import React from "react";

export interface AppHeaderProps {
  className?: string;
}

export function AppHeader({ className }: AppHeaderProps) {
  return (
    <div className={className}>
      <span>AppHeader placeholder</span>
    </div>
  );
}

export default AppHeader;
