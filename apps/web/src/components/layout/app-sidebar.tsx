import React from "react";

export interface AppSidebarProps {
  className?: string;
}

export function AppSidebar({ className }: AppSidebarProps) {
  return (
    <div className={className}>
      <span>AppSidebar placeholder</span>
    </div>
  );
}

export default AppSidebar;
