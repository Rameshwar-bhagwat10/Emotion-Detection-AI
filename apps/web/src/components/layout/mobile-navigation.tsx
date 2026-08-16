import React from "react";

export interface MobileNavigationProps {
  className?: string;
}

export function MobileNavigation({ className }: MobileNavigationProps) {
  return (
    <div className={className}>
      <span>MobileNavigation placeholder</span>
    </div>
  );
}

export default MobileNavigation;
