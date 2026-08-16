import React from "react";

export interface TabsProps {
  className?: string;
}

export function Tabs({ className }: TabsProps) {
  return (
    <div className={className}>
      <span>Tabs placeholder</span>
    </div>
  );
}

export default Tabs;
