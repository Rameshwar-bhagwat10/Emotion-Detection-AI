import React from "react";

export interface PageContainerProps {
  className?: string;
}

export function PageContainer({ className }: PageContainerProps) {
  return (
    <div className={className}>
      <span>PageContainer placeholder</span>
    </div>
  );
}

export default PageContainer;
