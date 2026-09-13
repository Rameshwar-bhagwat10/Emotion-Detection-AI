"use client";

import React, { useState, useEffect } from "react";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { AppHeader } from "@/components/layout/app-header";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  // Desktop defaults to open; mobile defaults to closed
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  useEffect(() => {
    if (typeof window !== "undefined" && window.innerWidth < 1024) {
      setIsSidebarOpen(false);
    }
  }, []);

  const toggleSidebar = () => {
    setIsSidebarOpen((prev) => !prev);
  };

  return (
    <div className="min-h-screen bg-black text-[#cccccc] flex selection:bg-white selection:text-black relative">
      {/* Sidebar for Desktop & Mobile Overlay */}
      <AppSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
      />

      {/* Backdrop for Mobile */}
      {isSidebarOpen && (
        <div
          onClick={() => setIsSidebarOpen(false)}
          className="fixed inset-0 z-30 bg-black/80 backdrop-blur-sm lg:hidden transition-opacity cursor-pointer"
        />
      )}

      {/* Main Content Area */}
      <div
        className={`flex-1 flex flex-col min-w-0 bg-black transition-[padding] duration-300 ease-in-out ${
          isSidebarOpen ? "lg:pl-64" : "lg:pl-0"
        }`}
      >
        <AppHeader
          isSidebarOpen={isSidebarOpen}
          onToggleSidebar={toggleSidebar}
        />
        <main className="flex-1 p-6 md:p-12 max-w-7xl w-full mx-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
