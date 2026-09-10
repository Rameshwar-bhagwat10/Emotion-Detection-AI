"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Camera, Image as ImageIcon, Menu, X } from "lucide-react";
import { getHealth } from "@/lib/api/endpoints";

export interface AppHeaderProps {
  onToggleSidebar?: () => void;
  isSidebarOpen?: boolean;
  className?: string;
}

export function AppHeader({ onToggleSidebar, isSidebarOpen, className = "" }: AppHeaderProps) {
  const pathname = usePathname();
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");

  const checkStatus = async () => {
    try {
      const data = await getHealth();
      if (data.status === "ok" || data.status === "degraded") {
        setApiStatus("online");
      } else {
        setApiStatus("offline");
      }
    } catch {
      setApiStatus("offline");
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  const getPageTitle = () => {
    if (pathname.includes("/live")) return "Real-Time Webcam Detection";
    if (pathname.includes("/image-analysis")) return "Image Facial Expression Analysis";
    if (pathname.includes("/model")) return "Model & Architecture Specs";
    if (pathname.includes("/settings")) return "Application Settings";
    return "Application Dashboard";
  };

  return (
    <header
      className={`sticky top-0 z-30 flex items-center justify-between h-16 px-4 md:px-8 bg-zinc-950/80 backdrop-blur-md border-b border-zinc-800/80 ${className}`}
    >
      {/* Left: Mobile Toggle & Page Title */}
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="p-2 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 lg:hidden cursor-pointer"
          aria-label={isSidebarOpen ? "Close menu" : "Open menu"}
        >
          {isSidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>

        <div>
          <h1 className="text-base md:text-lg font-semibold text-zinc-100">{getPageTitle()}</h1>
        </div>
      </div>

      {/* Right: API Status Pill & Action Shortcuts */}
      <div className="flex items-center gap-3">
        {/* Live Backend Connection Status Pill */}
        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-medium transition-colors ${
            apiStatus === "online"
              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/25"
              : apiStatus === "checking"
              ? "bg-amber-500/10 text-amber-400 border-amber-500/25"
              : "bg-rose-500/10 text-rose-400 border-rose-500/25"
          }`}
          title="Backend Health Probe Status"
        >
          <span
            className={`w-2 h-2 rounded-full ${
              apiStatus === "online"
                ? "bg-emerald-400 animate-pulse"
                : apiStatus === "checking"
                ? "bg-amber-400 animate-ping"
                : "bg-rose-500"
            }`}
          />
          <span className="hidden sm:inline">
            {apiStatus === "online" ? "FastAPI Online" : apiStatus === "checking" ? "Checking API..." : "FastAPI Offline"}
          </span>
        </div>

        {/* Quick Launch Buttons */}
        <div className="flex items-center gap-2">
          <Link
            href="/image-analysis"
            className="hidden md:flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-zinc-300 bg-zinc-900 border border-zinc-800 rounded-lg hover:bg-zinc-800 hover:text-white transition-colors"
          >
            <ImageIcon className="w-3.5 h-3.5 text-indigo-400" />
            <span>Upload Image</span>
          </Link>
          <Link
            href="/live"
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-gradient-to-r from-indigo-600 to-purple-600 rounded-lg hover:from-indigo-500 hover:to-purple-500 shadow-sm transition-all"
          >
            <Camera className="w-3.5 h-3.5" />
            <span>Live Camera</span>
          </Link>
        </div>
      </div>
    </header>
  );
}

export default AppHeader;
