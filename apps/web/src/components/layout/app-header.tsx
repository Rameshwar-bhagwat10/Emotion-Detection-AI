"use client";

import React, { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { PanelLeft, PanelLeftClose } from "lucide-react";
import { getHealth } from "@/lib/api/endpoints";

export interface AppHeaderProps {
  onToggleSidebar?: () => void;
  isSidebarOpen?: boolean;
  className?: string;
}

export function AppHeader({ onToggleSidebar, isSidebarOpen = true, className = "" }: AppHeaderProps) {
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
    if (pathname.includes("/live")) return "Live Emotion Detection";
    if (pathname.includes("/image-analysis")) return "Image Emotion Analysis";
    if (pathname.includes("/video-analysis")) return "Video Emotion Analysis";
    return "Dashboard";
  };

  return (
    <header
      className={`sticky top-0 z-30 flex items-center justify-between h-14 px-4 md:px-8 bg-black/90 backdrop-blur-md border-b border-[#262626] ${className}`}
    >
      {/* Left: Menu toggle & Breadcrumb */}
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="p-1.5 -ml-1.5 text-[#888888] hover:text-white hover:bg-[#161616] rounded transition-colors cursor-pointer flex items-center justify-center"
          title={isSidebarOpen ? "Hide sidebar" : "Show sidebar"}
          aria-label={isSidebarOpen ? "Hide sidebar" : "Show sidebar"}
        >
          {isSidebarOpen ? (
            <PanelLeftClose className="w-4 h-4" />
          ) : (
            <PanelLeft className="w-4 h-4" />
          )}
        </button>

        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] uppercase tracking-[2px] text-[#666666] hidden sm:inline">
            VALENCE
          </span>
          <span className="text-[#333333] hidden sm:inline font-mono text-xs">/</span>
          <span className="font-mono text-xs uppercase tracking-[2px] text-white">
            {getPageTitle()}
          </span>
        </div>
      </div>

      {/* Right: Telemetry / Status Pill */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 px-3 py-1 bg-[#111111] border border-[#262626] rounded-full">
          <span
            className={`w-2 h-2 rounded-full ${
              apiStatus === "online"
                ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]"
                : apiStatus === "checking"
                ? "bg-amber-400 animate-pulse"
                : "bg-red-500"
            }`}
          />
          <span className="font-mono text-[10px] uppercase tracking-[1.5px] text-[#999999]">
            {apiStatus === "online" ? "System Active" : apiStatus === "checking" ? "Connecting" : "Offline"}
          </span>
        </div>
      </div>
    </header>
  );
}

export default AppHeader;
