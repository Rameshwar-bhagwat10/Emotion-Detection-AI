"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Camera,
  Image as ImageIcon,
  LayoutDashboard,
  Video,
  PanelLeftClose,
} from "lucide-react";

interface NavItem {
  name: string;
  href: string;
  icon: React.ElementType;
  tag?: string;
  isAvailable: boolean;
}

const NAV_ITEMS: NavItem[] = [
  {
    name: "Dashboard",
    href: "/dashboard",
    icon: LayoutDashboard,
    isAvailable: true,
  },
  {
    name: "Live Detection",
    href: "/live",
    icon: Camera,
    tag: "LIVE",
    isAvailable: true,
  },
  {
    name: "Image Analysis",
    href: "/image-analysis",
    icon: ImageIcon,
    isAvailable: true,
  },
  {
    name: "Video Analysis",
    href: "/video-analysis",
    icon: Video,
    isAvailable: true,
  },
];

export interface AppSidebarProps {
  isOpen?: boolean;
  onClose?: () => void;
  className?: string;
}

export function AppSidebar({ isOpen = true, onClose, className = "" }: AppSidebarProps) {
  const pathname = usePathname();

  return (
    <aside
      className={`fixed inset-y-0 left-0 z-40 flex flex-col w-64 bg-black border-r border-[#262626] transition-transform duration-300 ease-in-out ${
        isOpen ? "translate-x-0" : "-translate-x-full"
      } ${className}`}
    >
      {/* Brand Header & Quick Collapse */}
      <div className="flex items-center justify-between h-14 px-5 border-b border-[#262626]">
        <Link href="/" className="flex flex-col group">
          <span className="font-display text-[15px] uppercase tracking-[5px] text-white font-normal group-hover:text-white/80 transition-colors">
            VALENCE
          </span>
          <span className="font-mono text-[9px] uppercase tracking-[3px] text-[#999999]">
            EMOTION AI
          </span>
        </Link>

        {onClose && (
          <button
            onClick={onClose}
            className="p-1.5 text-[#666666] hover:text-white hover:bg-[#161616] rounded-none transition-colors cursor-pointer"
            title="Collapse sidebar"
            aria-label="Collapse sidebar"
          >
            <PanelLeftClose className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Navigation Links */}
      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
        <div className="px-3 pb-2 font-mono text-[10px] uppercase tracking-[2px] text-[#555555]">
          Menu
        </div>

        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive =
            pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));

          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={() => {
                if (typeof window !== "undefined" && window.innerWidth < 1024 && onClose) {
                  onClose();
                }
              }}
              className={`flex items-center justify-between px-3 py-2.5 rounded-none text-xs font-mono uppercase tracking-[1.5px] transition-all duration-150 ${
                isActive
                  ? "bg-[#161616] text-white border-l-2 border-white pl-2.5 font-medium"
                  : "text-[#888888] hover:text-white hover:bg-[#111111]"
              }`}
            >
              <div className="flex items-center gap-3">
                <Icon
                  className={`w-3.5 h-3.5 transition-colors ${
                    isActive ? "text-white" : "text-[#666666]"
                  }`}
                />
                <span>{item.name}</span>
              </div>
              {item.tag && (
                <span
                  className={`text-[9px] px-1.5 py-0.5 font-mono uppercase tracking-[1px] border rounded-none ${
                    isActive
                      ? "border-emerald-500/50 text-emerald-400 bg-emerald-950/20"
                      : "border-[#262626] text-[#666666]"
                  }`}
                >
                  {item.tag}
                </span>
              )}
            </Link>
          );
        })}
      </div>

      {/* Footer / Telemetry & Collapse Button */}
      <div className="p-3 border-t border-[#262626] bg-black space-y-2">
        <div className="px-3 py-2 bg-[#0d0d0d] border border-[#222222] rounded-none flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]" />
            <span className="font-mono text-[10px] uppercase tracking-[1px] text-[#cccccc]">
              ResNet-18 CBAM
            </span>
          </div>
          <span className="font-mono text-[10px] text-emerald-400">
            ~18ms
          </span>
        </div>

        {onClose && (
          <button
            onClick={onClose}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 text-xs font-mono uppercase tracking-[1.5px] text-[#777777] hover:text-white hover:bg-[#141414] rounded-none transition-colors cursor-pointer"
          >
            <PanelLeftClose className="w-3.5 h-3.5" />
            <span>Collapse</span>
          </button>
        )}
      </div>
    </aside>
  );
}

export default AppSidebar;
