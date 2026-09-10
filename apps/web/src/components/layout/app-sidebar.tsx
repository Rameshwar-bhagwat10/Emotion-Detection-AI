"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Camera,
  Cpu,
  History,
  Image as ImageIcon,
  LayoutDashboard,
  LineChart,
  Video,
  Settings,
  Sparkles,
  Activity,
} from "lucide-react";

interface NavItem {
  name: string;
  href: string;
  icon: React.ElementType;
  badge?: string;
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
    badge: "Real-time",
    isAvailable: true,
  },
  {
    name: "Image Analysis",
    href: "/image-analysis",
    icon: ImageIcon,
    badge: "Single/Multi",
    isAvailable: true,
  },
  {
    name: "Video Analysis",
    href: "/video-analysis",
    icon: Video,
    badge: "Temporal",
    isAvailable: true,
  },
  {
    name: "Analytics",
    href: "/analytics",
    icon: LineChart,
    isAvailable: true,
  },
  {
    name: "History",
    href: "/history",
    icon: History,
    isAvailable: true,
  },
  {
    name: "Model Specs",
    href: "/model",
    icon: Cpu,
    isAvailable: true,
  },
  {
    name: "Settings",
    href: "/settings",
    icon: Settings,
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
      className={`fixed inset-y-0 left-0 z-40 flex flex-col w-64 bg-zinc-950/95 backdrop-blur-xl border-r border-zinc-800/80 transition-transform duration-300 lg:translate-x-0 ${
        isOpen ? "translate-x-0" : "-translate-x-full"
      } ${className}`}
    >
      {/* Brand Header */}
      <div className="flex items-center justify-between h-16 px-5 border-b border-zinc-800/80">
        <Link href="/dashboard" className="flex items-center gap-3 group">
          <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 via-purple-500 to-pink-500 shadow-md shadow-indigo-500/20 group-hover:scale-105 transition-transform">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="font-bold text-base text-zinc-100 tracking-tight group-hover:text-white transition-colors">
              Emotion<span className="text-indigo-400">AI</span>
            </span>
            <span className="block text-[10px] uppercase font-semibold text-zinc-500 tracking-wider">
              Phase 13 Integrated
            </span>
          </div>
        </Link>
      </div>

      {/* Navigation Links */}
      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-1.5 scrollbar-thin scrollbar-thumb-zinc-800">
        <div className="px-3 pb-2 text-[11px] font-bold text-zinc-500 uppercase tracking-wider">
          Core Applications
        </div>

        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));

          if (!item.isAvailable) {
            return (
              <div
                key={item.href}
                className="flex items-center justify-between px-3 py-2 rounded-xl text-zinc-600 text-sm cursor-not-allowed select-none"
                title="Available in Phase 13"
              >
                <div className="flex items-center gap-3">
                  <Icon className="w-4 h-4 text-zinc-700" />
                  <span>{item.name}</span>
                </div>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-600 font-mono">
                  Phase 13
                </span>
              </div>
            );
          }

          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={onClose}
              className={`flex items-center justify-between px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150 ${
                isActive
                  ? "bg-indigo-600/15 text-indigo-300 border border-indigo-500/30 shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/80"
              }`}
            >
              <div className="flex items-center gap-3">
                <Icon
                  className={`w-4 h-4 transition-colors ${
                    isActive ? "text-indigo-400" : "text-zinc-400 group-hover:text-zinc-200"
                  }`}
                />
                <span>{item.name}</span>
              </div>
              {item.badge && (
                <span
                  className={`text-[10px] px-2 py-0.5 rounded-full font-semibold border ${
                    isActive
                      ? "bg-indigo-500/20 text-indigo-300 border-indigo-500/30"
                      : "bg-zinc-900 text-zinc-400 border-zinc-800"
                  }`}
                >
                  {item.badge}
                </span>
              )}
            </Link>
          );
        })}
      </div>

      {/* System Status Footer */}
      <div className="p-3 border-t border-zinc-800/80 bg-zinc-950/60">
        <div className="flex items-center justify-between p-2.5 rounded-xl bg-zinc-900/60 border border-zinc-800/60">
          <div className="flex items-center gap-2.5">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <span className="text-xs font-medium text-zinc-300">ResNet-18 Champion</span>
          </div>
          <Activity className="w-3.5 h-3.5 text-zinc-500" />
        </div>
      </div>
    </aside>
  );
}

export default AppSidebar;
