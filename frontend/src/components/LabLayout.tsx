"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Shield,
  Bot,
  Eye,
  Cpu,
  RefreshCw,
  Archive,
  FileText,
  Settings,
  Menu,
  X,
  ChevronLeft,
  ChevronRight,
  PlusCircle,
  Activity,
  CheckCircle2,
  AlertTriangle,
  Zap,
  Sliders,
  ExternalLink,
} from "lucide-react";
import { getHealth, HealthCheckResponse } from "@/lib/api";

interface LabLayoutProps {
  children: React.ReactNode;
}

export default function LabLayout({ children }: LabLayoutProps) {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [health, setHealth] = useState<HealthCheckResponse | null>(null);
  const [backendStatus, setBackendStatus] = useState<"connected" | "disconnected" | "checking">("checking");

  useEffect(() => {
    let isMounted = true;
    getHealth()
      .then((data) => {
        if (isMounted) {
          setHealth(data);
          setBackendStatus("connected");
        }
      })
      .catch(() => {
        if (isMounted) {
          setBackendStatus("disconnected");
        }
      });
    return () => {
      isMounted = false;
    };
  }, [pathname]);

  const navItems = [
    { label: "Guardian Overview", href: "/", icon: Bot, desc: "Intelligence workspace & telemetry" },
    { label: "New Experiment", href: "/protect", icon: PlusCircle, desc: "Synthesize adversarial cloaks" },
    { label: "Artwork Microscope", href: "/results/a335e6c9-cdba-48e7-bccf-e004c7606b27", icon: Eye, desc: "Synchronized pixel inspection" },
    { label: "Model Arena", href: "/arena", icon: Cpu, desc: "Cross-architecture transferability" },
    { label: "Robustness Journey", href: "/robustness", icon: RefreshCw, desc: "Interactive digital transforms" },
    { label: "Experiment Memory", href: "/history", icon: Archive, desc: "Visual timeline & run comparison" },
    { label: "Research Reports", href: "/reports", icon: FileText, desc: "Exportable scientific audits" },
    { label: "Settings & System", href: "/about", icon: Settings, desc: "Mathematical & pipeline design" },
  ];

  return (
    <div className="min-h-screen flex flex-col lg:flex-row bg-[#101217] text-[#F4F1E9] antialiased selection:bg-[#5CE1D2]/25 selection:text-[#5CE1D2]">
      {/* Mobile Top Header */}
      <div className="lg:hidden flex items-center justify-between p-4 bg-[#181C24] border-b border-[#262C38] sticky top-0 z-50 backdrop-blur-md">
        <Link href="/" className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-[#5CE1D2] to-[#A9A5FF] flex items-center justify-center text-[#101217] font-bold shadow-md shadow-[#5CE1D2]/15">
            <Shield className="w-4 h-4 fill-current" />
          </div>
          <div>
            <span className="font-bold text-xs tracking-wider text-[#F4F1E9] block">
              AI-PROTECT
            </span>
            <span className="text-[9px] text-[#5CE1D2] font-mono tracking-widest uppercase">
              Guardian Intelligence
            </span>
          </div>
        </Link>
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="p-2 rounded-lg bg-[#202631] text-[#9BA5B4] hover:text-[#F4F1E9] transition"
          aria-label="Toggle Navigation"
        >
          {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Left Navigation Sidebar */}
      <aside
        className={`fixed lg:sticky top-0 left-0 bottom-0 z-40 bg-[#181C24] border-r border-[#262C38] flex flex-col justify-between transition-all duration-300 ${
          isCollapsed ? "w-20" : "w-64"
        } ${
          mobileMenuOpen
            ? "translate-x-0 w-72"
            : "-translate-x-full lg:translate-x-0"
        }`}
      >
        <div className="p-4 space-y-6 overflow-y-auto">
          {/* Brand Header & Collapse Toggle */}
          <div className="flex items-center justify-between">
            <Link
              href="/"
              className={`flex items-center space-x-3 group overflow-hidden ${
                isCollapsed ? "justify-center w-full" : ""
              }`}
            >
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-[#5CE1D2] via-[#A9A5FF] to-[#5CE1D2] p-0.5 shadow-md shadow-[#5CE1D2]/15 shrink-0 group-hover:scale-105 transition-transform">
                <div className="w-full h-full bg-[#101217] rounded-[10px] flex items-center justify-center text-[#5CE1D2]">
                  <Shield className="w-5 h-5" />
                </div>
              </div>
              {!isCollapsed && (
                <div className="truncate">
                  <div className="flex items-center space-x-1.5">
                    <span className="font-extrabold text-sm tracking-tight text-[#F4F1E9]">
                      AI-PROTECT
                    </span>
                    <span className="w-1.5 h-1.5 rounded-full bg-[#5CE1D2] animate-pulse" />
                  </div>
                  <span className="text-[10px] text-[#9BA5B4] font-mono tracking-wider block">
                    GUARDIAN INTELLIGENCE
                  </span>
                </div>
              )}
            </Link>

            {/* Desktop Collapse Toggle */}
            <button
              onClick={() => setIsCollapsed(!isCollapsed)}
              className="hidden lg:flex p-1.5 rounded-lg text-[#9BA5B4] hover:text-[#F4F1E9] hover:bg-[#202631] transition"
              title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
            >
              {isCollapsed ? (
                <ChevronRight className="w-4 h-4" />
              ) : (
                <ChevronLeft className="w-4 h-4" />
              )}
            </button>
          </div>

          {/* Navigation Links */}
          <nav className="space-y-1.5 pt-2">
            {!isCollapsed && (
              <span className="text-[10px] font-mono uppercase tracking-widest text-[#9BA5B4] px-3 block mb-2 font-bold select-none">
                Research Modules
              </span>
            )}
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive =
                item.href === "/"
                  ? pathname === "/"
                  : pathname?.startsWith(item.href.split("?")[0]) ||
                    (item.href.includes("/results/") && pathname?.startsWith("/results/"));

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setMobileMenuOpen(false)}
                  title={isCollapsed ? item.label : undefined}
                  className={`group relative flex items-center ${
                    isCollapsed ? "justify-center px-2 py-3" : "px-3 py-2.5 space-x-3"
                  } rounded-xl text-xs transition-all ${
                    isActive
                      ? "bg-[#202631] text-[#5CE1D2] font-semibold shadow-sm"
                      : "text-[#9BA5B4] hover:text-[#F4F1E9] hover:bg-[#202631]/60"
                  }`}
                >
                  {/* Active Teal Indicator Bar */}
                  {isActive && (
                    <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-5 rounded-r bg-[#5CE1D2]" />
                  )}

                  <Icon
                    className={`w-4 h-4 shrink-0 transition-colors ${
                      isActive ? "text-[#5CE1D2]" : "text-[#9BA5B4] group-hover:text-[#F4F1E9]"
                    }`}
                  />

                  {!isCollapsed && (
                    <div className="truncate">
                      <span className="block font-medium truncate">{item.label}</span>
                      <span className="block text-[10px] text-[#9BA5B4]/80 font-normal truncate">
                        {item.desc}
                      </span>
                    </div>
                  )}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Sidebar Telemetry Footer */}
        <div className="p-3.5 border-t border-[#262C38] bg-[#101217]/60 space-y-2.5 font-mono text-[11px]">
          {!isCollapsed ? (
            <>
              <div className="flex items-center justify-between text-xs">
                <span className="text-[#9BA5B4]">Backend Core:</span>
                <span
                  className={`inline-flex items-center space-x-1.5 ${
                    backendStatus === "connected"
                      ? "text-[#5CE1D2]"
                      : backendStatus === "checking"
                      ? "text-[#A9A5FF]"
                      : "text-[#FF8C79]"
                  }`}
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
                  <span className="font-semibold text-[10px] uppercase">
                    {backendStatus === "connected"
                      ? health?.cuda_available
                        ? "CUDA Active"
                        : "Online"
                      : backendStatus === "checking"
                      ? "Connecting"
                      : "Offline"}
                  </span>
                </span>
              </div>

              <div className="flex items-center justify-between text-[#9BA5B4] text-[10px]">
                <span>Surrogate Model:</span>
                <span className="text-[#F4F1E9] font-medium">ResNet-50</span>
              </div>

              <div className="pt-2 border-t border-[#202631] flex items-center justify-between text-[10px] text-[#9BA5B4]">
                <span>RTX 3050 &bull; 4GB</span>
                <span className="text-[#5CE1D2] font-semibold">v0.1.0</span>
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center py-1 text-center" title="Backend: Online">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  backendStatus === "connected"
                    ? "bg-[#5CE1D2]"
                    : backendStatus === "checking"
                    ? "bg-[#A9A5FF]"
                    : "bg-[#FF8C79]"
                }`}
              />
            </div>
          )}
        </div>
      </aside>

      {/* Main Workspace Frame */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Bar */}
        <header className="hidden lg:flex items-center justify-between px-8 py-3.5 bg-[#181C24]/80 border-b border-[#262C38] backdrop-blur-md sticky top-0 z-30">
          <div className="flex items-center space-x-3 text-xs font-mono">
            <span className="text-[#9BA5B4]">Workspace:</span>
            <span className="px-2.5 py-1 rounded-lg bg-[#101217] border border-[#262C38] text-[#F4F1E9] font-semibold">
              Research Workstation
            </span>
            <span className="text-[#262C38]">&bull;</span>
            <span className="text-[#9BA5B4]">Guardian Agent:</span>
            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-[#5CE1D2]/10 text-[#5CE1D2] border border-[#5CE1D2]/20 font-semibold text-[11px]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#5CE1D2] animate-pulse" />
              <span>ORCHESTRATOR READY</span>
            </span>
          </div>

          <div className="flex items-center space-x-4">
            {/* Backend Connectivity Status Pill */}
            <div className="flex items-center space-x-2 text-xs font-mono px-3 py-1 rounded-lg bg-[#101217] border border-[#262C38]">
              <span
                className={`w-2 h-2 rounded-full ${
                  backendStatus === "connected"
                    ? "bg-[#5CE1D2]"
                    : backendStatus === "checking"
                    ? "bg-[#A9A5FF]"
                    : "bg-[#FF8C79]"
                }`}
              />
              <span className="text-[#9BA5B4] text-[11px]">
                {backendStatus === "connected"
                  ? "API Connected (127.0.0.1:8000)"
                  : backendStatus === "checking"
                  ? "Checking API..."
                  : "Backend Offline"}
              </span>
            </div>

            {/* Quick Action Button */}
            <Link
              href="/protect"
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-gradient-to-r from-[#5CE1D2] to-[#38A89C] hover:from-[#5CE1D2] hover:to-[#5CE1D2] text-[#101217] text-xs font-bold shadow-md shadow-[#5CE1D2]/15 transition-all hover:scale-[1.02] cursor-pointer"
            >
              <Zap className="w-3.5 h-3.5 fill-current" />
              <span>New Experiment</span>
            </Link>
          </div>
        </header>

        {/* Dynamic Page Content */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}
