"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Shield, Layers, FlaskConical, Info, Sparkles, Bot } from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();

  const navItems = [
    { name: "Guardian Agent", href: "/guardian", icon: Bot },
    { name: "Protect Studio", href: "/protect", icon: Shield },
    { name: "Dashboard", href: "/dashboard", icon: Layers },
    { name: "Research Lab", href: "/research", icon: FlaskConical },
    { name: "About & Ethics", href: "/about", icon: Info },
  ];

  return (
    <header className="sticky top-0 z-50 backdrop-blur-md bg-background/80 border-b border-surface-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <Link href="/" className="flex items-center space-x-2 group">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 to-cyan-400 p-0.5 shadow-lg shadow-sky-500/20 group-hover:scale-105 transition-transform">
                <div className="w-full h-full bg-background rounded-[10px] flex items-center justify-center">
                  <Shield className="w-5 h-5 text-sky-400" />
                </div>
              </div>
              <div className="flex flex-col">
                <span className="font-bold text-lg tracking-tight text-white flex items-center gap-1.5">
                  AI-Protect
                  <span className="text-[10px] uppercase font-semibold tracking-wider px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20">
                    Engine
                  </span>
                </span>
                <span className="text-[11px] text-slate-400 -mt-1 hidden sm:block">
                  Adversarial Cloaking Platform
                </span>
              </div>
            </Link>
          </div>

          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-sky-500/15 text-sky-400 border border-sky-500/30 shadow-sm"
                      : "text-slate-300 hover:text-white hover:bg-slate-800/60"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center space-x-3">
            <Link
              href="/protect"
              className="inline-flex items-center space-x-2 px-4 py-2 text-sm font-semibold rounded-lg bg-gradient-to-r from-sky-500 to-cyan-500 hover:from-sky-400 hover:to-cyan-400 text-white shadow-md shadow-sky-500/25 transition-all hover:scale-[1.02]"
            >
              <Sparkles className="w-4 h-4" />
              <span>New Protection</span>
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}
