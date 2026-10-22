"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Shield, Sparkles, Layers, Cpu, CheckCircle2, RefreshCw, ArrowUpRight, Activity } from "lucide-react";
import { getHealth, HealthCheckResponse } from "@/lib/api";

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthCheckResponse | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {});
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">Protection Dashboard</h1>
          <p className="text-sm text-slate-400 mt-1">
            Real-time status of the adversarial cloaking engine, active surrogate models, and protection jobs.
          </p>
        </div>

        <Link
          href="/protect"
          className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl font-semibold text-white bg-gradient-to-r from-sky-500 to-cyan-500 hover:from-sky-400 hover:to-cyan-400 shadow-md shadow-sky-500/20 text-sm transition-all hover:scale-105"
        >
          <Sparkles className="w-4 h-4" />
          <span>New Protection Job</span>
        </Link>
      </div>

      {/* System Status Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-2 backdrop-blur-sm">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Engine Status</span>
            <Activity className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-lg font-bold text-white capitalize">{health?.status || "Connecting..."}</span>
          </div>
          <span className="text-[11px] text-slate-500 block">FastAPI v0.1.0 &bull; SQLite DB</span>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-2 backdrop-blur-sm">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>PyTorch Acceleration</span>
            <Cpu className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-lg font-bold text-white">
            {health?.cuda_available ? "CUDA GPU" : "CPU Engine"}
          </div>
          <span className="text-[11px] text-slate-500 block">
            {health?.torch_available ? "PyTorch Backend Active" : "Initializing..."}
          </span>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-2 backdrop-blur-sm">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Available Surrogates</span>
            <Shield className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-lg font-bold text-white">4 Architectures</div>
          <span className="text-[11px] text-slate-500 block">ResNet, ViT, ConvNeXt, CLIP</span>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-2 backdrop-blur-sm">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Robustness Suite</span>
            <RefreshCw className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-lg font-bold text-white">6 Transforms</div>
          <span className="text-[11px] text-slate-500 block">JPEG, Resize, Crop, Blur, etc.</span>
        </div>
      </div>

      {/* Workspace overview */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 space-y-4">
        <h2 className="text-base font-semibold text-white">Quick Start & Protection Workflow</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-2">
            <span className="font-semibold text-sky-400 block text-sm">Step 1: Upload & Mode</span>
            <p className="text-slate-400 leading-relaxed">
              Upload artwork, portrait, or generic imagery. The system inspects integrity, size, and generates unique privacy digests.
            </p>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-2">
            <span className="font-semibold text-emerald-400 block text-sm">Step 2: Configure Bounds</span>
            <p className="text-slate-400 leading-relaxed">
              Select perturbation budget (&epsilon;) and target surrogate model. Choose which evaluation models to benchmark cross-transferability.
            </p>
          </div>
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-2">
            <span className="font-semibold text-purple-400 block text-sm">Step 3: Analyze & Export</span>
            <p className="text-slate-400 leading-relaxed">
              Examine side-by-side perceptual quality (SSIM, PSNR, LPIPS), robustness survival against JPEG, and export comprehensive reports.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
