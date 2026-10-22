"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Shield,
  Bot,
  Sparkles,
  Zap,
  History,
  ArrowRight,
  Eye,
  Cpu,
  RefreshCw,
  Archive,
  FileText,
  Activity,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Layers,
  Database,
  ExternalLink,
  ChevronRight,
  BarChart3,
  Loader2,
} from "lucide-react";
import LiquidCanvas from "@/components/LiquidCanvas";
import {
  listJobs,
  getHealth,
  JobSummaryItem,
  HealthCheckResponse,
  BACKEND_ROOT_URL,
} from "@/lib/api";

export default function GuardianOverviewPage() {
  const [jobs, setJobs] = useState<JobSummaryItem[]>([]);
  const [health, setHealth] = useState<HealthCheckResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    Promise.all([
      listJobs().catch(() => []),
      getHealth().catch(() => null),
    ]).then(([jobsData, healthData]) => {
      if (isMounted) {
        setJobs(jobsData || []);
        setHealth(healthData);
        setLoading(false);
      }
    });

    return () => {
      isMounted = false;
    };
  }, []);

  const completedJobs = jobs.filter((j) => j.status === "completed");
  const runningJobs = jobs.filter((j) => j.status === "pending" || (j.status as string) === "processing");
  const failedJobs = jobs.filter((j) => j.status === "failed");
  const recentJobs = jobs.slice(0, 5);

  return (
    <div className="max-w-7xl mx-auto space-y-10 pb-16">
      {/* 1. Hero Area: Your Art. Your Experiments. Your Evidence. */}
      <section className="relative overflow-hidden rounded-3xl bg-[#181C24] border border-[#262C38] p-8 sm:p-12 lg:p-16 shadow-2xl">
        {/* Subtle Ambient Particle Field */}
        <div className="absolute inset-0 opacity-30 pointer-events-none">
          <LiquidCanvas active={true} showControls={false} className="w-full h-full" />
        </div>

        <div className="relative z-10 max-w-3xl space-y-6">
          {/* Badge */}
          <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-[#5CE1D2]/10 border border-[#5CE1D2]/20 text-[#5CE1D2] text-xs font-semibold tracking-wide">
            <Bot className="w-3.5 h-3.5" />
            <span>AI-Protect Guardian Intelligence &bull; Adaptive Research Workspace</span>
          </div>

          {/* Heading */}
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight text-[#F4F1E9] leading-tight">
            Your Art. Your Experiments.{" "}
            <span className="bg-gradient-to-r from-[#5CE1D2] via-[#A9A5FF] to-[#5CE1D2] bg-clip-text text-transparent">
              Your Evidence.
            </span>
          </h1>

          {/* Subtitle */}
          <p className="text-sm sm:text-base text-[#9BA5B4] leading-relaxed max-w-2xl">
            Guardian helps plan, run, compare, and understand adversarial artwork experiments using the existing AI-Protect ML pipeline. Strictly bounded perturbations (&epsilon; &le; 8/255) safeguard creator rights while maintaining museum-grade perceptual fidelity.
          </p>

          {/* Primary & Secondary Action CTAs */}
          <div className="flex flex-col sm:flex-row items-center gap-4 pt-2">
            <Link
              href="/protect"
              className="w-full sm:w-auto inline-flex items-center justify-center space-x-2.5 px-6 py-3.5 rounded-xl bg-gradient-to-r from-[#5CE1D2] to-[#38A89C] hover:from-[#5CE1D2] hover:to-[#5CE1D2] text-[#101217] font-bold shadow-lg shadow-[#5CE1D2]/20 transition-all hover:scale-[1.02] cursor-pointer text-sm"
            >
              <Zap className="w-4 h-4 fill-current" />
              <span>New Experiment</span>
              <ArrowRight className="w-4 h-4 ml-1" />
            </Link>

            <Link
              href="/history"
              className="w-full sm:w-auto inline-flex items-center justify-center space-x-2 px-6 py-3.5 rounded-xl bg-[#202631] border border-[#262C38] hover:bg-[#262C38] text-[#F4F1E9] text-sm font-semibold transition-colors cursor-pointer"
            >
              <Archive className="w-4 h-4 text-[#5CE1D2]" />
              <span>Explore Experiment Memory</span>
            </Link>
          </div>

          {/* Scientific Disclaimer Banner */}
          <div className="p-4 rounded-xl bg-[#101217]/80 border border-[#262C38] flex items-start space-x-3 text-xs text-[#9BA5B4] max-w-2xl">
            <AlertTriangle className="w-4 h-4 text-[#FF8C79] shrink-0 mt-0.5" />
            <div className="leading-relaxed">
              <strong className="text-[#F4F1E9]">Empirical Research Reality:</strong> Adversarial perturbations operate by shifting deep feature embeddings past model decision boundaries. Protection levels reflect measured shifts on evaluated architectures and vary under lossy post-processing.
            </div>
          </div>
        </div>
      </section>

      {/* 2. System Telemetry & Model Pool Grid */}
      <section className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Core Backend Status */}
        <div className="p-5 rounded-2xl bg-[#181C24] border border-[#262C38] space-y-2">
          <div className="flex items-center justify-between text-xs text-[#9BA5B4]">
            <span>FastAPI Backend Core</span>
            <Activity className="w-4 h-4 text-[#5CE1D2]" />
          </div>
          <div className="flex items-center space-x-2">
            <div
              className={`w-2.5 h-2.5 rounded-full ${
                health?.status === "healthy" ? "bg-[#5CE1D2] animate-pulse" : "bg-[#FF8C79]"
              }`}
            />
            <span className="text-lg font-bold font-mono text-[#F4F1E9] capitalize">
              {health?.status || "Connecting"}
            </span>
          </div>
          <span className="text-[11px] font-mono text-[#9BA5B4] block">
            SQLite Database &bull; Port 8000
          </span>
        </div>

        {/* PyTorch Acceleration */}
        <div className="p-5 rounded-2xl bg-[#181C24] border border-[#262C38] space-y-2">
          <div className="flex items-center justify-between text-xs text-[#9BA5B4]">
            <span>Hardware Engine</span>
            <Cpu className="w-4 h-4 text-[#A9A5FF]" />
          </div>
          <div className="text-lg font-bold font-mono text-[#F4F1E9]">
            {health?.cuda_available ? "NVIDIA RTX 3050" : "CPU Fallback"}
          </div>
          <span className="text-[11px] font-mono text-[#9BA5B4] block">
            {health?.cuda_available ? "4.0 GB VRAM &bull; CUDA 12.4" : "CPU Mode Only"}
          </span>
        </div>

        {/* Experiment Records Count */}
        <div className="p-5 rounded-2xl bg-[#181C24] border border-[#262C38] space-y-2">
          <div className="flex items-center justify-between text-xs text-[#9BA5B4]">
            <span>Recorded Experiments</span>
            <Archive className="w-4 h-4 text-[#5CE1D2]" />
          </div>
          <div className="text-2xl font-bold font-mono text-[#F4F1E9]">
            {loading ? "..." : completedJobs.length}
          </div>
          <span className="text-[11px] font-mono text-[#9BA5B4] block">
            {runningJobs.length} active &bull; {failedJobs.length} failed
          </span>
        </div>

        {/* Primary Surrogate */}
        <div className="p-5 rounded-2xl bg-[#181C24] border border-[#262C38] space-y-2">
          <div className="flex items-center justify-between text-xs text-[#9BA5B4]">
            <span>Surrogate Defense Model</span>
            <Shield className="w-4 h-4 text-[#5CE1D2]" />
          </div>
          <div className="text-lg font-bold font-mono text-[#F4F1E9]">
            ResNet-50
          </div>
          <span className="text-[11px] font-mono text-[#9BA5B4] block">
            38 WikiArt Styles &bull; &epsilon; &le; 8/255
          </span>
        </div>
      </section>

      {/* 3. Running & Blocked Jobs Alerts (If any) */}
      {runningJobs.length > 0 && (
        <section className="p-4 rounded-2xl bg-[#5CE1D2]/10 border border-[#5CE1D2]/20 flex items-center justify-between text-xs text-[#5CE1D2]">
          <div className="flex items-center space-x-2.5">
            <Loader2 className="w-4 h-4 animate-spin text-[#5CE1D2]" />
            <span>
              <strong>{runningJobs.length} active experiment(s)</strong> processing in background.
            </span>
          </div>
          <Link
            href={`/results/${runningJobs[0].job_id}`}
            className="px-3 py-1 rounded-lg bg-[#5CE1D2] hover:bg-[#5CE1D2] text-[#101217] font-semibold text-xs transition"
          >
            Monitor Job
          </Link>
        </section>
      )}

      {failedJobs.length > 0 && (
        <section className="p-4 rounded-2xl bg-[#FF8C79]/10 border border-[#FF8C79]/20 flex items-center justify-between text-xs text-[#FF8C79]">
          <div className="flex items-center space-x-2.5">
            <AlertTriangle className="w-4 h-4 text-[#FF8C79]" />
            <span>
              <strong>{failedJobs.length} experiment(s)</strong> encountered execution errors.
            </span>
          </div>
          <Link
            href="/history"
            className="px-3 py-1 rounded-lg bg-[#202631] text-[#F4F1E9] border border-[#262C38] text-xs hover:bg-[#262C38] transition"
          >
            Inspect Errors
          </Link>
        </section>
      )}

      {/* 4. Recent Experiments & Model Pool Showcase */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left: Recent Experiments Table (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Archive className="w-4 h-4 text-[#5CE1D2]" />
              <h2 className="text-base font-bold text-[#F4F1E9] tracking-tight">
                Recent Experiments
              </h2>
            </div>
            <Link
              href="/history"
              className="text-xs text-[#5CE1D2] hover:underline flex items-center space-x-1 font-mono"
            >
              <span>View All Records</span>
              <ChevronRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          {loading ? (
            <div className="p-12 text-center rounded-2xl bg-[#181C24] border border-[#262C38] text-xs text-[#9BA5B4] flex items-center justify-center space-x-2">
              <Loader2 className="w-4 h-4 animate-spin text-[#5CE1D2]" />
              <span>Loading saved experiment database...</span>
            </div>
          ) : recentJobs.length === 0 ? (
            <div className="p-12 text-center rounded-2xl bg-[#181C24] border border-[#262C38] text-[#9BA5B4] space-y-3">
              <Shield className="w-10 h-10 text-[#9BA5B4]/40 mx-auto" />
              <h3 className="text-sm font-semibold text-[#F4F1E9]">No Experiments Recorded Yet</h3>
              <p className="text-xs text-[#9BA5B4] max-w-sm mx-auto">
                Launch a new experiment or interact with Guardian Intelligence to generate your first cloaked artwork.
              </p>
              <Link
                href="/protect"
                className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-[#5CE1D2] text-[#101217] text-xs font-bold transition hover:opacity-90"
              >
                <PlusCircle className="w-3.5 h-3.5" />
                <span>Start First Experiment</span>
              </Link>
            </div>
          ) : (
            <div className="space-y-2.5">
              {recentJobs.map((j) => (
                <Link
                  key={j.job_id}
                  href={`/results/${j.job_id}`}
                  className="p-4 rounded-xl bg-[#181C24] border border-[#262C38] hover:border-[#5CE1D2]/40 transition-all flex items-center justify-between group block text-xs"
                >
                  <div className="flex items-center space-x-3.5 min-w-0">
                    <div className="w-10 h-10 rounded-lg bg-[#101217] border border-[#262C38] overflow-hidden shrink-0 flex items-center justify-center">
                      {(j.protected_image_url || j.original_image_url) ? (
                        <img
                          src={(j.protected_image_url || j.original_image_url)!.startsWith("http") ? (j.protected_image_url || j.original_image_url)! : `${BACKEND_ROOT_URL}${j.protected_image_url || j.original_image_url}`}
                          alt="Artwork"
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <Shield className="w-4 h-4 text-[#9BA5B4]" />
                      )}
                    </div>
                    <div className="truncate">
                      <div className="flex items-center space-x-2">
                        <span className="font-mono font-bold text-[#F4F1E9] truncate">
                          {j.job_id.slice(0, 8)}...
                        </span>
                        <span
                          className={`text-[9px] uppercase px-1.5 py-0.5 rounded font-mono font-semibold ${
                            j.status === "completed"
                              ? "bg-[#5CE1D2]/10 text-[#5CE1D2] border border-[#5CE1D2]/20"
                              : j.status === "failed"
                              ? "bg-[#FF8C79]/10 text-[#FF8C79] border border-[#FF8C79]/20"
                              : "bg-[#A9A5FF]/10 text-[#A9A5FF] border border-[#A9A5FF]/20"
                          }`}
                        >
                          {j.status}
                        </span>
                      </div>
                      <span className="text-[10px] text-[#9BA5B4] font-mono">
                        {j.created_at ? new Date(j.created_at).toLocaleString() : ""} &bull; {j.surrogate_model || "ResNet-50"}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center space-x-4 shrink-0 font-mono text-[11px]">
                    {j.overall_protection_score !== null && j.overall_protection_score !== undefined && (
                      <div className="text-right">
                        <span className="text-[9px] text-[#9BA5B4] uppercase block">Protection</span>
                        <span className="text-sm font-bold text-[#5CE1D2]">
                          {j.overall_protection_score}%
                        </span>
                      </div>
                    )}
                    <ChevronRight className="w-4 h-4 text-[#9BA5B4] group-hover:text-[#5CE1D2] transition-colors" />
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>

        {/* Right: Model Pool & Quick Research Navigation (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-[#A9A5FF]" />
            <h2 className="text-base font-bold text-[#F4F1E9] tracking-tight">
              Evaluated Model Pool
            </h2>
          </div>

          <div className="space-y-3">
            {[
              {
                name: "ResNet-50",
                type: "Deep Residual CNN",
                status: "Active Surrogate",
                statusColor: "text-[#5CE1D2]",
                detail: "38 WikiArt Style Classes &bull; Epoch 3 Checkpoint",
              },
              {
                name: "ConvNeXt-Tiny",
                type: "Modern Convolutional",
                status: "Transfer Target",
                statusColor: "text-[#A9A5FF]",
                detail: "Exp_F_ConvNeXt_Tiny_best.pth &bull; 768 Feature Dim",
              },
              {
                name: "ViT-B/16",
                type: "Vision Transformer",
                status: "Transfer Target",
                statusColor: "text-[#A9A5FF]",
                detail: "torchvision.models.vit_b_16(DEFAULT) &bull; Self-Attention",
              },
              {
                name: "CLIP-ViT-B/32",
                type: "Contrastive Multimodal",
                status: "Zero-Shot Evaluator",
                statusColor: "text-[#5CE1D2]",
                detail: "OpenAI Pretrained Hub &bull; Canonical Style Prompts",
              },
            ].map((model, idx) => (
              <div
                key={idx}
                className="p-3.5 rounded-xl bg-[#181C24] border border-[#262C38] flex items-center justify-between text-xs"
              >
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="font-bold text-[#F4F1E9]">{model.name}</span>
                    <span className="text-[10px] text-[#9BA5B4] font-mono">({model.type})</span>
                  </div>
                  <span
                    className="text-[10px] text-[#9BA5B4] block mt-0.5"
                    dangerouslySetInnerHTML={{ __html: model.detail }}
                  />
                </div>
                <span className={`text-[10px] font-mono font-semibold uppercase ${model.statusColor}`}>
                  {model.status}
                </span>
              </div>
            ))}
          </div>

          {/* Quick Laboratory Jump Cards */}
          <div className="pt-2 grid grid-cols-2 gap-3">
            <Link
              href="/arena"
              className="p-3.5 rounded-xl bg-[#202631] border border-[#262C38] hover:border-[#5CE1D2]/30 transition group block text-xs"
            >
              <Cpu className="w-4 h-4 text-[#A9A5FF] mb-1.5" />
              <span className="font-semibold text-[#F4F1E9] block group-hover:text-[#5CE1D2] transition">
                Model Arena
              </span>
              <span className="text-[10px] text-[#9BA5B4]">Cross-architecture disruption</span>
            </Link>

            <Link
              href="/robustness"
              className="p-3.5 rounded-xl bg-[#202631] border border-[#262C38] hover:border-[#5CE1D2]/30 transition group block text-xs"
            >
              <RefreshCw className="w-4 h-4 text-[#5CE1D2] mb-1.5" />
              <span className="font-semibold text-[#F4F1E9] block group-hover:text-[#5CE1D2] transition">
                Robustness
              </span>
              <span className="text-[10px] text-[#9BA5B4]">JPEG & resizing endurance</span>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}

function PlusCircle(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M8 12h8" />
      <path d="M12 8v8" />
    </svg>
  );
}
