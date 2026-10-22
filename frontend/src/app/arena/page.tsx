"use client";

import React, { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  Cpu,
  Layers,
  Shield,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  ArrowRight,
  TrendingDown,
  BarChart3,
  Check,
  X,
  Sparkles,
  Info,
  ExternalLink,
} from "lucide-react";
import { Suspense } from "react";
import {
  listJobs,
  getJobStatus,
  JobSummaryItem,
  ProtectionJobResponse,
  BACKEND_ROOT_URL,
} from "@/lib/api";

function ArenaContent() {
  const searchParams = useSearchParams();
  const initialJobId = searchParams.get("job");

  const [jobs, setJobs] = useState<JobSummaryItem[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const [currentJob, setCurrentJob] = useState<ProtectionJobResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listJobs()
      .then((data) => {
        const completed = data.filter((j) => j.status === "completed");
        setJobs(completed);

        if (initialJobId && completed.some((j) => j.job_id === initialJobId)) {
          setSelectedJobId(initialJobId);
        } else if (completed.length > 0) {
          setSelectedJobId(completed[0].job_id);
        }
      })
      .catch((err) => console.error("Error loading jobs for Arena:", err))
      .finally(() => setLoading(false));
  }, [initialJobId]);

  useEffect(() => {
    if (!selectedJobId) return;

    getJobStatus(selectedJobId)
      .then((data) => {
        setCurrentJob(data);
      })
      .catch((err) => console.error("Error loading job details:", err));
  }, [selectedJobId]);

  const getFullUrl = (path?: string | null) => {
    if (!path) return "";
    return path.startsWith("http") ? path : `${BACKEND_ROOT_URL}${path}`;
  };

  const cls = currentJob?.classification;
  const evaluations = currentJob?.evaluations || [];

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-16">
      {/* Workspace Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-[#262C38] pb-6">
        <div>
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-[#5CE1D2]/10 border border-[#5CE1D2]/20 text-[#5CE1D2] text-xs font-semibold mb-2">
            <Cpu className="w-3.5 h-3.5" />
            <span>Neural Benchmark Arena</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-[#F4F1E9] tracking-tight">
            Multi-Model Arena &amp; Transferability Lab
          </h1>
          <p className="text-xs sm:text-sm text-[#9BA5B4] mt-1 max-w-2xl">
            Evaluate cross-architecture adversarial transferability across ResNet-50 (Surrogate CNN), ConvNeXt-Tiny (Modern Black-Box CNN), ViT-B/16 (Vision Transformer), and CLIP-ViT-B/32 (Multimodal Dual-Encoder).
          </p>
        </div>

        {/* Experiment Selector */}
        <div className="flex items-center space-x-2">
          <span className="text-xs text-[#9BA5B4] font-mono">Artwork Run:</span>
          <select
            value={selectedJobId}
            onChange={(e) => setSelectedJobId(e.target.value)}
            className="bg-[#181C24] border border-[#262C38] rounded-xl px-3 py-2 text-xs text-[#F4F1E9] focus:outline-none focus:border-[#5CE1D2] max-w-xs font-mono"
          >
            {jobs.map((j) => (
              <option key={j.job_id} value={j.job_id}>
                {j.original_filename} ({j.job_id.slice(0, 8)})
              </option>
            ))}
          </select>
        </div>
      </div>

      {!currentJob ? (
        <div className="min-h-[40vh] flex flex-col items-center justify-center space-y-4">
          <Loader2 className="w-8 h-8 text-[#5CE1D2] animate-spin" />
          <p className="text-[#9BA5B4] text-xs font-mono">Loading model inference evaluations...</p>
        </div>
      ) : (
        <div className="space-y-8">
          {/* Artwork Summary Bar */}
          <div className="p-5 rounded-2xl bg-[#181C24] border border-[#262C38] flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center space-x-4">
              <div className="w-14 h-14 rounded-xl overflow-hidden bg-[#101217] border border-[#262C38] shrink-0">
                <img
                  src={getFullUrl(currentJob.protected_image_url || currentJob.original_image_url)}
                  alt="Artwork Preview"
                  className="w-full h-full object-cover"
                />
              </div>
              <div>
                <h3 className="text-sm font-bold text-[#F4F1E9]">
                  {currentJob.image_id ? `Asset ${currentJob.job_id.slice(0, 8)}` : "Target Artwork"}
                </h3>
                <div className="flex items-center space-x-2 text-xs text-[#9BA5B4] font-mono">
                  <span>Surrogate:</span>
                  <span className="text-[#5CE1D2] uppercase font-semibold">
                    {currentJob.config_id ? "ResNet-50" : "ResNet-50"}
                  </span>
                  <span className="text-[#262C38]">&bull;</span>
                  <span>Dataset:</span>
                  <span className="text-[#F4F1E9]">WikiArt (38 Classes)</span>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-3 text-xs font-mono">
              <div className="px-3 py-1.5 rounded-xl bg-[#101217] border border-[#262C38] text-center">
                <span className="text-[#9BA5B4] block text-[9px] uppercase">Transferability</span>
                <span className="text-[#A9A5FF] font-bold text-sm">
                  {currentJob.transferability_score !== null ? `${currentJob.transferability_score}%` : "—"}
                </span>
              </div>
              <div className="px-3 py-1.5 rounded-xl bg-[#101217] border border-[#262C38] text-center">
                <span className="text-[#9BA5B4] block text-[9px] uppercase">AI Disruption</span>
                <span className="text-[#5CE1D2] font-bold text-sm">
                  {currentJob.ai_disruption_score !== null ? `${currentJob.ai_disruption_score}%` : "—"}
                </span>
              </div>
              <div className="px-3 py-1.5 rounded-xl bg-[#101217] border border-[#262C38] text-center">
                <span className="text-[#9BA5B4] block text-[9px] uppercase">SSIM Fidelity</span>
                <span className="text-[#5CE1D2] font-bold text-sm">
                  {currentJob.quality_metrics?.ssim ?? "—"}
                </span>
              </div>
            </div>
          </div>

          {/* Neural Architecture Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Card 1: ResNet-50 (Surrogate White-Box) */}
            <div className="p-6 rounded-2xl bg-[#0b0f19] border border-cyan-500/40 shadow-lg shadow-cyan-500/5 space-y-5 flex flex-col justify-between">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 text-[10px] font-mono font-bold uppercase tracking-wider">
                    Surrogate (White-Box)
                  </span>
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                </div>

                <div>
                  <h3 className="text-base font-bold text-white">ResNet-50</h3>
                  <p className="text-[11px] text-slate-400">
                    Residual network backbone with skip connections. Used directly to compute adversarial loss gradients during cloaking synthesis.
                  </p>
                </div>

                {cls ? (
                  <div className="space-y-3 pt-3 border-t border-slate-800/80 text-xs font-mono">
                    <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                      <span className="text-slate-500 text-[10px] block">Original Top-1 Style:</span>
                      <div className="flex justify-between items-center">
                        <span className="text-white font-semibold">{cls.original_prediction.style_name}</span>
                        <span className="text-slate-400">{cls.original_prediction.confidence_percent}%</span>
                      </div>
                    </div>

                    <div className="p-3 rounded-xl bg-cyan-950/20 border border-cyan-500/30 space-y-1">
                      <span className="text-cyan-400 text-[10px] block">Cloaked Top-1 Style:</span>
                      <div className="flex justify-between items-center">
                        <span className="text-cyan-200 font-bold">{cls.cloaked_prediction.style_name}</span>
                        <span className="text-cyan-400">{cls.cloaked_prediction.confidence_percent}%</span>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-center pt-1">
                      <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                        <span className="text-slate-500 text-[9px] block">Confidence Drop</span>
                        <span className="text-emerald-400 font-bold">
                          {cls.confidence_drop > 0 ? `-${cls.confidence_drop_percent}%` : "0.0%"}
                        </span>
                      </div>
                      <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                        <span className="text-slate-500 text-[9px] block">Cosine Distance</span>
                        <span className="text-sky-400 font-bold">{cls.embedding_cosine_distance}</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <span className="text-xs text-slate-500 font-mono">Pending inference</span>
                )}
              </div>

              <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 text-[11px] flex items-center justify-between">
                <span className="text-slate-400">Classification Outcome:</span>
                <span className="text-emerald-400 font-bold flex items-center space-x-1">
                  <Check className="w-3.5 h-3.5" />
                  <span>Disrupted</span>
                </span>
              </div>
            </div>

            {/* Card 2: ConvNeXt-Tiny (Target Black-Box CNN) */}
            {(() => {
              const evConv = evaluations.find((e) => e.model_name.includes("convnext"));
              return (
                <div className="p-6 rounded-2xl bg-[#0b0f19] border border-purple-500/30 space-y-5 flex flex-col justify-between">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20 text-[10px] font-mono font-bold uppercase tracking-wider">
                        Target (Black-Box CNN)
                      </span>
                      <span className="w-2 h-2 rounded-full bg-purple-400" />
                    </div>

                    <div>
                      <h3 className="text-base font-bold text-white">ConvNeXt-Tiny</h3>
                      <p className="text-[11px] text-slate-400">
                        Modern pure-convolutional network with $7\times 7$ depthwise kernels and inverted bottleneck design.
                      </p>
                    </div>

                    {evConv ? (
                      <div className="space-y-3 pt-3 border-t border-slate-800/80 text-xs font-mono">
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                          <span className="text-slate-500 text-[10px] block">Original Top-1 Style:</span>
                          <div className="flex justify-between items-center">
                            <span className="text-white font-semibold">{evConv.original_top1_class || "—"}</span>
                            <span className="text-slate-400">
                              {evConv.original_top1_confidence ? `${(evConv.original_top1_confidence * 100).toFixed(1)}%` : "—"}
                            </span>
                          </div>
                        </div>

                        <div className="p-3 rounded-xl bg-purple-950/20 border border-purple-500/30 space-y-1">
                          <span className="text-purple-400 text-[10px] block">Cloaked Top-1 Style:</span>
                          <div className="flex justify-between items-center">
                            <span className="text-purple-200 font-bold">{evConv.protected_top1_class || "—"}</span>
                            <span className="text-purple-400">
                              {evConv.protected_top1_confidence ? `${(evConv.protected_top1_confidence * 100).toFixed(1)}%` : "—"}
                            </span>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-center pt-1">
                          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                            <span className="text-slate-500 text-[9px] block">Confidence Drop</span>
                            <span className="text-emerald-400 font-bold">
                              {evConv.confidence_drop !== null && evConv.confidence_drop !== undefined
                                ? `-${(evConv.confidence_drop * 100).toFixed(1)}%`
                                : "—"}
                            </span>
                          </div>
                          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                            <span className="text-slate-500 text-[9px] block">Cosine Distance</span>
                            <span className="text-purple-400 font-bold">
                              {evConv.embedding_cosine_distance ?? "—"}
                            </span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center text-xs text-slate-500 font-mono">
                        Not evaluated in this run
                      </div>
                    )}
                  </div>

                  <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 text-[11px] flex items-center justify-between">
                    <span className="text-slate-400">Transferability Verdict:</span>
                    <span
                      className={`font-bold flex items-center space-x-1 ${
                        evConv?.classification_disrupted ? "text-emerald-400" : "text-amber-400"
                      }`}
                    >
                      {evConv?.classification_disrupted ? (
                        <>
                          <Check className="w-3.5 h-3.5" />
                          <span>Transferred (Disrupted)</span>
                        </>
                      ) : (
                        <>
                          <X className="w-3.5 h-3.5" />
                          <span>Resistant</span>
                        </>
                      )}
                    </span>
                  </div>
                </div>
              );
            })()}

            {/* Card 3: ViT-B/16 (Vision Transformer) */}
            {(() => {
              const evVit = evaluations.find(
                (e) => e.model_name.toLowerCase().includes("vit_b_16") ||
                       (e.model_name.toLowerCase().includes("vit") && !e.model_name.toLowerCase().includes("clip"))
              );
              return (
                <div className="p-6 rounded-2xl bg-[#0b0f19] border border-sky-500/30 space-y-5 flex flex-col justify-between">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20 text-[10px] font-mono font-bold uppercase tracking-wider">
                        Target (Black-Box ViT)
                      </span>
                      <span className="w-2 h-2 rounded-full bg-sky-400" />
                    </div>

                    <div>
                      <h3 className="text-base font-bold text-white">Vision Transformer (ViT-B/16)</h3>
                      <p className="text-[11px] text-slate-400">
                        Multi-head self-attention architecture operating on non-overlapping $16\times 16$ image patches without local spatial convolutions.
                      </p>
                    </div>

                    {evVit ? (
                      <div className="space-y-3 pt-3 border-t border-slate-800/80 text-xs font-mono">
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                          <span className="text-slate-500 text-[10px] block">Original Top-1 Style:</span>
                          <div className="flex justify-between items-center">
                            <span className="text-white font-semibold">{evVit.original_top1_class || "—"}</span>
                            <span className="text-slate-400">
                              {evVit.original_top1_confidence ? `${(evVit.original_top1_confidence * 100).toFixed(1)}%` : "—"}
                            </span>
                          </div>
                        </div>

                        <div className="p-3 rounded-xl bg-sky-950/20 border border-sky-500/30 space-y-1">
                          <span className="text-sky-400 text-[10px] block">Cloaked Top-1 Style:</span>
                          <div className="flex justify-between items-center">
                            <span className="text-sky-200 font-bold">{evVit.protected_top1_class || "—"}</span>
                            <span className="text-sky-400">
                              {evVit.protected_top1_confidence ? `${(evVit.protected_top1_confidence * 100).toFixed(1)}%` : "—"}
                            </span>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-center pt-1">
                          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                            <span className="text-slate-500 text-[9px] block">Confidence Drop</span>
                            <span className="text-emerald-400 font-bold">
                              {evVit.confidence_drop !== null && evVit.confidence_drop !== undefined
                                ? `-${(evVit.confidence_drop * 100).toFixed(1)}%`
                                : "—"}
                            </span>
                          </div>
                          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                            <span className="text-slate-500 text-[9px] block">Cosine Distance</span>
                            <span className="text-sky-400 font-bold">
                              {evVit.embedding_cosine_distance ?? "—"}
                            </span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center text-xs text-slate-500 font-mono">
                        Not evaluated in this run
                      </div>
                    )}
                  </div>

                  <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 text-[11px] flex items-center justify-between">
                    <span className="text-slate-400">Transferability Verdict:</span>
                    <span
                      className={`font-bold flex items-center space-x-1 ${
                        evVit?.classification_disrupted ? "text-emerald-400" : "text-amber-400"
                      }`}
                    >
                      {evVit?.classification_disrupted ? (
                        <>
                          <Check className="w-3.5 h-3.5" />
                          <span>Transferred (Disrupted)</span>
                        </>
                      ) : (
                        <>
                          <X className="w-3.5 h-3.5" />
                          <span>Resistant</span>
                        </>
                      )}
                    </span>
                  </div>
                </div>
              );
            })()}

            {/* Card 4: CLIP-ViT-B/32 (Multimodal Dual-Encoder) */}
            {(() => {
              const evClip = evaluations.find((e) => e.model_name.toLowerCase().includes("clip"));
              return (
                <div className="p-6 rounded-2xl bg-[#0b0f19] border border-cyan-500/30 space-y-5 flex flex-col justify-between">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 text-[10px] font-mono font-bold uppercase tracking-wider">
                        Target (Multimodal Dual)
                      </span>
                      <span className="w-2 h-2 rounded-full bg-cyan-400" />
                    </div>

                    <div>
                      <h3 className="text-base font-bold text-white">CLIP ViT-B/32</h3>
                      <p className="text-[11px] text-slate-400">
                        Zero-shot multimodal evaluator scoring text prompt similarity: <span className="text-cyan-400 italic">&ldquo;a painting in {"{style}"} style&rdquo;</span>.
                      </p>
                    </div>

                    {evClip && evClip.status === "evaluated" ? (
                      <div className="space-y-3 pt-3 border-t border-slate-800/80 text-xs font-mono">
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                          <span className="text-slate-500 text-[10px] block">Original Top-1 Style:</span>
                          <div className="flex justify-between items-center">
                            <span className="text-white font-semibold">{evClip.original_top1_class || "—"}</span>
                            <span className="text-slate-400">
                              {evClip.original_top1_confidence ? `${(evClip.original_top1_confidence * 100).toFixed(1)}%` : "—"}
                            </span>
                          </div>
                        </div>

                        <div className="p-3 rounded-xl bg-cyan-950/20 border border-cyan-500/30 space-y-1">
                          <span className="text-cyan-400 text-[10px] block">Cloaked Top-1 Style:</span>
                          <div className="flex justify-between items-center">
                            <span className="text-cyan-200 font-bold">{evClip.protected_top1_class || "—"}</span>
                            <span className="text-cyan-400">
                              {evClip.protected_top1_confidence ? `${(evClip.protected_top1_confidence * 100).toFixed(1)}%` : "—"}
                            </span>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-center pt-1">
                          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                            <span className="text-slate-500 text-[9px] block">Confidence Drop</span>
                            <span className="text-emerald-400 font-bold">
                              {evClip.confidence_drop !== null && evClip.confidence_drop !== undefined
                                ? `-${(evClip.confidence_drop * 100).toFixed(1)}%`
                                : "—"}
                            </span>
                          </div>
                          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                            <span className="text-slate-500 text-[9px] block">Cosine Distance</span>
                            <span className="text-cyan-400 font-bold">
                              {evClip.embedding_cosine_distance !== null && evClip.embedding_cosine_distance !== undefined
                                ? evClip.embedding_cosine_distance.toFixed(4)
                                : "—"}
                            </span>
                          </div>
                        </div>
                      </div>
                    ) : evClip ? (
                      <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center space-y-1 text-xs font-mono">
                        <span className="text-amber-400 uppercase font-bold block">{evClip.status}</span>
                        <span className="text-[10px] text-slate-500 block">{evClip.status_detail || "Model not evaluated"}</span>
                      </div>
                    ) : (
                      <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center text-xs text-slate-500 font-mono">
                        Not evaluated in this run
                      </div>
                    )}
                  </div>

                  <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 text-[11px] flex items-center justify-between">
                    <span className="text-slate-400">Transferability Verdict:</span>
                    <span
                      className={`font-bold flex items-center space-x-1 ${
                        evClip?.classification_disrupted
                          ? "text-emerald-400"
                          : evClip?.status === "evaluated"
                          ? "text-amber-400"
                          : "text-slate-500"
                      }`}
                    >
                      {evClip?.classification_disrupted ? (
                        <>
                          <Check className="w-3.5 h-3.5" />
                          <span>Transferred (Disrupted)</span>
                        </>
                      ) : evClip?.status === "evaluated" ? (
                        <>
                          <X className="w-3.5 h-3.5" />
                          <span>Resistant</span>
                        </>
                      ) : (
                        <span>Unavailable</span>
                      )}
                    </span>
                  </div>
                </div>
              );
            })()}
          </div>

          {/* Cross-Architecture Scientific Evaluation Table */}
          <div className="p-6 rounded-2xl bg-[#0b0f19] border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-white">Cross-Inductive-Bias Transfer Matrix</h3>
                <p className="text-xs text-slate-400">
                  Measuring empirical vulnerability across distinct model topologies and feature spaces.
                </p>
              </div>
              <Link
                href={`/results/${currentJob.job_id}`}
                className="text-xs text-cyan-400 hover:underline flex items-center space-x-1"
              >
                <span>Artwork Microscope</span>
                <ExternalLink className="w-3 h-3" />
              </Link>
            </div>

            <div className="overflow-x-auto pt-2">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-slate-950 text-slate-400 border-b border-slate-800 uppercase text-[10px]">
                  <tr>
                    <th className="py-3 px-4">Architecture</th>
                    <th className="py-3 px-4">Inductive Bias</th>
                    <th className="py-3 px-4">Threat Model</th>
                    <th className="py-3 px-4">Pre-Cloak Top-1</th>
                    <th className="py-3 px-4">Post-Cloak Top-1</th>
                    <th className="py-3 px-4">Confidence Shift</th>
                    <th className="py-3 px-4">Cosine Distance</th>
                    <th className="py-3 px-4">Disruption</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {/* ResNet-50 Row */}
                  <tr className="hover:bg-slate-950/50">
                    <td className="py-3.5 px-4 font-bold text-white uppercase">ResNet-50</td>
                    <td className="py-3.5 px-4 text-slate-400">Residual Conv (Local)</td>
                    <td className="py-3.5 px-4">
                      <span className="px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 text-[10px]">
                        White-Box Surrogate
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-slate-300">
                      {cls?.original_prediction.style_name} ({cls?.original_prediction.confidence_percent}%)
                    </td>
                    <td className="py-3.5 px-4 text-cyan-300 font-bold">
                      {cls?.cloaked_prediction.style_name} ({cls?.cloaked_prediction.confidence_percent}%)
                    </td>
                    <td className="py-3.5 px-4 text-emerald-400 font-bold">
                      {cls ? `-${cls.confidence_drop_percent}%` : "—"}
                    </td>
                    <td className="py-3.5 px-4 text-sky-400 font-bold">
                      {cls?.embedding_cosine_distance ?? "—"}
                    </td>
                    <td className="py-3.5 px-4">
                      <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px] font-bold">
                        {cls?.top1_prediction_changed ? "Disrupted" : "Depressed"}
                      </span>
                    </td>
                  </tr>

                  {/* Other Evaluated Models */}
                  {evaluations
                    .filter((ev) => !ev.is_surrogate)
                    .map((ev, idx) => {
                      const isClip = ev.model_name.toLowerCase().includes("clip") || ev.evaluation_type === "zero_shot_multimodal";
                      const isVit = ev.model_name.toLowerCase().includes("vit") && !isClip;
                      const inductiveBias = isClip
                        ? "Multimodal Dual-Encoder (Zero-Shot)"
                        : isVit
                        ? "Self-Attention (Global)"
                        : "Depthwise Conv (Local)";
                      const badgeColor = isClip
                        ? "bg-cyan-500/10 text-cyan-400"
                        : "bg-purple-500/10 text-purple-400";
                      const roleLabel = isClip ? "Multimodal Target" : "Black-Box Target";

                      return (
                        <tr key={idx} className="hover:bg-slate-950/50">
                          <td className="py-3.5 px-4 font-bold text-white uppercase">
                            <div>
                              <span>{ev.model_name.replace(/_/g, "-")}</span>
                              {isClip && (
                                <span className="block text-[9px] text-cyan-400/80 font-sans normal-case italic">
                                  &ldquo;a painting in {"{style}"} style&rdquo;
                                </span>
                              )}
                            </div>
                          </td>
                          <td className="py-3.5 px-4 text-slate-400">{inductiveBias}</td>
                          <td className="py-3.5 px-4">
                            <span className={`px-2 py-0.5 rounded text-[10px] ${badgeColor}`}>
                              {roleLabel}
                            </span>
                          </td>
                          <td className="py-3.5 px-4 text-slate-300">
                            {ev.original_top1_class ? (
                              `${ev.original_top1_class} (${((ev.original_top1_confidence || 0) * 100).toFixed(1)}%)`
                            ) : (
                              <span className="text-slate-600">—</span>
                            )}
                          </td>
                          <td className="py-3.5 px-4 text-purple-300 font-bold">
                            {ev.protected_top1_class ? (
                              `${ev.protected_top1_class} (${((ev.protected_top1_confidence || 0) * 100).toFixed(1)}%)`
                            ) : (
                              <span className="text-slate-600">—</span>
                            )}
                          </td>
                          <td className="py-3.5 px-4 text-emerald-400 font-bold">
                            {ev.confidence_drop !== null && ev.confidence_drop !== undefined
                              ? `-${(ev.confidence_drop * 100).toFixed(1)}%`
                              : "—"}
                          </td>
                          <td className="py-3.5 px-4 text-purple-400 font-bold">
                            {ev.embedding_cosine_distance !== null && ev.embedding_cosine_distance !== undefined
                              ? ev.embedding_cosine_distance.toFixed(4)
                              : "—"}
                          </td>
                          <td className="py-3.5 px-4">
                            {ev.status === "evaluated" ? (
                              <span
                                className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                  ev.classification_disrupted
                                    ? "bg-emerald-500/10 text-emerald-400"
                                    : "bg-amber-500/10 text-amber-400"
                                }`}
                              >
                                {ev.classification_disrupted ? "Transferred" : "Resistant"}
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-400 uppercase">
                                {ev.status}
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Scientific Research Note */}
          <div className="p-5 rounded-2xl bg-[#0b0f19] border border-slate-800 flex items-start space-x-3 text-xs text-slate-400">
            <Info className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="text-white font-bold block">Scientific Transferability Insight</span>
              <p className="leading-relaxed">
                Adversarial perturbations generated against convolutional surrogates transfer most effectively to models sharing similar convolutional inductive biases (e.g. ConvNeXt). Cross-paradigm transfer to Vision Transformers is naturally lower due to patch projection and self-attention mechanisms, representing an important area of ongoing adversarial robustness research.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function MultiModelArenaPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-[40vh] flex flex-col items-center justify-center space-y-4">
          <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
          <p className="text-slate-400 text-xs font-mono">Loading Neural Arena...</p>
        </div>
      }
    >
      <ArenaContent />
    </Suspense>
  );
}
