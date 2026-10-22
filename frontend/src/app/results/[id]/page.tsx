"use client";

import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  Shield,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  Download,
  FileText,
  Layers,
  ArrowLeft,
  Cpu,
  RefreshCw,
  Eye,
  BarChart3,
  Lock,
  Sparkles,
  Info,
  Check,
  X,
} from "lucide-react";
import {
  getJobStatus,
  ProtectionJobResponse,
  BACKEND_ROOT_URL,
} from "@/lib/api";
import ArtworkMicroscope from "@/components/ArtworkMicroscope";
import LivingMetrics from "@/components/LivingMetrics";

export default function ResultsPage() {
  const params = useParams();
  const jobId = params?.id as string;

  const [job, setJob] = useState<ProtectionJobResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"comparison" | "models" | "robustness" | "quality">("comparison");

  useEffect(() => {
    if (!jobId) return;

    let isMounted = true;
    let interval: ReturnType<typeof setInterval>;

    const fetchStatus = async () => {
      try {
        const data = await getJobStatus(jobId);
        if (isMounted) {
          setJob(data);
          setLoading(false);
          if (data.status === "completed" || data.status === "failed") {
            clearInterval(interval);
          }
        }
      } catch (err: any) {
        if (isMounted) {
          setError(err?.response?.data?.detail || "Failed to load protection job results.");
          setLoading(false);
          clearInterval(interval);
        }
      }
    };

    fetchStatus();
    interval = setInterval(fetchStatus, 3000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [jobId]);

  if (loading) {
    return (
      <div className="min-h-[60vh] flex flex-col items-center justify-center space-y-4">
        <Loader2 className="w-10 h-10 text-sky-400 animate-spin" />
        <p className="text-slate-400 text-sm">Loading protection job details and evaluation benchmarks...</p>
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="max-w-2xl mx-auto my-20 p-6 rounded-2xl bg-slate-900/80 border border-slate-800 text-center space-y-4">
        <AlertTriangle className="w-10 h-10 text-amber-400 mx-auto" />
        <h3 className="text-lg font-bold text-white">Job Not Found or Error</h3>
        <p className="text-sm text-slate-400">{error || "Unable to locate job ID."}</p>
        <Link
          href="/protect"
          className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-sm font-semibold text-white transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Protect Studio</span>
        </Link>
      </div>
    );
  }

  const cls = job.classification;
  const qm = job.quality_metrics;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      {/* Breadcrumb & Navigation */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <Link
          href="/protect"
          className="inline-flex items-center space-x-2 text-xs font-semibold text-slate-400 hover:text-white transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Protect Studio</span>
        </Link>

        <div className="flex items-center space-x-2 text-xs text-slate-400 font-mono">
          <span>Job ID:</span>
          <span className="text-sky-400 font-bold">{job.job_id}</span>
        </div>
      </div>

      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-[#181C24] border border-[#262C38] rounded-2xl p-6 backdrop-blur-sm">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wider ${
                job.status === "completed"
                  ? "bg-[#5CE1D2]/10 text-[#5CE1D2] border border-[#5CE1D2]/20"
                  : job.status === "processing"
                  ? "bg-[#A9A5FF]/10 text-[#A9A5FF] border border-[#A9A5FF]/20 animate-pulse"
                  : job.status === "failed"
                  ? "bg-[#FF8C79]/10 text-[#FF8C79] border border-[#FF8C79]/20"
                  : "bg-[#FF8C79]/10 text-[#FF8C79] border border-[#FF8C79]/20"
              }`}
            >
              {job.status}
            </span>
            <span className="text-xs text-[#9BA5B4]">
              Created {new Date(job.created_at).toLocaleTimeString()}
            </span>
          </div>
          <h1 className="text-2xl font-bold text-[#F4F1E9] tracking-tight">
            Adversarial Cloaking Verification & Evaluation Dashboard
          </h1>
          <p className="text-xs text-[#9BA5B4] max-w-2xl">
            Empirical neural disruption benchmarks, mathematical quality verification, cross-model transferability, and transformation survival analysis.
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center gap-3">
          {job.status === "completed" && job.protected_image_url && (
            <a
              href={job.protected_image_url.startsWith("http") ? job.protected_image_url : `${BACKEND_ROOT_URL}${job.protected_image_url}`}
              download={`cloaked_${job.job_id}.png`}
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-[#5CE1D2] to-[#38A89C] hover:from-[#5CE1D2] hover:to-[#5CE1D2] text-xs font-bold text-[#101217] shadow-md shadow-[#5CE1D2]/20 transition-all cursor-pointer"
            >
              <Download className="w-4 h-4 fill-current" />
              <span>Download Protected Image</span>
            </a>
          )}
          {job.status === "completed" && (
            <a
              href={`${BACKEND_ROOT_URL}/api/reports/${job.job_id}/csv`}
              download={`evaluation_${job.job_id}.csv`}
              className="inline-flex items-center space-x-2 px-3 py-2.5 rounded-xl bg-[#202631] hover:bg-[#262C38] text-xs font-semibold text-[#F4F1E9] border border-[#262C38] transition-colors cursor-pointer"
            >
              <FileText className="w-4 h-4 text-[#5CE1D2]" />
              <span>Export CSV</span>
            </a>
          )}
        </div>
      </div>

      {/* Progress Card if pending or processing */}
      {job.status !== "completed" && (
        <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 text-center space-y-4 shadow-xl">
          <div className="w-12 h-12 rounded-2xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center text-sky-400 mx-auto">
            <Loader2 className="w-6 h-6 animate-spin" />
          </div>
          <div className="space-y-1">
            <h3 className="text-lg font-semibold text-white">
              {job.status === "pending"
                ? "Job Queued in Protection Engine..."
                : "Synthesizing Native Adversarial Perturbations..."}
            </h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto">
              Running bounded PGD optimization on deep feature representations at native asset resolution without downsampling blur.
            </p>
          </div>
          <div className="max-w-md mx-auto w-full bg-slate-800 rounded-full h-2 overflow-hidden">
            <div
              className="bg-gradient-to-r from-sky-500 to-cyan-400 h-2 transition-all duration-500"
              style={{ width: `${Math.max(25, job.progress * 100)}%` }}
            />
          </div>
        </div>
      )}

      {/* Completed Metrics Summary Banner */}
      {job.status === "completed" && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1 text-center">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">Overall Protection</span>
            <span className="text-2xl font-bold font-mono text-emerald-400">
              {job.overall_protection_score !== null && job.overall_protection_score !== undefined ? `${job.overall_protection_score}%` : "—"}
            </span>
            <span className="text-[10px] text-emerald-500/80 block font-medium">Composite Assessment</span>
          </div>
          <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1 text-center">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">AI Disruption</span>
            <span className="text-2xl font-bold font-mono text-sky-400">
              {job.ai_disruption_score !== null && job.ai_disruption_score !== undefined ? `${job.ai_disruption_score}%` : "—"}
            </span>
            <span className="text-[10px] text-sky-400/80 block font-medium">
              {cls?.top1_prediction_changed ? "Top-1 Style Disrupted" : "Confidence Depressed"}
            </span>
          </div>
          <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1 text-center">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">Transferability</span>
            <span className="text-2xl font-bold font-mono text-cyan-400">
              {job.transferability_score !== null && job.transferability_score !== undefined ? `${job.transferability_score}%` : "—"}
            </span>
            <span className="text-[10px] text-cyan-400/80 block font-medium">Black-Box Survival</span>
          </div>
          <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-1 text-center">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">Structural Fidelity (SSIM)</span>
            <span className="text-2xl font-bold font-mono text-indigo-400">
              {qm?.ssim !== null && qm?.ssim !== undefined ? qm.ssim : "—"}
            </span>
            <span className="text-[10px] text-indigo-400/80 block font-medium">Native Resolution Preserved</span>
          </div>
        </div>
      )}

      {/* Interactive Artwork Microscope Workspace */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-bold text-white uppercase tracking-wider font-mono">
              Artwork Microscope &bull; Interactive Inspection Canvas
            </span>
          </div>
          <span className="text-xs text-slate-400 font-mono hidden sm:inline">
            Synchronized Zoom &amp; Pan Active &bull; Native Resolution
          </span>
        </div>

        <ArtworkMicroscope
          originalUrl={job.original_image_url || ""}
          cloakedUrl={job.protected_image_url}
          differenceMapUrl={job.difference_map_url}
          metrics={{
            ssim: qm?.ssim,
            psnr_db: qm?.psnr_db,
            lpips: qm?.lpips,
            linf_norm: qm?.linf_norm,
            epsilon_passed: qm?.epsilon_passed,
          }}
        />
      </div>

      {/* CLASSIFICATION RESULTS PANEL */}
      {cls && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <h3 className="text-base font-bold text-white flex items-center space-x-2">
                <span>Target Classifier Style Prediction Assessment</span>
                {cls.top1_prediction_changed ? (
                  <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <Check className="w-3.5 h-3.5" />
                    <span>Top-1 Prediction Disrupted</span>
                  </span>
                ) : (
                  <span className="inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    <span>Confidence Depressed (Top-1 Unchanged)</span>
                  </span>
                )}
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Both original and cloaked assets evaluated through the identical surrogate ResNet-50 classifier under standard 224&times;224 ImageNet preprocessing.
              </p>
            </div>

            <div className="flex items-center space-x-4 text-xs font-mono">
              <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Orig Class Confidence Drop:</span>
                <span className="text-emerald-400 font-bold text-sm">
                  {cls.confidence_drop > 0 ? `-${cls.confidence_drop_percent}%` : "0.0%"}
                </span>
              </div>
              <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                <span className="text-slate-400 block text-[10px]">Embedding Cosine Distance:</span>
                <span className="text-sky-400 font-bold text-sm">
                  {cls.embedding_cosine_distance}
                </span>
              </div>
            </div>
          </div>

          {/* Top-5 Predictions Comparison Side by Side */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Original Top 5 */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-300">Original Predictions (Top-5)</span>
                <span className="text-[11px] text-slate-500 font-mono">Top-1: {cls.original_prediction.style_name}</span>
              </div>
              <div className="space-y-2">
                {cls.original_prediction.top5.map((p, idx) => (
                  <div key={idx} className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-white font-medium">{idx + 1}. {p.style_name}</span>
                      <span className="font-mono text-slate-400">{p.confidence_percent}%</span>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div
                        className="bg-slate-400 h-1.5 rounded-full"
                        style={{ width: `${Math.min(100, p.confidence_percent)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Cloaked Top 5 */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-sky-400">Cloaked Predictions (Top-5)</span>
                <span className="text-[11px] text-sky-400 font-mono">Top-1: {cls.cloaked_prediction.style_name}</span>
              </div>
              <div className="space-y-2">
                {cls.cloaked_prediction.top5.map((p, idx) => {
                  const isOrigClass = p.style_name === cls.original_prediction.style_name;
                  return (
                    <div
                      key={idx}
                      className={`p-2.5 rounded-xl border space-y-1 ${
                        isOrigClass
                          ? "bg-amber-950/20 border-amber-500/30"
                          : "bg-slate-950/80 border-slate-800"
                      }`}
                    >
                      <div className="flex items-center justify-between text-xs">
                        <span className={`font-medium ${isOrigClass ? "text-amber-300" : "text-sky-300"}`}>
                          {idx + 1}. {p.style_name} {isOrigClass && "(Original Class)"}
                        </span>
                        <span className="font-mono text-sky-400">{p.confidence_percent}%</span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                        <div
                          className={`h-1.5 rounded-full ${
                            isOrigClass ? "bg-amber-500" : "bg-gradient-to-r from-sky-500 to-cyan-400"
                          }`}
                          style={{ width: `${Math.min(100, p.confidence_percent)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Scientific Clarification Alert */}
          <div className="flex items-start space-x-3 p-3.5 rounded-xl bg-slate-950/90 border border-slate-800 text-xs text-slate-400">
            <Info className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
            <p>
              <strong className="text-slate-200">Scientific Context: </strong>
              {cls.scientific_disclaimer}
            </p>
          </div>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="flex border-b border-slate-800 space-x-4">
        {[
          { id: "comparison", label: "Image Quality & Difference Map", icon: Eye },
          { id: "models", label: "Multi-Model Transferability", icon: Cpu },
          { id: "robustness", label: "Transformation Robustness", icon: RefreshCw },
          { id: "quality", label: "Mathematical Verification & Reports", icon: BarChart3 },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center space-x-2 py-3 px-3 text-sm font-medium border-b-2 transition-colors ${
                isActive
                  ? "border-sky-400 text-sky-400"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* TAB 1: Visual Quality & Difference Map */}
      {activeTab === "comparison" && (
        <div className="space-y-6">
          {/* Difference Map */}
          {job.difference_map_url && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-sm font-semibold text-white">Amplified Perturbation Pattern (Difference Map)</span>
                  <p className="text-xs text-slate-400">
                    High-frequency adversarial gradients synthesized to disrupt deep neural feature extractors.
                  </p>
                </div>
                <span className="text-[11px] px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 font-mono">
                  Amplified &times;32 (&epsilon;=8/255)
                </span>
              </div>
              <div className="rounded-xl overflow-hidden border border-slate-800 bg-slate-950 flex items-center justify-center min-h-[240px] p-2">
                <img
                  src={job.difference_map_url.startsWith("http") ? job.difference_map_url : `${BACKEND_ROOT_URL}${job.difference_map_url}`}
                  alt="Amplified Difference Map"
                  className="max-h-[320px] w-auto max-w-full object-contain rounded-lg"
                />
              </div>
            </div>
          )}

          {/* Image Quality Metrics Grid */}
          {qm && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-base font-semibold text-white">Mathematical Quality & Perturbation Metrics</h3>
                <span
                  className={`text-xs px-2.5 py-0.5 rounded-full font-semibold ${
                    qm.epsilon_passed
                      ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                      : "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                  }`}
                >
                  &epsilon;-Constraint: {qm.epsilon_passed ? "PASSED (Strictly Bounded)" : "FAILED"}
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 text-center space-y-1">
                  <span className="text-xs text-slate-400 block font-medium">SSIM</span>
                  <span className="text-lg font-mono font-bold text-sky-400">{qm.ssim !== null ? qm.ssim : "—"}</span>
                  <span className="text-[10px] text-slate-500 block">Structural Fidelity (&gt;0.90)</span>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 text-center space-y-1">
                  <span className="text-xs text-slate-400 block font-medium">PSNR</span>
                  <span className="text-lg font-mono font-bold text-sky-400">{qm.psnr_db !== null ? `${qm.psnr_db} dB` : "—"}</span>
                  <span className="text-[10px] text-slate-500 block">Signal-to-Noise (&gt;30dB)</span>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 text-center space-y-1">
                  <span className="text-xs text-slate-400 block font-medium">LPIPS</span>
                  <span className="text-lg font-mono font-bold text-sky-400">{qm.lpips !== null ? qm.lpips : "N/A"}</span>
                  <span className="text-[10px] text-slate-500 block">Perceptual Distance (&lt;0.08)</span>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 text-center space-y-1">
                  <span className="text-xs text-slate-400 block font-medium">L-&infin; Norm</span>
                  <span className="text-lg font-mono font-bold text-emerald-400">{qm.linf_norm !== null ? qm.linf_norm : "—"}</span>
                  <span className="text-[10px] text-slate-500 block">Max Pixel Delta (&le;8/255)</span>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 text-center space-y-1">
                  <span className="text-xs text-slate-400 block font-medium">L2 Norm (RMSE)</span>
                  <span className="text-lg font-mono font-bold text-sky-400">{qm.rmse !== null ? qm.rmse : "—"}</span>
                  <span className="text-[10px] text-slate-500 block">Root Mean Square Error</span>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 text-center space-y-1">
                  <span className="text-xs text-slate-400 block font-medium">Mean Abs Error</span>
                  <span className="text-lg font-mono font-bold text-sky-400">{qm.mae !== null ? qm.mae : "—"}</span>
                  <span className="text-[10px] text-slate-500 block">Per-Pixel Shift</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: Multi-Model Evaluation */}
      {activeTab === "models" && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-6">
          <div>
            <h3 className="text-base font-semibold text-white">Cross-Architecture Transferability Benchmark</h3>
            <p className="text-xs text-slate-400 mt-1">
              Evaluates the disruption of classification and deep embeddings when the cloaked artwork is inspected by non-surrogate target architectures.
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950 text-slate-400 border-b border-slate-800 uppercase font-mono">
                <tr>
                  <th className="py-3 px-4">Model Architecture</th>
                  <th className="py-3 px-4">Role / Paradigm</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Original Prediction</th>
                  <th className="py-3 px-4">Cloaked Prediction</th>
                  <th className="py-3 px-4">Conf Drop</th>
                  <th className="py-3 px-4">Cosine Dist</th>
                  <th className="py-3 px-4">Disrupted</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {job.evaluations && job.evaluations.length > 0 ? (
                  job.evaluations.map((ev, idx) => {
                    const isClip = ev.model_name.toLowerCase().includes("clip") || ev.evaluation_type === "zero_shot_multimodal";
                    const statusColor =
                      ev.status === "evaluated"
                        ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                        : ev.status === "loading"
                        ? "bg-sky-500/10 text-sky-400 border-sky-500/20"
                        : ev.status === "not_configured"
                        ? "bg-slate-500/10 text-slate-400 border-slate-500/20"
                        : ev.status === "unavailable"
                        ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                        : "bg-rose-500/10 text-rose-400 border-rose-500/20";

                    return (
                      <tr key={idx} className="hover:bg-slate-950/40">
                        <td className="py-3.5 px-4 font-bold text-white">
                          <div className="space-y-1">
                            <span className="uppercase">{ev.model_name.replace(/_/g, "-")}</span>
                            {isClip && (
                              <div className="flex items-center space-x-1">
                                <span className="px-1.5 py-0.2 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 text-[9px] font-sans font-medium">
                                  Zero-Shot Multimodal
                                </span>
                              </div>
                            )}
                            {isClip && ev.prompt_template && (
                              <span className="block text-[10px] text-slate-400 font-sans italic">
                                Prompt: &ldquo;{ev.prompt_template}&rdquo;
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="py-3.5 px-4">
                          {ev.is_surrogate ? (
                            <span className="px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20 text-[10px]">
                              Surrogate (White-Box)
                            </span>
                          ) : isClip ? (
                            <span className="px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 text-[10px]">
                              Target (Multimodal Dual)
                            </span>
                          ) : (
                            <span className="px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20 text-[10px]">
                              Target (Black-Box)
                            </span>
                          )}
                        </td>
                        <td className="py-3.5 px-4">
                          <div className="space-y-0.5">
                            <span className={`inline-block px-2 py-0.5 rounded border text-[10px] font-bold ${statusColor}`}>
                              {ev.status}
                            </span>
                            {ev.status_detail && (
                              <span className="block text-[10px] text-slate-400 max-w-[160px] truncate" title={ev.status_detail}>
                                {ev.status_detail}
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="py-3.5 px-4 text-slate-300">
                          {ev.original_top1_class ? (
                            <span>
                              {ev.original_top1_class} ({((ev.original_top1_confidence || 0) * 100).toFixed(1)}%)
                            </span>
                          ) : (
                            <span className="text-slate-600">—</span>
                          )}
                        </td>
                        <td className="py-3.5 px-4 text-sky-300">
                          {ev.protected_top1_class ? (
                            <span>
                              {ev.protected_top1_class} ({((ev.protected_top1_confidence || 0) * 100).toFixed(1)}%)
                            </span>
                          ) : (
                            <span className="text-slate-600">—</span>
                          )}
                        </td>
                        <td className="py-3.5 px-4 text-emerald-400 font-bold">
                          {ev.confidence_drop !== null && ev.confidence_drop !== undefined
                            ? `-${(ev.confidence_drop * 100).toFixed(1)}%`
                            : "—"}
                        </td>
                        <td className="py-3.5 px-4 text-indigo-400 font-bold">
                          {ev.embedding_cosine_distance !== null && ev.embedding_cosine_distance !== undefined
                            ? ev.embedding_cosine_distance.toFixed(4)
                            : "—"}
                        </td>
                        <td className="py-3.5 px-4">
                          {ev.classification_disrupted !== null && ev.classification_disrupted !== undefined ? (
                            ev.classification_disrupted ? (
                              <span className="text-emerald-400 flex items-center space-x-1">
                                <Check className="w-3.5 h-3.5" />
                                <span>YES</span>
                              </span>
                            ) : (
                              <span className="text-amber-400 flex items-center space-x-1">
                                <X className="w-3.5 h-3.5" />
                                <span>NO</span>
                              </span>
                            )
                          ) : (
                            <span className="text-slate-600">—</span>
                          )}
                        </td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan={8} className="py-6 text-center text-slate-500">
                      No multi-model evaluation records available for this job.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-400 space-y-1">
            <span className="text-white font-semibold flex items-center space-x-1.5">
              <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
              <span>Zero-Shot Multimodal Style Evaluation</span>
            </span>
            <p className="leading-relaxed text-[11px]">
              CLIP ViT-B/32 is evaluated using zero-shot image-text alignment against 38 canonical WikiArt style prompts (e.g. &ldquo;a painting in Impressionism style&rdquo;). Rather than a closed-set linear classification head, disruption reflects how the adversarial perturbation weakens joint embedding similarity with the ground-truth artistic movement.
            </p>
          </div>
        </div>
      )}

      {/* TAB 3: Robustness Benchmarks */}
      {activeTab === "robustness" && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-6">
          <div>
            <h3 className="text-base font-semibold text-white">Empirical Transformation Robustness Benchmarks</h3>
            <p className="text-xs text-slate-400 mt-1">
              Tests whether the adversarial cloaking perturbation survives typical online digital image processing, social media compression, and filtering operations.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {job.robustness && job.robustness.length > 0 ? (
              job.robustness.map((rb, idx) => (
                <div key={idx} className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-white">{rb.transformation_name}</span>
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded font-semibold ${
                        rb.prediction_disrupted
                          ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                          : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                      }`}
                    >
                      {rb.status}
                    </span>
                  </div>

                  <div className="text-xs text-slate-400 space-y-1 font-mono">
                    <div className="flex justify-between">
                      <span className="text-slate-500">Post-Transform Style:</span>
                      <span className="text-slate-200">{rb.top1_class_after_transform || "—"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Post-Transform Conf:</span>
                      <span className="text-sky-400">
                        {rb.top1_confidence_after_transform ? `${(rb.top1_confidence_after_transform * 100).toFixed(1)}%` : "—"}
                      </span>
                    </div>
                    {rb.ssim_after_transform !== null && rb.ssim_after_transform !== undefined && (
                      <div className="flex justify-between">
                        <span className="text-slate-500">Transform SSIM:</span>
                        <span className="text-indigo-400">{rb.ssim_after_transform}</span>
                      </div>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <div className="col-span-full py-8 text-center text-slate-500 text-xs">
                No transformation robustness benchmarks recorded.
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 4: Mathematical Verification & Reports */}
      {activeTab === "quality" && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <h3 className="text-base font-semibold text-white">Reproducible Evaluation Report & Audit</h3>
              <p className="text-xs text-slate-400 mt-1">
                Download structured machine-readable reports containing exact model weights, parameters, timestamps, and metric scores.
              </p>
            </div>
            <div className="flex items-center space-x-3">
              <a
                href={`${BACKEND_ROOT_URL}/api/reports/${job.job_id}`}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center space-x-2 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-white border border-slate-700 transition-colors"
              >
                <FileText className="w-4 h-4 text-sky-400" />
                <span>View Full JSON Report</span>
              </a>
              <a
                href={`${BACKEND_ROOT_URL}/api/reports/${job.job_id}/csv`}
                download={`ai_protect_summary_${job.job_id}.csv`}
                className="inline-flex items-center space-x-2 px-3 py-2 rounded-xl bg-sky-600 hover:bg-sky-500 text-xs font-semibold text-white shadow-md shadow-sky-600/20 transition-all cursor-pointer"
              >
                <Download className="w-4 h-4" />
                <span>Download CSV Summary</span>
              </a>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <span className="font-bold text-white block">Adversarial Bound & Pixel Convention</span>
              <ul className="space-y-1.5 text-slate-400">
                <li>&bull; <strong className="text-slate-200">Pixel Space:</strong> Native sRGB in range [0.0, 1.0]</li>
                <li>&bull; <strong className="text-slate-200">Configured &epsilon;:</strong> {qm?.epsilon_configured || "8/255 (0.03137)"}</li>
                <li>&bull; <strong className="text-slate-200">Measured L-&infin; Norm:</strong> {qm?.linf_norm || "—"}</li>
                <li>&bull; <strong className="text-slate-200">Quantization Tolerance:</strong> &plusmn;1/255 (0.00392) for 8-bit PNG</li>
                <li>&bull; <strong className="text-slate-200">Strict Bound Check:</strong> {qm?.epsilon_passed ? "PASSED" : "FAILED"}</li>
              </ul>
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <span className="font-bold text-white block">Scientific Integrity Declarations</span>
              <ul className="space-y-1.5 text-slate-400">
                <li>&bull; All classifier predictions and confidences are derived from forward passes on model checkpoints.</li>
                <li>&bull; No synthetic metrics or fabricated results are generated.</li>
                <li>&bull; Adversarial cloaking provides empirical disruption, not cryptographic security against all AI systems.</li>
                <li>&bull; Preservation of native resolution ensures artwork sharpness without downsampling blur.</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
