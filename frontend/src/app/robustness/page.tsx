"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  RefreshCw,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  ShieldAlert,
  ShieldCheck,
  Eye,
  Layers,
  ArrowRight,
  Maximize2,
  Info,
  Sparkles,
} from "lucide-react";
import {
  listJobs,
  getJobStatus,
  testCustomRobustness,
  JobSummaryItem,
  ProtectionJobResponse,
  CustomRobustnessResponse,
  BACKEND_ROOT_URL,
} from "@/lib/api";

function RobustnessContent() {
  const searchParams = useSearchParams();
  const initialJobId = searchParams.get("job");

  const [jobs, setJobs] = useState<JobSummaryItem[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const [currentJob, setCurrentJob] = useState<ProtectionJobResponse | null>(null);
  const [loadingJob, setLoadingJob] = useState(false);

  // Transformation states
  const [activeTransform, setActiveTransform] = useState<"jpeg" | "resize" | "crop" | "blur" | "brightness" | "contrast">("jpeg");
  const [paramValue, setParamValue] = useState<number>(75);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<CustomRobustnessResponse | null>(null);
  const [testError, setTestError] = useState<string | null>(null);

  // Transform config metadata
  const transformConfigs: Record<string, { label: string; min: number; max: number; step: number; default: number; unit: string; desc: string }> = {
    jpeg: { label: "JPEG Compression", min: 10, max: 100, step: 1, default: 75, unit: "Quality (10-100)", desc: "Lossy DCT quantisation applied by social platforms." },
    resize: { label: "Downscale & Upscale", min: 0.2, max: 1.0, step: 0.05, default: 0.8, unit: "Scale Factor (0.2x - 1.0x)", desc: "Bilinear downscale followed by recovery." },
    crop: { label: "Center Crop", min: 0.5, max: 0.99, step: 0.01, default: 0.9, unit: "Crop Fraction (50% - 99%)", desc: "Removes peripheral pixels and resizes to canvas." },
    blur: { label: "Gaussian Blur", min: 0.1, max: 5.0, step: 0.1, default: 1.0, unit: "Radius Sigma (0.1 - 5.0 px)", desc: "Low-pass spatial filter attenuating high frequencies." },
    brightness: { label: "Brightness Shift", min: -0.5, max: 0.5, step: 0.05, default: 0.1, unit: "Shift (-0.5 to +0.5)", desc: "Additive luminance shift across RGB channels." },
    contrast: { label: "Contrast Scale", min: 0.5, max: 2.0, step: 0.1, default: 1.2, unit: "Factor (0.5x - 2.0x)", desc: "Linear expansion around midtone luminance." },
  };

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
      .catch((err) => console.error("Error loading jobs for robustness:", err));
  }, [initialJobId]);

  useEffect(() => {
    if (!selectedJobId) return;
    setLoadingJob(true);
    setTestResult(null);
    setTestError(null);

    getJobStatus(selectedJobId)
      .then((data) => {
        setCurrentJob(data);
      })
      .catch((err) => {
        console.error("Error loading job details:", err);
      })
      .finally(() => setLoadingJob(false));
  }, [selectedJobId]);

  const handleTransformChange = (tf: any) => {
    setActiveTransform(tf);
    setParamValue(transformConfigs[tf].default);
    setTestResult(null);
  };

  const handleRunTest = async () => {
    if (!selectedJobId) return;
    setTesting(true);
    setTestError(null);

    try {
      const res = await testCustomRobustness(selectedJobId, {
        transformation: activeTransform,
        param_value: paramValue,
      });
      setTestResult(res);
    } catch (err: any) {
      setTestError(err?.response?.data?.detail || "Transformation benchmark test failed.");
    } finally {
      setTesting(false);
    }
  };

  const getFullUrl = (path?: string | null) => {
    if (!path) return "";
    return path.startsWith("http") ? path : `${BACKEND_ROOT_URL}${path}`;
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-16">
      {/* Workspace Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-[#262C38] pb-6">
        <div>
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-[#5CE1D2]/10 border border-[#5CE1D2]/20 text-[#5CE1D2] text-xs font-semibold mb-2">
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Real-World Survival Suite</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-[#F4F1E9] tracking-tight">
            Robustness Journey &amp; Transformation Playground
          </h1>
          <p className="text-xs sm:text-sm text-[#9BA5B4] mt-1 max-w-2xl">
            Empirically evaluate whether adversarial perturbations withstand social media compression, image resizing, cropping, Gaussian blurring, and color adjustments.
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

      {loadingJob ? (
        <div className="min-h-[40vh] flex flex-col items-center justify-center space-y-4">
          <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
          <p className="text-slate-400 text-xs font-mono">Loading artwork models and robustness checkpoints...</p>
        </div>
      ) : !currentJob ? (
        <div className="p-12 text-center rounded-2xl bg-[#0b0f19] border border-slate-800 space-y-4">
          <AlertTriangle className="w-10 h-10 text-amber-400 mx-auto" />
          <h3 className="text-base font-bold text-white">No Completed Experiments Available</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            You must have at least one completed adversarial cloaking job to test real-world transformations.
          </p>
          <Link
            href="/protect"
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-cyan-500 text-slate-950 text-xs font-bold hover:bg-cyan-400 transition-colors"
          >
            <span>Launch New Experiment</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      ) : (
        <div className="space-y-8">
          {/* Main Playground Grid: Sliders & Live Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column: Interactive Controls */}
            <div className="lg:col-span-5 space-y-5">
              <div className="p-6 rounded-2xl bg-[#0b0f19] border border-slate-800 space-y-6">
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                  <div className="flex items-center space-x-2">
                    <Sliders className="w-4 h-4 text-cyan-400" />
                    <span className="text-sm font-bold text-white">Transformation Parameters</span>
                  </div>
                  <span className="text-[10px] text-cyan-400 font-mono">Differentiable Simulation</span>
                </div>

                {/* Transformation Tabs */}
                <div className="grid grid-cols-3 gap-2">
                  {Object.entries(transformConfigs).map(([key, cfg]) => {
                    const isActive = activeTransform === key;
                    return (
                      <button
                        key={key}
                        onClick={() => handleTransformChange(key)}
                        className={`p-2.5 rounded-xl text-left border transition-all ${
                          isActive
                            ? "bg-cyan-500/10 border-cyan-500 text-cyan-300 shadow-sm"
                            : "bg-slate-950/80 border-slate-800/80 text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                        }`}
                      >
                        <span className="text-[11px] font-bold block truncate">{cfg.label}</span>
                        <span className="text-[9px] text-slate-500 block truncate">{key}</span>
                      </button>
                    );
                  })}
                </div>

                {/* Selected Transformation Description */}
                <p className="text-xs text-slate-400 bg-slate-950/60 p-3 rounded-xl border border-slate-800/60">
                  {transformConfigs[activeTransform].desc}
                </p>

                {/* Interactive Slider */}
                <div className="space-y-3 p-4 rounded-xl bg-slate-950 border border-slate-800/80">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-300">
                      {transformConfigs[activeTransform].label}
                    </span>
                    <span className="font-mono font-bold text-cyan-400 text-sm">
                      {paramValue} {activeTransform === "resize" ? "x" : activeTransform === "blur" ? "px" : ""}
                    </span>
                  </div>

                  <input
                    type="range"
                    min={transformConfigs[activeTransform].min}
                    max={transformConfigs[activeTransform].max}
                    step={transformConfigs[activeTransform].step}
                    value={paramValue}
                    onChange={(e) => setParamValue(parseFloat(e.target.value))}
                    className="w-full accent-cyan-400 bg-slate-800 h-2 rounded-lg cursor-pointer"
                  />

                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>Min: {transformConfigs[activeTransform].min}</span>
                    <span>Preset Default: {transformConfigs[activeTransform].default}</span>
                    <span>Max: {transformConfigs[activeTransform].max}</span>
                  </div>
                </div>

                {/* Preset Quick Buttons */}
                <div className="space-y-2">
                  <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block font-bold">
                    Quick Scenario Presets
                  </span>
                  <div className="flex flex-wrap gap-2">
                    <button
                      onClick={() => {
                        setActiveTransform("jpeg");
                        setParamValue(65);
                      }}
                      className="px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-[10px] text-slate-300 hover:border-cyan-500/50 hover:text-cyan-300 transition-colors"
                    >
                      Social Compression (JPEG 65)
                    </button>
                    <button
                      onClick={() => {
                        setActiveTransform("resize");
                        setParamValue(0.5);
                      }}
                      className="px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-[10px] text-slate-300 hover:border-cyan-500/50 hover:text-cyan-300 transition-colors"
                    >
                      Heavy Rescale (50%)
                    </button>
                    <button
                      onClick={() => {
                        setActiveTransform("crop");
                        setParamValue(0.75);
                      }}
                      className="px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-[10px] text-slate-300 hover:border-cyan-500/50 hover:text-cyan-300 transition-colors"
                    >
                      Thumbnail Crop (75%)
                    </button>
                    <button
                      onClick={() => {
                        setActiveTransform("blur");
                        setParamValue(1.5);
                      }}
                      className="px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-[10px] text-slate-300 hover:border-cyan-500/50 hover:text-cyan-300 transition-colors"
                    >
                      Lens Softening (1.5px)
                    </button>
                  </div>
                </div>

                {/* Execute Button */}
                <button
                  onClick={handleRunTest}
                  disabled={testing}
                  className="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-sky-400 hover:from-cyan-400 hover:to-sky-300 text-slate-950 font-bold text-xs shadow-lg shadow-cyan-500/20 transition-all flex items-center justify-center space-x-2 cursor-pointer disabled:opacity-50"
                >
                  {testing ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Applying &amp; Re-Evaluating Forward Pass...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      <span>Simulate Transformation &amp; Benchmark</span>
                    </>
                  )}
                </button>

                {testError && (
                  <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-center space-x-2">
                    <AlertTriangle className="w-4 h-4 shrink-0" />
                    <span>{testError}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column: Visual Verification & Real-time Outcome */}
            <div className="lg:col-span-7 space-y-5">
              <div className="p-6 rounded-2xl bg-[#0b0f19] border border-slate-800 space-y-5">
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                  <div className="flex items-center space-x-2">
                    <Eye className="w-4 h-4 text-cyan-400" />
                    <span className="text-sm font-bold text-white">Visual &amp; Neural Disruption Verdict</span>
                  </div>
                  <Link
                    href={`/results/${currentJob.job_id}`}
                    className="text-xs text-cyan-400 hover:underline flex items-center space-x-1"
                  >
                    <span>Full Microscope</span>
                    <ArrowRight className="w-3 h-3" />
                  </Link>
                </div>

                {/* Visual Viewport: Cloaked Asset vs Transformed Asset */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Baseline Cloaked Image */}
                  <div className="space-y-2">
                    <span className="text-xs font-semibold text-slate-300 flex items-center justify-between">
                      <span>Baseline Cloaked Asset</span>
                      <span className="text-[10px] text-cyan-400 font-mono">Pre-Transform</span>
                    </span>
                    <div className="h-60 rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center p-1">
                      {currentJob.protected_image_url ? (
                        <img
                          src={getFullUrl(currentJob.protected_image_url)}
                          alt="Protected Artwork"
                          className="max-h-full max-w-full object-contain rounded-lg"
                        />
                      ) : (
                        <span className="text-xs text-slate-600 font-mono">No Image</span>
                      )}
                    </div>
                    <div className="text-[11px] font-mono text-slate-400 flex justify-between px-1">
                      <span>Top-1 Style:</span>
                      <span className="text-cyan-300 font-bold">
                        {currentJob.classification?.cloaked_prediction?.style_name || "Disrupted"}
                      </span>
                    </div>
                  </div>

                  {/* Transformed Preview */}
                  <div className="space-y-2">
                    <span className="text-xs font-semibold text-slate-300 flex items-center justify-between">
                      <span>Post-Transformation Asset</span>
                      <span className="text-[10px] text-indigo-400 font-mono">Simulated</span>
                    </span>
                    <div className="h-60 rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center p-1 relative">
                      {testResult?.preview_url ? (
                        <img
                          src={getFullUrl(testResult.preview_url)}
                          alt="Transformed Artwork Preview"
                          className="max-h-full max-w-full object-contain rounded-lg"
                        />
                      ) : currentJob.protected_image_url ? (
                        <img
                          src={getFullUrl(currentJob.protected_image_url)}
                          alt="Transformed Artwork"
                          className="max-h-full max-w-full object-contain rounded-lg"
                          style={{
                            filter:
                              activeTransform === "blur"
                                ? `blur(${paramValue}px)`
                                : activeTransform === "brightness"
                                ? `brightness(${1 + paramValue})`
                                : activeTransform === "contrast"
                                ? `contrast(${paramValue})`
                                : undefined,
                          }}
                        />
                      ) : null}

                      {testing && (
                        <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm flex flex-col items-center justify-center space-y-2">
                          <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
                          <span className="text-xs font-mono text-cyan-300">Evaluating Classifier...</span>
                        </div>
                      )}
                    </div>
                    <div className="text-[11px] font-mono text-slate-400 flex justify-between px-1">
                      <span>Post-Transform Style:</span>
                      <span className="text-white font-bold">
                        {testResult?.top1_class_after_transform || "Click Benchmark to Test"}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Outcome Verdict Card */}
                {testResult ? (
                  <div
                    className={`p-4 rounded-xl border space-y-3 ${
                      testResult.prediction_disrupted
                        ? "bg-emerald-950/20 border-emerald-500/30 text-emerald-300"
                        : "bg-amber-950/20 border-amber-500/30 text-amber-300"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        {testResult.prediction_disrupted ? (
                          <ShieldCheck className="w-5 h-5 text-emerald-400" />
                        ) : (
                          <ShieldAlert className="w-5 h-5 text-amber-400" />
                        )}
                        <span className="font-bold text-sm">
                          {testResult.prediction_disrupted
                            ? "Adversarial Cloak Retained Across Transformation!"
                            : "Perturbation Attenuated — Partial Style Recovery Detected"}
                        </span>
                      </div>
                      <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-black/40">
                        Survival: {testResult.retained_protection_rate}%
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-3 text-xs font-mono pt-2 border-t border-white/10">
                      <div>
                        <span className="text-slate-400 block text-[10px]">Transformed Top-1:</span>
                        <span className="font-bold text-white">{testResult.top1_class_after_transform}</span>
                      </div>
                      <div>
                        <span className="text-slate-400 block text-[10px]">Model Confidence:</span>
                        <span className="font-bold text-cyan-400">
                          {(testResult.top1_confidence_after_transform * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-400 block text-[10px]">Transform SSIM:</span>
                        <span className="font-bold text-indigo-300">
                          {testResult.ssim_after_transform ?? "N/A"}
                        </span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 text-xs text-slate-400 flex items-start space-x-3">
                    <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                    <p>
                      Adjust the slider above and click <strong>Simulate Transformation &amp; Benchmark</strong> to perform an on-the-fly PyTorch forward pass with standard ImageNet normalization.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Persistent Benchmark Ledger (Standard Pre-computed Transformations) */}
          <div className="p-6 rounded-2xl bg-[#0b0f19] border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-white">Standard Multi-Transformation Audit Suite</h3>
                <p className="text-xs text-slate-400">
                  Pre-computed empirical tests executed automatically during the synthesis job.
                </p>
              </div>
              <span className="text-xs font-mono text-cyan-400">Surrogate Model: ResNet-50</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 pt-2">
              {currentJob.robustness && currentJob.robustness.length > 0 ? (
                currentJob.robustness.map((rb, idx) => (
                  <div
                    key={idx}
                    className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 hover:border-slate-700 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-white">{rb.transformation_name}</span>
                      <span
                        className={`text-[9px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider ${
                          rb.prediction_disrupted
                            ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                            : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                        }`}
                      >
                        {rb.prediction_disrupted ? "Protected" : "Vulnerable"}
                      </span>
                    </div>

                    <div className="space-y-1 text-xs font-mono text-slate-400">
                      <div className="flex justify-between">
                        <span className="text-slate-500">Predicted:</span>
                        <span className="text-slate-200">{rb.top1_class_after_transform || "—"}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Confidence:</span>
                        <span className="text-cyan-400">
                          {rb.top1_confidence_after_transform
                            ? `${(rb.top1_confidence_after_transform * 100).toFixed(1)}%`
                            : "—"}
                        </span>
                      </div>
                      {rb.ssim_after_transform !== null && rb.ssim_after_transform !== undefined && (
                        <div className="flex justify-between">
                          <span className="text-slate-500">SSIM:</span>
                          <span className="text-indigo-400">{rb.ssim_after_transform}</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <div className="col-span-full py-8 text-center text-slate-500 text-xs">
                  No baseline robustness ledger records available for this run.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function RobustnessPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-[40vh] flex flex-col items-center justify-center space-y-4">
          <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
          <p className="text-slate-400 text-xs font-mono">Loading Robustness Playground...</p>
        </div>
      }
    >
      <RobustnessContent />
    </Suspense>
  );
}
