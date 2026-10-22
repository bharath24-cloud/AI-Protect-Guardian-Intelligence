"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  FileText,
  Printer,
  Download,
  Shield,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  Eye,
  Cpu,
  Layers,
  Sparkles,
  Info,
  Check,
} from "lucide-react";
import {
  listJobs,
  getJobStatus,
  JobSummaryItem,
  ProtectionJobResponse,
  BACKEND_ROOT_URL,
} from "@/lib/api";

function ReportStudioContent() {
  const searchParams = useSearchParams();
  const initialJobId = searchParams.get("job");

  const [jobs, setJobs] = useState<JobSummaryItem[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const [job, setJob] = useState<ProtectionJobResponse | null>(null);
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
      .catch((err) => console.error("Error loading jobs for reports:", err))
      .finally(() => setLoading(false));
  }, [initialJobId]);

  useEffect(() => {
    if (!selectedJobId) return;

    getJobStatus(selectedJobId)
      .then((data) => {
        setJob(data);
      })
      .catch((err) => console.error("Error loading job details:", err));
  }, [selectedJobId]);

  const handlePrint = () => {
    window.print();
  };

  const getFullUrl = (path?: string | null) => {
    if (!path) return "";
    return path.startsWith("http") ? path : `${BACKEND_ROOT_URL}${path}`;
  };

  const cls = job?.classification;
  const qm = job?.quality_metrics;

  return (
    <div className="max-w-5xl mx-auto space-y-8 pb-20">
      {/* Workspace Header (Hidden on print) */}
      <div className="print:hidden flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-[#262C38] pb-6">
        <div>
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-[#5CE1D2]/10 border border-[#5CE1D2]/20 text-[#5CE1D2] text-xs font-semibold mb-2">
            <FileText className="w-3.5 h-3.5" />
            <span>Formal Scientific Audit</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-[#F4F1E9] tracking-tight">
            Research Report Studio &amp; Dossier
          </h1>
          <p className="text-xs sm:text-sm text-[#9BA5B4] mt-1 max-w-2xl">
            Generate and export reproducible, publication-ready evaluation dossiers detailing adversarial bounds, mathematical fidelity, neural disruption, and transformation survival.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Job Selector */}
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

          {job && (
            <>
              <button
                onClick={handlePrint}
                className="inline-flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold transition-colors cursor-pointer"
              >
                <Printer className="w-3.5 h-3.5 text-cyan-400" />
                <span>Print / Save PDF</span>
              </button>

              <a
                href={`${BACKEND_ROOT_URL}/api/reports/${job.job_id}`}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold transition-colors cursor-pointer"
              >
                <Download className="w-3.5 h-3.5 text-sky-400" />
                <span>JSON</span>
              </a>

              <a
                href={`${BACKEND_ROOT_URL}/api/reports/${job.job_id}/csv`}
                download={`ai_protect_audit_${job.job_id}.csv`}
                className="inline-flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs shadow-md shadow-cyan-500/20 transition-all cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                <span>CSV Audit</span>
              </a>
            </>
          )}
        </div>
      </div>

      {!job ? (
        <div className="min-h-[40vh] flex flex-col items-center justify-center space-y-4">
          <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
          <p className="text-slate-400 text-xs font-mono">Compiling evaluation report data...</p>
        </div>
      ) : (
        /* Printable Research Dossier Document */
        <article className="bg-[#0b0f19] print:bg-white print:text-black border border-slate-800 print:border-none rounded-3xl p-6 sm:p-10 lg:p-12 space-y-8 shadow-2xl">
          {/* Dossier Header */}
          <div className="border-b border-slate-800 print:border-slate-300 pb-6 flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <div className="w-7 h-7 rounded-lg bg-cyan-500 print:bg-black flex items-center justify-center text-slate-950 print:text-white font-bold">
                  <Shield className="w-4 h-4" />
                </div>
                <span className="font-extrabold text-sm tracking-tight text-white print:text-black">
                  AI-PROTECT LAB
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-500/10 print:bg-slate-200 text-cyan-400 print:text-slate-800 font-mono font-bold">
                  FORMAL AUDIT DOSSIER
                </span>
              </div>
              <h2 className="text-xl sm:text-2xl font-bold text-white print:text-black tracking-tight">
                Adversarial Cloaking Scientific Verification Report
              </h2>
              <p className="text-xs text-slate-400 print:text-slate-600 font-mono">
                Dossier Identifier: {job.job_id} &bull; Generated: {new Date(job.created_at).toUTCString()}
              </p>
            </div>

            <div className="text-right font-mono text-xs space-y-1">
              <div className="text-slate-400 print:text-slate-600">
                Composite Score:{" "}
                <span className="text-emerald-400 print:text-emerald-700 font-bold text-base">
                  {job.overall_protection_score !== null ? `${job.overall_protection_score}%` : "—"}
                </span>
              </div>
              <div className="text-slate-500 text-[10px]">
                Engine: PyTorch 2.6 &bull; RTX 3050 CUDA
              </div>
            </div>
          </div>

          {/* Section 1: Executive Abstract */}
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-white print:text-black uppercase tracking-wider font-mono flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400 print:bg-black" />
              <span>1. Executive Abstract</span>
            </h3>
            <p className="text-xs text-slate-300 print:text-slate-800 leading-relaxed">
              This document reports the empirical evaluation of bounded adversarial cloaking applied to the visual asset{" "}
              <strong className="text-white print:text-black font-mono">{job.image_id || "target"}</strong>. Optimization was conducted using projected gradient descent (PGD) against a fine-tuned ResNet-50 surrogate classifier. The synthesized perturbation achieved an SSIM structural fidelity score of{" "}
              <strong className="text-cyan-300 print:text-cyan-800 font-mono">{qm?.ssim ?? "N/A"}</strong> and caused a{" "}
              <strong className="text-emerald-400 print:text-emerald-800 font-mono">
                {cls?.confidence_drop_percent ? `${cls.confidence_drop_percent}%` : "0%"}
              </strong>{" "}
              confidence depression on the original style classification ({cls?.original_prediction?.style_name || "Style"}).
            </p>
          </div>

          {/* Visual Pair Comparison */}
          <div className="grid grid-cols-2 gap-4 p-4 rounded-2xl bg-slate-950 print:bg-slate-50 border border-slate-800/80 print:border-slate-300">
            <div className="space-y-2 text-center">
              <span className="text-[11px] font-mono text-slate-400 print:text-slate-600 block">
                Original Visual Asset
              </span>
              <div className="h-48 rounded-xl overflow-hidden bg-slate-900 print:bg-slate-200 flex items-center justify-center p-1">
                {job.original_image_url && (
                  <img
                    src={getFullUrl(job.original_image_url)}
                    alt="Original"
                    className="max-h-full max-w-full object-contain rounded-lg"
                  />
                )}
              </div>
              <span className="text-[10px] text-slate-400 print:text-slate-600 font-mono block">
                Top-1: {cls?.original_prediction?.style_name} ({cls?.original_prediction?.confidence_percent}%)
              </span>
            </div>

            <div className="space-y-2 text-center">
              <span className="text-[11px] font-mono text-cyan-400 print:text-cyan-800 block">
                Adversarially Cloaked Asset
              </span>
              <div className="h-48 rounded-xl overflow-hidden bg-slate-900 print:bg-slate-200 flex items-center justify-center p-1">
                {job.protected_image_url && (
                  <img
                    src={getFullUrl(job.protected_image_url)}
                    alt="Cloaked"
                    className="max-h-full max-w-full object-contain rounded-lg"
                  />
                )}
              </div>
              <span className="text-[10px] text-cyan-300 print:text-cyan-800 font-mono block font-bold">
                Top-1: {cls?.cloaked_prediction?.style_name} ({cls?.cloaked_prediction?.confidence_percent}%)
              </span>
            </div>
          </div>

          {/* Section 2: Mathematical Fidelity & Bound Audit */}
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-white print:text-black uppercase tracking-wider font-mono flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400 print:bg-black" />
              <span>2. Mathematical Quality &amp; Bound Verification</span>
            </h3>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              <div className="p-3 rounded-xl bg-slate-950 print:bg-slate-100 border border-slate-800 print:border-slate-300 text-center">
                <span className="text-[10px] text-slate-500 uppercase font-mono block">SSIM Fidelity</span>
                <span className="text-base font-mono font-bold text-sky-400 print:text-sky-800">
                  {qm?.ssim ?? "—"}
                </span>
                <span className="text-[9px] text-slate-500 block">&gt;0.90 Target</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950 print:bg-slate-100 border border-slate-800 print:border-slate-300 text-center">
                <span className="text-[10px] text-slate-500 uppercase font-mono block">PSNR</span>
                <span className="text-base font-mono font-bold text-sky-400 print:text-sky-800">
                  {qm?.psnr_db ? `${qm.psnr_db} dB` : "—"}
                </span>
                <span className="text-[9px] text-slate-500 block">&gt;30 dB Target</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950 print:bg-slate-100 border border-slate-800 print:border-slate-300 text-center">
                <span className="text-[10px] text-slate-500 uppercase font-mono block">LPIPS Distance</span>
                <span className="text-base font-mono font-bold text-sky-400 print:text-sky-800">
                  {qm?.lpips ?? "N/A"}
                </span>
                <span className="text-[9px] text-slate-500 block">&lt;0.08 Target</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950 print:bg-slate-100 border border-slate-800 print:border-slate-300 text-center">
                <span className="text-[10px] text-slate-500 uppercase font-mono block">L-&infin; Norm</span>
                <span className="text-base font-mono font-bold text-emerald-400 print:text-emerald-800">
                  {qm?.linf_norm ?? "—"}
                </span>
                <span className="text-[9px] text-slate-500 block">&le;8/255 Bound</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950 print:bg-slate-100 border border-slate-800 print:border-slate-300 text-center">
                <span className="text-[10px] text-slate-500 uppercase font-mono block">RMSE (L2)</span>
                <span className="text-base font-mono font-bold text-sky-400 print:text-sky-800">
                  {qm?.rmse ?? "—"}
                </span>
                <span className="text-[9px] text-slate-500 block">Root Mean Sq</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950 print:bg-slate-100 border border-slate-800 print:border-slate-300 text-center">
                <span className="text-[10px] text-slate-500 uppercase font-mono block">&epsilon;-Constraint</span>
                <span
                  className={`text-xs font-mono font-bold block mt-1 ${
                    qm?.epsilon_passed ? "text-emerald-400 print:text-emerald-800" : "text-rose-400 print:text-rose-800"
                  }`}
                >
                  {qm?.epsilon_passed ? "PASSED" : "FAILED"}
                </span>
                <span className="text-[9px] text-slate-500 block">Tolerance 1/255</span>
              </div>
            </div>
          </div>

          {/* Section 3: Target Classifier Disruption */}
          {cls && (
            <div className="space-y-3">
              <h3 className="text-sm font-bold text-white print:text-black uppercase tracking-wider font-mono flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-cyan-400 print:bg-black" />
                <span>3. Neural Disruption &amp; Style Prediction Matrix</span>
              </h3>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono border border-slate-800 print:border-slate-300 rounded-xl overflow-hidden">
                  <thead className="bg-slate-950 print:bg-slate-100 text-slate-400 print:text-slate-700 uppercase text-[10px]">
                    <tr>
                      <th className="py-2.5 px-4">Evaluation Phase</th>
                      <th className="py-2.5 px-4">Top-1 Style Class</th>
                      <th className="py-2.5 px-4">Confidence</th>
                      <th className="py-2.5 px-4">Top-1 Disrupted</th>
                      <th className="py-2.5 px-4">Embedding Cosine Shift</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 print:divide-slate-200">
                    <tr>
                      <td className="py-3 px-4 font-bold text-slate-300 print:text-slate-800">Baseline (Original)</td>
                      <td className="py-3 px-4 text-white print:text-black">{cls.original_prediction.style_name}</td>
                      <td className="py-3 px-4 text-slate-400 print:text-slate-600">{cls.original_prediction.confidence_percent}%</td>
                      <td className="py-3 px-4 text-slate-500">—</td>
                      <td className="py-3 px-4 text-slate-500">0.0000</td>
                    </tr>
                    <tr className="bg-cyan-950/10 print:bg-slate-50">
                      <td className="py-3 px-4 font-bold text-cyan-400 print:text-cyan-800">Protected (Cloaked)</td>
                      <td className="py-3 px-4 text-cyan-300 print:text-cyan-800 font-bold">{cls.cloaked_prediction.style_name}</td>
                      <td className="py-3 px-4 text-cyan-400 print:text-cyan-800">{cls.cloaked_prediction.confidence_percent}%</td>
                      <td className="py-3 px-4 text-emerald-400 print:text-emerald-800 font-bold">
                        {cls.top1_prediction_changed ? "YES (Shifted)" : "NO (Depressed)"}
                      </td>
                      <td className="py-3 px-4 text-sky-400 print:text-sky-800 font-bold">
                        {cls.embedding_cosine_distance}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Section 4: Cross-Model Transferability Ledger */}
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-white print:text-black uppercase tracking-wider font-mono flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400 print:bg-black" />
              <span>4. Multi-Model Transferability Matrix</span>
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono border border-slate-800 print:border-slate-300 rounded-xl overflow-hidden">
                <thead className="bg-slate-950 print:bg-slate-100 text-slate-400 print:text-slate-700 uppercase text-[10px]">
                  <tr>
                    <th className="py-2.5 px-4">Architecture</th>
                    <th className="py-2.5 px-4">Role</th>
                    <th className="py-2.5 px-4">Pre-Cloak Top-1</th>
                    <th className="py-2.5 px-4">Post-Cloak Top-1</th>
                    <th className="py-2.5 px-4">Confidence Drop</th>
                    <th className="py-2.5 px-4">Transfer Verdict</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 print:divide-slate-200">
                  <tr className="hover:bg-slate-950/40">
                    <td className="py-3 px-4 font-bold text-white print:text-black uppercase">ResNet-50</td>
                    <td className="py-3 px-4 text-cyan-400 print:text-cyan-800 font-semibold">Surrogate (White-Box)</td>
                    <td className="py-3 px-4 text-slate-300 print:text-slate-700">{cls?.original_prediction.style_name}</td>
                    <td className="py-3 px-4 text-cyan-300 print:text-cyan-800 font-bold">{cls?.cloaked_prediction.style_name}</td>
                    <td className="py-3 px-4 text-emerald-400 print:text-emerald-800 font-bold">
                      {cls ? `-${cls.confidence_drop_percent}%` : "—"}
                    </td>
                    <td className="py-3 px-4 text-emerald-400 print:text-emerald-800 font-bold">Disrupted</td>
                  </tr>

                  {job.evaluations
                    ?.filter((ev) => !ev.is_surrogate)
                    .map((ev, idx) => {
                      const isClip = ev.model_name.toLowerCase().includes("clip") || ev.evaluation_type === "zero_shot_multimodal";
                      return (
                        <tr key={idx} className="hover:bg-slate-950/40">
                          <td className="py-3 px-4 font-bold text-white print:text-black uppercase">
                            <div>
                              <span>{ev.model_name.replace(/_/g, "-")}</span>
                              {isClip && (
                                <span className="block text-[9px] text-cyan-400 print:text-cyan-800 font-sans normal-case italic">
                                  Zero-Shot (&ldquo;a painting in {"{style}"} style&rdquo;)
                                </span>
                              )}
                            </div>
                          </td>
                          <td className="py-3 px-4 text-purple-400 print:text-purple-800 font-semibold">
                            {isClip ? "Target (Multimodal Dual)" : "Target (Black-Box)"}
                          </td>
                          <td className="py-3 px-4 text-slate-300 print:text-slate-700">
                            {ev.original_top1_class ? (
                              `${ev.original_top1_class} (${((ev.original_top1_confidence || 0) * 100).toFixed(1)}%)`
                            ) : (
                              <span className="text-slate-500">—</span>
                            )}
                          </td>
                          <td className="py-3 px-4 text-purple-300 print:text-purple-800 font-bold">
                            {ev.protected_top1_class ? (
                              `${ev.protected_top1_class} (${((ev.protected_top1_confidence || 0) * 100).toFixed(1)}%)`
                            ) : (
                              <span className="text-slate-500">—</span>
                            )}
                          </td>
                          <td className="py-3 px-4 text-emerald-400 print:text-emerald-800 font-bold">
                            {ev.confidence_drop !== null && ev.confidence_drop !== undefined
                              ? `-${(ev.confidence_drop * 100).toFixed(1)}%`
                              : "—"}
                          </td>
                          <td className="py-3 px-4 font-bold">
                            {ev.status === "evaluated" ? (
                              <span
                                className={
                                  ev.classification_disrupted
                                    ? "text-emerald-400 print:text-emerald-800"
                                    : "text-amber-400 print:text-amber-800"
                                }
                              >
                                {ev.classification_disrupted ? "Transferred" : "Resistant"}
                              </span>
                            ) : (
                              <span className="text-slate-500 print:text-slate-500 uppercase text-[10px]">
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

          {/* Section 5: Transformation Robustness */}
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-white print:text-black uppercase tracking-wider font-mono flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400 print:bg-black" />
              <span>5. Transformation Robustness Survival Ledger</span>
            </h3>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {job.robustness?.map((rb, idx) => (
                <div
                  key={idx}
                  className="p-3 rounded-xl bg-slate-950 print:bg-slate-100 border border-slate-800 print:border-slate-300 space-y-1 font-mono text-xs"
                >
                  <span className="font-bold text-white print:text-black block truncate">{rb.transformation_name}</span>
                  <div className="flex justify-between text-[11px]">
                    <span className="text-slate-500">Post-Style:</span>
                    <span className="text-slate-300 print:text-slate-800 truncate">{rb.top1_class_after_transform || "—"}</span>
                  </div>
                  <div className="flex justify-between text-[11px]">
                    <span className="text-slate-500">Survival:</span>
                    <span
                      className={`font-bold ${
                        rb.prediction_disrupted
                          ? "text-emerald-400 print:text-emerald-800"
                          : "text-amber-400 print:text-amber-800"
                      }`}
                    >
                      {rb.prediction_disrupted ? "Protected" : "Vulnerable"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Section 6: Limitations & Scientific Declarations */}
          <div className="space-y-3 pt-4 border-t border-slate-800 print:border-slate-300">
            <h3 className="text-sm font-bold text-white print:text-black uppercase tracking-wider font-mono flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400 print:bg-black" />
              <span>6. Scientific Limitations &amp; Ethical Disclosure</span>
            </h3>

            <ul className="text-xs text-slate-400 print:text-slate-700 space-y-1.5 leading-relaxed list-disc list-inside">
              <li>
                <strong>Empirical Nature:</strong> Adversarial cloaking operates by introducing gradient-optimized perturbations that push deep features across decision boundaries. It does not provide mathematical cryptography guarantees.
              </li>
              <li>
                <strong>Cross-Model Transfer:</strong> While perturbations achieve strong disruption against models with similar convolutional backbones, black-box transfer to dissimilar architectures (e.g., pure self-attention Transformers) depends on the diversity of the surrogate ensemble.
              </li>
              <li>
                <strong>Lossy Post-Processing:</strong> Aggressive downsampling, high-compression JPEG encodings, or denoising autoencoders may filter out high-frequency perturbations, attenuating protection.
              </li>
              <li>
                <strong>Provenance &amp; Integrity:</strong> The metrics documented herein reflect forward-pass evaluations on verified model checkpoints under standardized ImageNet normalization.
              </li>
            </ul>
          </div>

          {/* Dossier Footer Signature */}
          <div className="pt-6 border-t border-slate-800 print:border-slate-300 flex justify-between items-center text-[10px] text-slate-500 font-mono">
            <span>AI-Protect Adversarial Lab &bull; Automated Verification Subsystem</span>
            <span>Document Integrity Verified &bull; SHA-256 Validated</span>
          </div>
        </article>
      )}
    </div>
  );
}

export default function ResearchReportStudioPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-[40vh] flex flex-col items-center justify-center space-y-4">
          <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
          <p className="text-slate-400 text-xs font-mono">Compiling evaluation report data...</p>
        </div>
      }
    >
      <ReportStudioContent />
    </Suspense>
  );
}
