"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  History,
  Search,
  Filter,
  ArrowUpDown,
  Eye,
  Trash2,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  Clock,
  Sparkles,
  Shield,
  Layers,
  ArrowRight,
  X,
  Scale,
  RefreshCw,
  ExternalLink,
} from "lucide-react";
import {
  listJobs,
  deleteJob,
  JobSummaryItem,
  BACKEND_ROOT_URL,
} from "@/lib/api";

export default function HistoryPage() {
  const [jobs, setJobs] = useState<JobSummaryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [modelFilter, setModelFilter] = useState<string>("all");
  const [sortBy, setSortBy] = useState<"date_desc" | "date_asc" | "score_desc" | "ssim_desc">("date_desc");
  const [selectedForCompare, setSelectedForCompare] = useState<string[]>([]);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [compareModalOpen, setCompareModalOpen] = useState(false);

  const fetchJobs = async () => {
    try {
      setLoading(true);
      const data = await listJobs();
      setJobs(data);
    } catch (err) {
      console.error("Failed to load jobs list:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
  }, []);

  const handleDelete = async (jobId: string) => {
    try {
      setDeletingId(jobId);
      await deleteJob(jobId);
      setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
      setSelectedForCompare((prev) => prev.filter((id) => id !== jobId));
      setConfirmDeleteId(null);
    } catch (err) {
      console.error("Failed to delete job:", err);
    } finally {
      setDeletingId(null);
    }
  };

  const toggleCompare = (jobId: string) => {
    if (selectedForCompare.includes(jobId)) {
      setSelectedForCompare(selectedForCompare.filter((id) => id !== jobId));
    } else {
      if (selectedForCompare.length >= 2) {
        // replace second item
        setSelectedForCompare([selectedForCompare[0], jobId]);
      } else {
        setSelectedForCompare([...selectedForCompare, jobId]);
      }
    }
  };

  const getFullUrl = (path?: string | null) => {
    if (!path) return "";
    return path.startsWith("http") ? path : `${BACKEND_ROOT_URL}${path}`;
  };

  // Filter and sort jobs
  const filteredJobs = jobs
    .filter((j) => {
      const matchesSearch =
        j.original_filename?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        j.job_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (j.original_style && j.original_style.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (j.cloaked_style && j.cloaked_style.toLowerCase().includes(searchTerm.toLowerCase()));

      const matchesStatus = statusFilter === "all" || j.status === statusFilter;
      const matchesModel = modelFilter === "all" || j.surrogate_model === modelFilter;

      return matchesSearch && matchesStatus && matchesModel;
    })
    .sort((a, b) => {
      if (sortBy === "date_desc") {
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      }
      if (sortBy === "date_asc") {
        return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
      }
      if (sortBy === "score_desc") {
        return (b.overall_protection_score ?? -1) - (a.overall_protection_score ?? -1);
      }
      if (sortBy === "ssim_desc") {
        return (b.ssim ?? -1) - (a.ssim ?? -1);
      }
      return 0;
    });

  const jobA = jobs.find((j) => j.job_id === selectedForCompare[0]);
  const jobB = jobs.find((j) => j.job_id === selectedForCompare[1]);

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-16">
      {/* Workspace Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-[#262C38] pb-6">
        <div>
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-[#5CE1D2]/10 border border-[#5CE1D2]/20 text-[#5CE1D2] text-xs font-semibold mb-2">
            <History className="w-3.5 h-3.5" />
            <span>Telemetry &amp; Audit Trail</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-[#F4F1E9] tracking-tight">
            Experiment Memory &amp; Run Comparison
          </h1>
          <p className="text-xs sm:text-sm text-[#9BA5B4] mt-1 max-w-2xl">
            Audit, reproduce, and compare historical adversarial cloaking runs. Re-inspect any artwork in the Microscope or launch comparative multi-run analyses.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={fetchJobs}
            disabled={loading}
            className="inline-flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs text-slate-300 font-medium transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh Runs</span>
          </button>
          <Link
            href="/protect"
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-sky-400 hover:from-cyan-400 hover:to-sky-300 text-slate-950 font-bold text-xs shadow-md shadow-cyan-500/20 transition-all cursor-pointer"
          >
            <Shield className="w-4 h-4" />
            <span>New Cloaking Run</span>
          </Link>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-[#0b0f19] border border-slate-800 rounded-2xl p-4 flex flex-col md:flex-row gap-4 items-center justify-between">
        {/* Search Input */}
        <div className="relative w-full md:w-80">
          <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search by filename, style, or ID..."
            className="w-full pl-9 pr-4 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 font-mono"
          />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm("")}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Filter Controls */}
        <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
          {/* Status Filter */}
          <div className="flex items-center space-x-1.5 text-xs text-slate-400">
            <Filter className="w-3.5 h-3.5 text-slate-500" />
            <span>Status:</span>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
            >
              <option value="all">All ({jobs.length})</option>
              <option value="completed">Completed</option>
              <option value="processing">Processing</option>
              <option value="pending">Pending</option>
              <option value="failed">Failed</option>
            </select>
          </div>

          {/* Model Filter */}
          <div className="flex items-center space-x-1.5 text-xs text-slate-400">
            <span>Surrogate:</span>
            <select
              value={modelFilter}
              onChange={(e) => setModelFilter(e.target.value)}
              className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
            >
              <option value="all">All Models</option>
              <option value="resnet50">ResNet-50</option>
              <option value="convnext_tiny">ConvNeXt-Tiny</option>
              <option value="vit_base_patch16_224">ViT-B/16</option>
            </select>
          </div>

          {/* Sort By */}
          <div className="flex items-center space-x-1.5 text-xs text-slate-400">
            <ArrowUpDown className="w-3.5 h-3.5 text-slate-500" />
            <span>Sort:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
            >
              <option value="date_desc">Newest First</option>
              <option value="date_asc">Oldest First</option>
              <option value="score_desc">Highest Protection Score</option>
              <option value="ssim_desc">Highest SSIM Fidelity</option>
            </select>
          </div>
        </div>
      </div>

      {/* Compare Floating Banner */}
      {selectedForCompare.length > 0 && (
        <div className="sticky top-20 z-20 p-4 rounded-2xl bg-cyan-950/90 border border-cyan-500/40 shadow-2xl backdrop-blur-md flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
              <Scale className="w-4 h-4" />
            </div>
            <div>
              <span className="text-xs font-bold text-white block">
                {selectedForCompare.length === 1
                  ? "1 Run Selected (Select 1 more to compare)"
                  : "2 Runs Selected for Side-by-Side Comparison"}
              </span>
              <span className="text-[11px] text-cyan-200/80 font-mono">
                {selectedForCompare.map((id) => id.slice(0, 8)).join(" vs ")}
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setSelectedForCompare([])}
              className="px-3 py-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 text-xs text-slate-300 font-medium transition-colors"
            >
              Clear
            </button>
            <button
              onClick={() => setCompareModalOpen(true)}
              disabled={selectedForCompare.length < 2}
              className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all shadow-md ${
                selectedForCompare.length === 2
                  ? "bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-cyan-500/20 cursor-pointer"
                  : "bg-slate-800 text-slate-500 cursor-not-allowed"
              }`}
            >
              Compare Runs Side-by-Side
            </button>
          </div>
        </div>
      )}

      {/* Runs Grid / List */}
      {loading ? (
        <div className="min-h-[40vh] flex flex-col items-center justify-center space-y-4">
          <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
          <p className="text-slate-400 text-xs font-mono">Loading experiment records from database...</p>
        </div>
      ) : filteredJobs.length === 0 ? (
        <div className="p-12 text-center rounded-2xl bg-[#0b0f19] border border-slate-800 space-y-4">
          <History className="w-10 h-10 text-slate-600 mx-auto" />
          <h3 className="text-base font-bold text-white">No Experiments Found</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            {searchTerm || statusFilter !== "all" || modelFilter !== "all"
              ? "No runs match your current query and filters. Try clearing your search parameters."
              : "No adversarial cloaking jobs have been executed yet. Create your first experiment in the Studio."}
          </p>
          <Link
            href="/protect"
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-cyan-500 text-slate-950 text-xs font-bold hover:bg-cyan-400 transition-colors"
          >
            <span>Launch Experiment</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {filteredJobs.map((item) => {
            const isSelected = selectedForCompare.includes(item.job_id);
            const isDeleting = deletingId === item.job_id;
            const isCompleted = item.status === "completed";

            return (
              <div
                key={item.job_id}
                className={`relative rounded-2xl bg-[#0b0f19] border transition-all flex flex-col justify-between overflow-hidden group ${
                  isSelected
                    ? "border-cyan-500 shadow-lg shadow-cyan-500/10 ring-1 ring-cyan-500/40"
                    : "border-slate-800 hover:border-slate-700"
                }`}
              >
                {/* Card Top: Thumbnails & Status Badge */}
                <div className="p-4 space-y-3">
                  <div className="flex items-center justify-between text-xs">
                    <span
                      className={`px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider ${
                        item.status === "completed"
                          ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                          : item.status === "processing"
                          ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20"
                          : item.status === "failed"
                          ? "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                          : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                      }`}
                    >
                      {item.status}
                    </span>

                    <span className="text-[11px] text-slate-500 font-mono">
                      {new Date(item.created_at).toLocaleDateString()} &bull;{" "}
                      {new Date(item.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </span>
                  </div>

                  {/* Visual Thumbnails Side-by-Side */}
                  <div className="grid grid-cols-2 gap-2 h-36 rounded-xl overflow-hidden bg-slate-950 p-1 border border-slate-800/80">
                    {/* Original Thumbnail */}
                    <div className="relative rounded-lg overflow-hidden bg-slate-900 flex items-center justify-center">
                      {item.original_image_url ? (
                        <img
                          src={getFullUrl(item.original_image_url)}
                          alt="Original asset"
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <span className="text-[10px] text-slate-600 font-mono">No Image</span>
                      )}
                      <span className="absolute bottom-1 left-1 px-1.5 py-0.5 rounded bg-black/70 text-[9px] text-slate-300 font-mono">
                        Original
                      </span>
                    </div>

                    {/* Cloaked Thumbnail */}
                    <div className="relative rounded-lg overflow-hidden bg-slate-900 flex items-center justify-center">
                      {item.protected_image_url ? (
                        <img
                          src={getFullUrl(item.protected_image_url)}
                          alt="Cloaked asset"
                          className="w-full h-full object-cover"
                        />
                      ) : item.status === "processing" ? (
                        <div className="flex flex-col items-center space-y-1 text-cyan-400">
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span className="text-[9px] font-mono">Synthesizing</span>
                        </div>
                      ) : (
                        <span className="text-[10px] text-slate-600 font-mono">Pending</span>
                      )}
                      <span className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded bg-cyan-950/80 text-[9px] text-cyan-300 font-mono border border-cyan-500/30">
                        Cloaked
                      </span>
                    </div>
                  </div>

                  {/* Asset Filename & Surrogate Info */}
                  <div className="space-y-1">
                    <h3 className="text-xs font-bold text-white truncate" title={item.original_filename}>
                      {item.original_filename || "Untitled Artwork"}
                    </h3>
                    <div className="flex items-center space-x-2 text-[10px] text-slate-400 font-mono">
                      <span>Model:</span>
                      <span className="text-slate-200 uppercase">{item.surrogate_model}</span>
                      <span className="text-slate-600">&bull;</span>
                      <span className="truncate text-slate-500">ID: {item.job_id.slice(0, 8)}</span>
                    </div>
                  </div>

                  {/* Style Shift Badge */}
                  {item.original_style && item.cloaked_style && (
                    <div className="p-2 rounded-xl bg-slate-950 border border-slate-800 text-[11px] flex items-center justify-between">
                      <div className="flex items-center space-x-1.5 truncate">
                        <span className="text-slate-400">{item.original_style}</span>
                        <ArrowRight className="w-3 h-3 text-cyan-400 shrink-0" />
                        <span className="text-cyan-300 font-bold truncate">{item.cloaked_style}</span>
                      </div>
                      {item.top1_prediction_changed && (
                        <span className="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[9px] font-mono shrink-0">
                          Disrupted
                        </span>
                      )}
                    </div>
                  )}

                  {/* Key Metrics Grid */}
                  <div className="grid grid-cols-3 gap-2 pt-1 border-t border-slate-900 text-center font-mono">
                    <div className="p-1.5 rounded-lg bg-slate-950/60">
                      <span className="text-[9px] text-slate-500 block uppercase">Protection</span>
                      <span className="text-xs font-bold text-emerald-400">
                        {item.overall_protection_score !== null && item.overall_protection_score !== undefined
                          ? `${item.overall_protection_score}%`
                          : "—"}
                      </span>
                    </div>
                    <div className="p-1.5 rounded-lg bg-slate-950/60">
                      <span className="text-[9px] text-slate-500 block uppercase">SSIM</span>
                      <span className="text-xs font-bold text-indigo-400">
                        {item.ssim !== null && item.ssim !== undefined ? item.ssim : "—"}
                      </span>
                    </div>
                    <div className="p-1.5 rounded-lg bg-slate-950/60">
                      <span className="text-[9px] text-slate-500 block uppercase">Disruption</span>
                      <span className="text-xs font-bold text-sky-400">
                        {item.ai_disruption_score !== null && item.ai_disruption_score !== undefined
                          ? `${item.ai_disruption_score}%`
                          : "—"}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Card Footer Actions */}
                <div className="p-3 border-t border-slate-800/80 bg-slate-950/40 flex items-center justify-between">
                  {/* Select for compare toggle */}
                  <label className="flex items-center space-x-1.5 text-[11px] text-slate-400 hover:text-slate-200 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => toggleCompare(item.job_id)}
                      className="rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0 focus:ring-offset-0"
                    />
                    <span>Compare</span>
                  </label>

                  <div className="flex items-center space-x-2">
                    {/* Delete action */}
                    {confirmDeleteId === item.job_id ? (
                      <div className="flex items-center space-x-1">
                        <button
                          onClick={() => handleDelete(item.job_id)}
                          disabled={isDeleting}
                          className="px-2 py-1 rounded bg-rose-500/20 text-rose-400 hover:bg-rose-500/30 text-[10px] font-bold"
                        >
                          {isDeleting ? "..." : "Confirm"}
                        </button>
                        <button
                          onClick={() => setConfirmDeleteId(null)}
                          className="px-1.5 py-1 text-slate-400 hover:text-white text-[10px]"
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => setConfirmDeleteId(item.job_id)}
                        className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                        title="Delete Experiment Run"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}

                    {/* Inspect in Microscope Link */}
                    <Link
                      href={`/results/${item.job_id}`}
                      className="inline-flex items-center space-x-1 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-cyan-500/20 hover:text-cyan-400 text-slate-200 text-xs font-semibold border border-slate-700 hover:border-cyan-500/40 transition-all cursor-pointer"
                    >
                      <Eye className="w-3.5 h-3.5" />
                      <span>Microscope</span>
                    </Link>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Side-by-Side Run Comparison Modal */}
      {compareModalOpen && jobA && jobB && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in">
          <div className="bg-[#0b0f19] border border-slate-800 rounded-3xl max-w-5xl w-full max-h-[90vh] overflow-y-auto shadow-2xl p-6 sm:p-8 space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-xl bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
                  <Scale className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-lg font-bold text-white">Adversarial Run Comparison</h2>
                  <p className="text-xs text-slate-400">
                    Side-by-side empirical performance metrics across two separate cloaking runs.
                  </p>
                </div>
              </div>
              <button
                onClick={() => setCompareModalOpen(false)}
                className="p-2 rounded-xl bg-slate-900 text-slate-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Comparison Cards */}
            <div className="grid grid-cols-2 gap-6">
              {/* Run A */}
              <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-900 pb-3">
                  <div>
                    <span className="text-[10px] font-mono text-cyan-400 block font-bold">RUN A</span>
                    <span className="text-xs font-bold text-white truncate block max-w-[200px]">
                      {jobA.original_filename}
                    </span>
                  </div>
                  <Link
                    href={`/results/${jobA.job_id}`}
                    target="_blank"
                    className="inline-flex items-center space-x-1 text-[11px] text-cyan-400 hover:underline"
                  >
                    <span>Inspect</span>
                    <ExternalLink className="w-3 h-3" />
                  </Link>
                </div>

                <div className="grid grid-cols-2 gap-2 h-36 rounded-xl overflow-hidden bg-slate-900 p-1 border border-slate-800">
                  <img
                    src={getFullUrl(jobA.original_image_url)}
                    alt="Run A Orig"
                    className="w-full h-full object-cover rounded-lg"
                  />
                  <img
                    src={getFullUrl(jobA.protected_image_url)}
                    alt="Run A Cloaked"
                    className="w-full h-full object-cover rounded-lg"
                  />
                </div>

                <div className="space-y-2 text-xs font-mono">
                  <div className="flex justify-between py-1 border-b border-slate-900">
                    <span className="text-slate-500">Surrogate Model:</span>
                    <span className="text-white uppercase">{jobA.surrogate_model}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-900">
                    <span className="text-slate-500">Overall Protection:</span>
                    <span className="text-emerald-400 font-bold">{jobA.overall_protection_score ?? "—"}%</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-900">
                    <span className="text-slate-500">AI Disruption Rate:</span>
                    <span className="text-sky-400 font-bold">{jobA.ai_disruption_score ?? "—"}%</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-900">
                    <span className="text-slate-500">SSIM Fidelity:</span>
                    <span className="text-indigo-400 font-bold">{jobA.ssim ?? "—"}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-900">
                    <span className="text-slate-500">Style Shift:</span>
                    <span className="text-cyan-300">
                      {jobA.original_style || "?"} &rarr; {jobA.cloaked_style || "?"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Run B */}
              <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-900 pb-3">
                  <div>
                    <span className="text-[10px] font-mono text-purple-400 block font-bold">RUN B</span>
                    <span className="text-xs font-bold text-white truncate block max-w-[200px]">
                      {jobB.original_filename}
                    </span>
                  </div>
                  <Link
                    href={`/results/${jobB.job_id}`}
                    target="_blank"
                    className="inline-flex items-center space-x-1 text-[11px] text-purple-400 hover:underline"
                  >
                    <span>Inspect</span>
                    <ExternalLink className="w-3 h-3" />
                  </Link>
                </div>

                <div className="grid grid-cols-2 gap-2 h-36 rounded-xl overflow-hidden bg-slate-900 p-1 border border-slate-800">
                  <img
                    src={getFullUrl(jobB.original_image_url)}
                    alt="Run B Orig"
                    className="w-full h-full object-cover rounded-lg"
                  />
                  <img
                    src={getFullUrl(jobB.protected_image_url)}
                    alt="Run B Cloaked"
                    className="w-full h-full object-cover rounded-lg"
                  />
                </div>

                <div className="space-y-2 text-xs font-mono">
                  <div className="flex justify-between py-1 border-b border-slate-900">
                    <span className="text-slate-500">Surrogate Model:</span>
                    <span className="text-white uppercase">{jobB.surrogate_model}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-900">
                    <span className="text-slate-500">Overall Protection:</span>
                    <span className="text-emerald-400 font-bold">{jobB.overall_protection_score ?? "—"}%</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-900">
                    <span className="text-slate-500">AI Disruption Rate:</span>
                    <span className="text-sky-400 font-bold">{jobB.ai_disruption_score ?? "—"}%</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-900">
                    <span className="text-slate-500">SSIM Fidelity:</span>
                    <span className="text-indigo-400 font-bold">{jobB.ssim ?? "—"}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-900">
                    <span className="text-slate-500">Style Shift:</span>
                    <span className="text-cyan-300">
                      {jobB.original_style || "?"} &rarr; {jobB.cloaked_style || "?"}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex justify-end pt-4 border-t border-slate-800">
              <button
                onClick={() => setCompareModalOpen(false)}
                className="px-5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-white"
              >
                Close Comparison
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
