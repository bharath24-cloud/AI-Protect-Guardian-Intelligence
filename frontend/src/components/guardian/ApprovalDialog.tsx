"use client";

import React, { useState } from "react";
import {
  ShieldAlert,
  Cpu,
  Clock,
  HardDrive,
  KeyRound,
  CheckCircle,
  X,
  AlertTriangle,
  Loader2,
} from "lucide-react";
import { GuardianPlanResponse } from "@/lib/api";

interface ApprovalDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (notes?: string) => Promise<void>;
  plan: GuardianPlanResponse;
  isSubmitting: boolean;
}

export default function ApprovalDialog({
  isOpen,
  onClose,
  onConfirm,
  plan,
  isSubmitting,
}: ApprovalDialogProps) {
  const [notes, setNotes] = useState("");

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onConfirm(notes);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-xl bg-slate-900 border border-amber-500/30 rounded-2xl shadow-2xl shadow-amber-500/10 overflow-hidden text-slate-100">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2.5 text-amber-400">
            <ShieldAlert className="w-5 h-5 text-amber-400" />
            <h3 className="font-semibold text-lg text-white">Authorize Experiment Execution</h3>
          </div>
          <button
            onClick={onClose}
            disabled={isSubmitting}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-5">
          {/* Plan Info */}
          <div>
            <h4 className="text-sm font-semibold text-white">{plan.title}</h4>
            <p className="text-xs text-slate-400 mt-1 line-clamp-2">{plan.goal}</p>
          </div>

          {/* Warning Banner */}
          <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/20 flex gap-3 text-xs text-amber-200">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-amber-300">GPU Compute Authorization Required</p>
              <p className="mt-0.5 text-amber-200/90 leading-relaxed">
                This plan executes neural network inference and adversarial gradient optimization.
                Your authorization signs the exact configuration fingerprint below. If any step parameters change,
                the authorization will be invalidated.
              </p>
            </div>
          </div>

          {/* Resource Estimates */}
          <div className="grid grid-cols-3 gap-2.5 text-xs">
            <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60">
              <div className="flex items-center gap-1.5 text-slate-400 mb-1">
                <Cpu className="w-3.5 h-3.5 text-sky-400" />
                <span>Peak VRAM</span>
              </div>
              <span className="font-mono text-sm font-semibold text-white">
                {plan.resource_estimate.gpu_vram_gb.toFixed(1)} GB
              </span>
            </div>
            <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60">
              <div className="flex items-center gap-1.5 text-slate-400 mb-1">
                <Clock className="w-3.5 h-3.5 text-emerald-400" />
                <span>Runtime</span>
              </div>
              <span className="font-mono text-sm font-semibold text-white">
                ~{plan.resource_estimate.estimated_runtime_seconds.toFixed(0)}s
              </span>
            </div>
            <div className="p-3 rounded-lg bg-slate-800/60 border border-slate-700/60">
              <div className="flex items-center gap-1.5 text-slate-400 mb-1">
                <HardDrive className="w-3.5 h-3.5 text-purple-400" />
                <span>Device</span>
              </div>
              <span className="font-mono text-xs font-semibold uppercase text-purple-300">
                {plan.resource_estimate.hardware_device}
              </span>
            </div>
          </div>

          {/* Bound Config Hash */}
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-between text-xs font-mono">
            <div className="flex items-center gap-2 text-slate-400 truncate">
              <KeyRound className="w-4 h-4 text-amber-400 shrink-0" />
              <span className="truncate">Bound Hash: {plan.config_hash}</span>
            </div>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shrink-0 ml-2">
              LOCKED
            </span>
          </div>

          {/* Authorization Notes */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              Authorization Notes (Optional)
            </label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="e.g., Authorized for held-out artwork evaluation"
              className="w-full h-18 px-3 py-2 text-xs bg-slate-950 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-amber-500 transition"
              disabled={isSubmitting}
            />
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              className="px-4 py-2 text-xs font-medium rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="inline-flex items-center gap-2 px-5 py-2 text-xs font-semibold rounded-lg bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 text-slate-950 shadow-md shadow-amber-500/20 transition-all font-sans"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Authorizing...</span>
                </>
              ) : (
                <>
                  <CheckCircle className="w-3.5 h-3.5" />
                  <span>Authorize & Run Experiment</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
