"use client";

import React, { useState } from "react";
import {
  FileText,
  Cpu,
  Clock,
  HardDrive,
  KeyRound,
  CheckCircle2,
  AlertCircle,
  Play,
  ShieldAlert,
  ChevronDown,
  ChevronUp,
  Sliders,
  Layers,
  Sparkles,
  Loader2,
} from "lucide-react";
import { GuardianPlanResponse } from "@/lib/api";

interface ExperimentPlanProps {
  plan: GuardianPlanResponse;
  onAuthorize: () => void;
  onExecute: () => void;
  isExecuting?: boolean;
}

export default function ExperimentPlan({
  plan,
  onAuthorize,
  onExecute,
  isExecuting = false,
}: ExperimentPlanProps) {
  const [expandedSteps, setExpandedSteps] = useState<Record<number, boolean>>({});

  const toggleStep = (idx: number) => {
    setExpandedSteps((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  // Status badge styling
  const getStatusBadge = (status: string) => {
    switch (status) {
      case "awaiting_approval":
        return "bg-amber-500/10 text-amber-400 border-amber-500/30";
      case "approved":
        return "bg-emerald-500/10 text-emerald-400 border-emerald-500/30";
      case "running":
        return "bg-sky-500/10 text-sky-400 border-sky-500/30 animate-pulse";
      case "completed":
        return "bg-cyan-500/10 text-cyan-400 border-cyan-500/30";
      case "blocked":
        return "bg-rose-500/10 text-rose-400 border-rose-500/30";
      default:
        return "bg-slate-700/20 text-slate-300 border-slate-700";
    }
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl overflow-hidden backdrop-blur-md shadow-xl text-slate-100">
      {/* Plan Header */}
      <div className="p-6 border-b border-slate-800/80 bg-slate-900/40">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20">
              {plan.objective_type.replace(/_/g, " ")}
            </span>
            <span className="text-xs text-slate-400 font-mono">v{plan.version}</span>
          </div>

          <span
            className={`text-xs uppercase tracking-wider font-semibold px-2.5 py-1 rounded-full border ${getStatusBadge(
              plan.status
            )}`}
          >
            {plan.status.replace(/_/g, " ")}
          </span>
        </div>

        <h3 className="text-xl font-bold text-white tracking-tight">{plan.title}</h3>
        <p className="text-xs text-slate-300 mt-1.5 leading-relaxed">{plan.goal}</p>

        {/* Bound Config Hash */}
        <div className="mt-3.5 flex items-center gap-2 text-[11px] font-mono text-slate-400 bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
          <KeyRound className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          <span className="text-slate-500 select-none">Config Fingerprint:</span>
          <span className="text-slate-300 truncate">{plan.config_hash}</span>
        </div>
      </div>

      {/* Resource Estimates Grid */}
      <div className="grid grid-cols-3 divide-x divide-slate-800/80 border-b border-slate-800/80 bg-slate-950/40 text-xs">
        <div className="p-3.5 flex items-center gap-2.5">
          <Cpu className="w-4 h-4 text-sky-400 shrink-0" />
          <div>
            <div className="text-[10px] text-slate-400">Peak VRAM</div>
            <div className="font-semibold text-slate-100">
              {plan.resource_estimate.gpu_vram_gb.toFixed(1)} GB
            </div>
          </div>
        </div>

        <div className="p-3.5 flex items-center gap-2.5">
          <Clock className="w-4 h-4 text-emerald-400 shrink-0" />
          <div>
            <div className="text-[10px] text-slate-400">Estimated Runtime</div>
            <div className="font-semibold text-slate-100">
              ~{plan.resource_estimate.estimated_runtime_seconds.toFixed(0)}s
            </div>
          </div>
        </div>

        <div className="p-3.5 flex items-center gap-2.5">
          <HardDrive className="w-4 h-4 text-purple-400 shrink-0" />
          <div>
            <div className="text-[10px] text-slate-400">Target Device</div>
            <div className="font-semibold uppercase text-purple-300">
              {plan.resource_estimate.hardware_device}
            </div>
          </div>
        </div>
      </div>

      {/* Missing Prerequisites Warning */}
      {plan.missing_prerequisites && plan.missing_prerequisites.length > 0 && (
        <div className="m-4 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-rose-200">Missing Prerequisites:</span>
            <ul className="list-disc list-inside mt-1 space-y-0.5 text-rose-300/90">
              {plan.missing_prerequisites.map((req, i) => (
                <li key={i}>{req}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Steps List */}
      <div className="p-6 space-y-4">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
          <Layers className="w-3.5 h-3.5 text-sky-400" />
          <span>Execution Sequence ({plan.steps.length} Steps)</span>
        </h4>

        <div className="space-y-2">
          {plan.steps.map((step, idx) => {
            const isExpanded = !!expandedSteps[idx];
            return (
              <div
                key={idx}
                className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden text-xs transition-colors hover:border-slate-700"
              >
                <div
                  onClick={() => toggleStep(idx)}
                  className="p-3.5 flex items-center justify-between cursor-pointer select-none"
                >
                  <div className="flex items-center gap-3">
                    <span className="w-5 h-5 rounded-full bg-slate-800 text-sky-400 font-mono text-[10px] flex items-center justify-center font-bold">
                      {step.step_number}
                    </span>
                    <div>
                      <span className="font-mono text-sky-300 font-medium">
                        {step.tool_name}
                      </span>
                      <p className="text-[11px] text-slate-400 mt-0.5">{step.description}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    {step.requires_approval ? (
                      <span className="text-[10px] px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 font-medium">
                        GPU Compute
                      </span>
                    ) : (
                      <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700 font-medium">
                        Read-Only
                      </span>
                    )}
                    {isExpanded ? (
                      <ChevronUp className="w-4 h-4 text-slate-400" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-slate-400" />
                    )}
                  </div>
                </div>

                {isExpanded && (
                  <div className="px-3.5 pb-3.5 pt-1 border-t border-slate-800/60 bg-slate-950/60 font-mono text-[11px]">
                    <span className="text-[10px] uppercase text-slate-500 font-sans font-semibold mb-1 block">
                      Parameters:
                    </span>
                    <pre className="text-slate-300 overflow-x-auto p-2 rounded bg-slate-900 border border-slate-800">
                      {JSON.stringify(step.parameters, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Success Criteria */}
        {plan.success_criteria && plan.success_criteria.length > 0 && (
          <div className="pt-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2 mb-2.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              <span>Measurable Success Criteria</span>
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
              {plan.success_criteria.map((sc, i) => (
                <div
                  key={i}
                  className="p-2.5 rounded-lg bg-slate-950 border border-slate-800/80 flex items-start gap-2 text-slate-300"
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 shrink-0" />
                  <div>
                    <span className="font-semibold text-slate-200">{sc.description}</span>
                    <div className="font-mono text-[10px] text-emerald-400 mt-0.5">
                      {sc.condition}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Plan Action Footer */}
      <div className="p-4 px-6 border-t border-slate-800 bg-slate-900/60 flex items-center justify-between">
        <div className="text-xs text-slate-400">
          {plan.status === "awaiting_approval" && (
            <span className="flex items-center gap-1.5 text-amber-400">
              <ShieldAlert className="w-3.5 h-3.5" />
              <span>Awaiting authorization from researcher.</span>
            </span>
          )}
          {plan.status === "approved" && (
            <span className="flex items-center gap-1.5 text-emerald-400">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Authorized. Ready to run.</span>
            </span>
          )}
          {plan.status === "running" && (
            <span className="flex items-center gap-1.5 text-sky-400">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              <span>Executing steps in background...</span>
            </span>
          )}
          {plan.status === "completed" && (
            <span className="flex items-center gap-1.5 text-cyan-400">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Execution completed. See Evidence tab.</span>
            </span>
          )}
        </div>

        <div>
          {plan.status === "awaiting_approval" && (
            <button
              onClick={onAuthorize}
              disabled={isExecuting}
              className="inline-flex items-center gap-2 px-5 py-2 text-xs font-semibold rounded-lg bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 text-slate-950 shadow-md shadow-amber-500/20 transition-all font-sans"
            >
              <ShieldAlert className="w-3.5 h-3.5" />
              <span>Authorize Execution</span>
            </button>
          )}

          {plan.status === "approved" && (
            <button
              onClick={onExecute}
              disabled={isExecuting}
              className="inline-flex items-center gap-2 px-5 py-2 text-xs font-semibold rounded-lg bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 shadow-md shadow-emerald-500/20 transition-all font-sans"
            >
              {isExecuting ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Launching...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Execute Plan</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
