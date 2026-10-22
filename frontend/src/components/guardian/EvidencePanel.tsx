"use client";

import React from "react";
import {
  FileSearch,
  CheckCircle2,
  Lightbulb,
  AlertTriangle,
  ArrowRight,
  ShieldCheck,
  Scale,
  Hash,
  Sparkles,
} from "lucide-react";
import { GuardianEvidenceSummary } from "@/lib/api";

interface EvidencePanelProps {
  evidence: GuardianEvidenceSummary | null;
  onSelectRecommendation?: (rec: string) => void;
}

export default function EvidencePanel({
  evidence,
  onSelectRecommendation,
}: EvidencePanelProps) {
  if (!evidence) {
    return (
      <div className="p-8 text-center bg-slate-900/60 border border-slate-800 rounded-2xl text-slate-400">
        <FileSearch className="w-8 h-8 text-slate-600 mx-auto mb-2 animate-pulse" />
        <p className="text-sm font-medium text-slate-300">No Evidence Registered</p>
        <p className="text-xs text-slate-500 mt-1">
          Execute an experiment plan to synthesize grounded empirical observations,
          hypothesis evaluations, and next-step recommendations.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl overflow-hidden backdrop-blur-md shadow-xl text-slate-100 space-y-6 p-6">
      {/* Header */}
      <div className="border-b border-slate-800 pb-4">
        <div className="flex items-center gap-2 text-cyan-400 text-xs font-semibold uppercase tracking-wider mb-1">
          <Sparkles className="w-4 h-4" />
          <span>Grounded Research Intelligence</span>
        </div>
        <h3 className="text-xl font-bold text-white tracking-tight">
          Empirical Evidence & Analysis
        </h3>
        <p className="text-xs text-slate-400 mt-1">
          Conclusions derived strictly from real tensor-level measurements and verified model inference.
        </p>
      </div>

      {/* Cited Experiments */}
      {evidence.cited_experiment_ids && evidence.cited_experiment_ids.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Hash className="w-3.5 h-3.5 text-slate-500" />
            <span>Cited Experiment Identifiers</span>
          </h4>
          <div className="flex flex-wrap gap-2">
            {evidence.cited_experiment_ids.map((id, idx) => (
              <span
                key={idx}
                className="font-mono text-[11px] px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-800 text-slate-300 select-all"
              >
                {id}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 1. Empirical Observations */}
      <div className="space-y-2.5">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          <span>Empirical Observations</span>
        </h4>
        <div className="space-y-2 text-xs">
          {evidence.empirical_observations?.map((obs, i) => (
            <div
              key={i}
              className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 flex items-start gap-2.5 text-slate-200"
            >
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 shrink-0" />
              <span className="leading-relaxed">{obs}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 2. Tested Hypotheses */}
      {evidence.hypotheses_tested && evidence.hypotheses_tested.length > 0 && (
        <div className="space-y-2.5">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <Scale className="w-3.5 h-3.5 text-purple-400" />
            <span>Hypotheses Evaluated</span>
          </h4>
          <div className="space-y-2 text-xs">
            {evidence.hypotheses_tested.map((hyp, i) => (
              <div
                key={i}
                className="p-3 rounded-xl bg-purple-500/5 border border-purple-500/20 text-purple-200/90 leading-relaxed text-xs"
              >
                {hyp}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 3. Actionable Next-Step Recommendations */}
      {evidence.grounded_recommendations && evidence.grounded_recommendations.length > 0 && (
        <div className="space-y-2.5">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <Lightbulb className="w-3.5 h-3.5 text-amber-400" />
            <span>Actionable Next-Step Recommendations</span>
          </h4>
          <div className="space-y-2.5">
            {evidence.grounded_recommendations.map((rec, i) => (
              <div
                key={i}
                className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs flex items-center justify-between gap-3 text-amber-200"
              >
                <div className="leading-relaxed">{rec}</div>
                {onSelectRecommendation && (
                  <button
                    onClick={() => onSelectRecommendation(rec)}
                    className="shrink-0 px-3 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold text-[11px] inline-flex items-center gap-1 transition"
                  >
                    <span>Plan Experiment</span>
                    <ArrowRight className="w-3 h-3" />
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 4. Methodological Uncertainties & Caveats */}
      {evidence.uncertainties_and_caveats && evidence.uncertainties_and_caveats.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <AlertTriangle className="w-3.5 h-3.5 text-slate-500" />
            <span>Uncertainties & Methodological Bounds</span>
          </h4>
          <div className="space-y-1.5 text-[11px] text-slate-400">
            {evidence.uncertainties_and_caveats.map((unc, i) => (
              <div key={i} className="flex items-start gap-2">
                <span className="text-slate-600 mt-0.5">•</span>
                <span className="leading-relaxed">{unc}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 5. Scientific Disclaimer */}
      <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-[11px] text-slate-400 flex items-start gap-2.5">
        <ShieldCheck className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <strong className="text-slate-300">Empirical Integrity Notice:</strong> Adversarial perturbations
          induce measured feature misalignment in evaluated vision models. They are non-cryptographic
          and do not provide absolute mathematical immunity against future generative models or scraping pipelines.
        </p>
      </div>
    </div>
  );
}
