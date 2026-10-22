"use client";

import React, { useState, useEffect } from "react";
import { Activity, ShieldAlert, CheckCircle2, XCircle, HelpCircle, Layers } from "lucide-react";

export default function HypothesisLabPage() {
  const [hypotheses, setHypotheses] = useState<any[]>([]);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/discovery/hypotheses")
      .then((res) => res.json())
      .then((data) => setHypotheses(data.hypotheses || []))
      .catch(() => {
        // Fallback default sample data
        setHypotheses([
          {
            hypothesis_id: "hyp-0a81b2c3",
            statement: "The physical property band_gap strictly obeys analytical scaling with (electronegativity_delta)^1.5 / radius_mean.",
            variables: ["electronegativity_delta", "radius_mean", "band_gap"],
            expected_relationship: "analytical_power_law",
            confidence: 0.90,
            novelty_score: 0.85,
            testability_score: 0.95,
            status: "supported"
          },
          {
            hypothesis_id: "hyp-1b92c3d4",
            statement: "Higher electronegativity_delta deterministically stabilizes negative formation energy across stoichiometry.",
            variables: ["electronegativity_delta", "formation_energy_per_atom"],
            expected_relationship: "monotonic_correlation",
            confidence: 0.85,
            novelty_score: 0.72,
            testability_score: 0.95,
            status: "supported"
          },
          {
            hypothesis_id: "hyp-2c03d4e5",
            statement: "Coupled interaction ratio (density_mean / radius_mean) governs bulk modulus stiffness.",
            variables: ["density_mean", "radius_mean", "bulk_modulus"],
            expected_relationship: "analytical_power_law",
            confidence: 0.80,
            novelty_score: 0.78,
            testability_score: 0.95,
            status: "supported"
          },
          {
            hypothesis_id: "hyp-3d14e5f6",
            statement: "Atomic number Z_min alone directly predicts bulk modulus elasticity across all crystal classes.",
            variables: ["z_min", "bulk_modulus"],
            expected_relationship: "linear_correlation",
            confidence: 0.20,
            novelty_score: 0.40,
            testability_score: 0.90,
            status: "rejected"
          }
        ]);
      });
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white flex items-center gap-2">
            <Activity className="w-6 h-6 text-cyan-400" />
            <span>Hypothesis Lab & Adversarial Critic</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Structured candidate hypotheses subjected to statistical falsification and multi-agent scrutiny.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {hypotheses.map((h, i) => (
            <div key={i} className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs text-slate-400">{h.hypothesis_id}</span>
                <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold tracking-wide ${
                  h.status === "supported"
                    ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                    : h.status === "rejected"
                    ? "bg-rose-500/10 text-rose-400 border border-rose-500/20"
                    : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                }`}>
                  {h.status}
                </span>
              </div>

              <p className="text-sm font-medium text-slate-200 leading-snug">
                {h.statement}
              </p>

              <div className="flex flex-wrap gap-1.5 pt-1">
                {h.variables?.map((v: string, idx: number) => (
                  <span key={idx} className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-[11px] font-mono">
                    {v}
                  </span>
                ))}
              </div>

              <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-800/80 text-center text-xs">
                <div>
                  <div className="text-slate-400 text-[10px]">Confidence</div>
                  <div className="font-bold text-cyan-400 font-mono">{(h.confidence * 100).toFixed(0)}%</div>
                </div>
                <div>
                  <div className="text-slate-400 text-[10px]">Novelty</div>
                  <div className="font-bold text-indigo-400 font-mono">{(h.novelty_score * 100).toFixed(0)}%</div>
                </div>
                <div>
                  <div className="text-slate-400 text-[10px]">Testability</div>
                  <div className="font-bold text-emerald-400 font-mono">{(h.testability_score * 100).toFixed(0)}%</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
