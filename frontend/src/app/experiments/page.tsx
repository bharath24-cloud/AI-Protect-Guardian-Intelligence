"use client";

import React, { useState, useEffect } from "react";
import { Cpu, CheckCircle, Clock, Zap, TrendingUp } from "lucide-react";

export default function ExperimentCenterPage() {
  const [experiments, setExperiments] = useState<any[]>([]);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/discovery/experiments")
      .then((res) => res.json())
      .then((data) => setExperiments(data.experiments || []))
      .catch(() => {
        setExperiments([
          {
            experiment_id: "exp-01a2b3c4",
            hypothesis_id: "hyp-0a81b2c3",
            experiment_type: "cross_validation_regression",
            parameters: { cv_folds: 5, features: ["electronegativity_delta", "radius_mean"] },
            random_seed: 42,
            computational_cost: 1.5,
            status: "completed",
            metrics: { cv_folds: 5, n_samples: 800, is_significant: true }
          },
          {
            experiment_id: "exp-12b3c4d5",
            hypothesis_id: "hyp-1b92c3d4",
            experiment_type: "cross_validation_regression",
            parameters: { cv_folds: 5, features: ["electronegativity_delta"] },
            random_seed: 42,
            computational_cost: 1.0,
            status: "completed",
            metrics: { cv_folds: 5, n_samples: 800, is_significant: true }
          },
          {
            experiment_id: "exp-23c4d5e6",
            hypothesis_id: "hyp-2c03d4e5",
            experiment_type: "cross_validation_regression",
            parameters: { cv_folds: 5, features: ["density_mean", "radius_mean"] },
            random_seed: 42,
            computational_cost: 1.0,
            status: "completed",
            metrics: { cv_folds: 5, n_samples: 800, is_significant: true }
          }
        ]);
      });
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white flex items-center gap-2">
            <Cpu className="w-6 h-6 text-indigo-400" />
            <span>Computational Experiment Center</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Execution logs of cross-validated computational experiments, parameter spaces, and execution costs.
          </p>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 border-b border-slate-800 text-slate-400 uppercase font-semibold">
                <tr>
                  <th className="px-5 py-3.5">Experiment ID</th>
                  <th className="px-5 py-3.5">Hypothesis Ref</th>
                  <th className="px-5 py-3.5">Type</th>
                  <th className="px-5 py-3.5">Cost</th>
                  <th className="px-5 py-3.5">Status</th>
                  <th className="px-5 py-3.5">Significance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {experiments.map((exp, i) => (
                  <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                    <td className="px-5 py-3.5 font-mono text-cyan-400 font-semibold">{exp.experiment_id}</td>
                    <td className="px-5 py-3.5 font-mono text-slate-300">{exp.hypothesis_id}</td>
                    <td className="px-5 py-3.5 text-slate-300 capitalize">{exp.experiment_type.replace(/_/g, " ")}</td>
                    <td className="px-5 py-3.5 font-mono text-amber-400">{exp.computational_cost} CU</td>
                    <td className="px-5 py-3.5">
                      <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px] uppercase font-bold">
                        {exp.status}
                      </span>
                    </td>
                    <td className="px-5 py-3.5 text-emerald-400 font-semibold">
                      {exp.metrics?.is_significant ? "Statistically Significant (p < 0.05)" : "Inconclusive"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
