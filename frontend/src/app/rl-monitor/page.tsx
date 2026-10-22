"use client";

import React, { useState, useEffect } from "react";
import { Network, TrendingUp, Zap, Target, Award, BarChart3 } from "lucide-react";

export default function RLMonitorPage() {
  const [baselines, setBaselines] = useState<any[]>([]);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/discovery/baselines")
      .then((res) => res.json())
      .then((data) => setBaselines(data.data || []))
      .catch(() => {
        setBaselines([
          { Strategy: "Random Selection", Cumulative_RL_Reward: 9.131, Mean_Experiment_R2: 0.885, Success_Rate: "100.0%", Runtime: "6.8s" },
          { Strategy: "Uncertainty Sampling", Cumulative_RL_Reward: 8.308, Mean_Experiment_R2: 0.678, Success_Rate: "100.0%", Runtime: "7.7s" },
          { Strategy: "Greedy Information Gain", Cumulative_RL_Reward: 8.567, Mean_Experiment_R2: 0.743, Success_Rate: "100.0%", Runtime: "11.2s" },
          { Strategy: "Heuristic Multi-Criteria", Cumulative_RL_Reward: 7.477, Mean_Experiment_R2: 0.621, Success_Rate: "75.0%", Runtime: "8.2s" },
          { Strategy: "Proposed RL AI Scientist", Cumulative_RL_Reward: 4.957, Mean_Experiment_R2: 0.390, Success_Rate: "25.0%", Runtime: "6.5s" }
        ]);
      });
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white flex items-center gap-2">
            <Network className="w-6 h-6 text-cyan-400" />
            <span>Reinforcement Learning Policy Monitor</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Markov Decision Process (MDP) optimization curves, multi-objective reward tradeoffs, and baseline comparisons.
          </p>
        </div>

        {/* Reward Function Card */}
        <div className="p-6 rounded-2xl bg-gradient-to-r from-slate-900 to-indigo-950/40 border border-slate-800 space-y-4">
          <h2 className="text-sm font-bold text-white flex items-center gap-2">
            <Award className="w-4 h-4 text-amber-400" />
            <span>Mathematical Multi-Objective Scientific Reward Formulation</span>
          </h2>
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 font-mono text-xs text-cyan-300">
            R(s, a) = 1.0 &times; InfoGain(e) + 0.5 &times; Novelty(e) + 1.0 &times; &Delta;Acc(e) - 0.2 &times; Cost(e) - 0.5 &times; Redundancy(e)
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-center text-xs pt-2">
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <div className="text-slate-400 text-[10px]">Info Gain Weight</div>
              <div className="font-bold text-white font-mono">w₁ = 1.0</div>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <div className="text-slate-400 text-[10px]">Novelty Weight</div>
              <div className="font-bold text-white font-mono">w₂ = 0.5</div>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <div className="text-slate-400 text-[10px]">Acc Delta Weight</div>
              <div className="font-bold text-white font-mono">w₃ = 1.0</div>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <div className="text-slate-400 text-[10px]">Cost Penalty</div>
              <div className="font-bold text-white font-mono">w₄ = 0.2</div>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
              <div className="text-slate-400 text-[10px]">Redundancy Penalty</div>
              <div className="font-bold text-white font-mono">w₅ = 0.5</div>
            </div>
          </div>
        </div>

        {/* Baseline Benchmark Table */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl overflow-hidden shadow-xl space-y-4 p-6">
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-cyan-400" />
            <span>Comparative Strategy Benchmarks Across Discovery Cycles</span>
          </h2>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 border-b border-slate-800 text-slate-400 uppercase font-semibold">
                <tr>
                  <th className="px-4 py-3">Experiment Selection Strategy</th>
                  <th className="px-4 py-3">Cumulative Reward</th>
                  <th className="px-4 py-3">Mean Experiment R²</th>
                  <th className="px-4 py-3">Supported Ratio</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {baselines.map((b, i) => (
                  <tr key={i} className="hover:bg-slate-800/30">
                    <td className="px-4 py-3 font-semibold text-slate-200">{b.Strategy}</td>
                    <td className="px-4 py-3 font-mono text-cyan-400 font-bold">{b["Cumulative RL Reward"] ?? b.Cumulative_RL_Reward}</td>
                    <td className="px-4 py-3 font-mono text-indigo-300">{b["Mean Experiment R²"] ?? b.Mean_Experiment_R2}</td>
                    <td className="px-4 py-3 font-mono text-emerald-400 font-bold">{b["Success Rate (%)"] ? `${b["Success Rate (%)"]}%` : b.Success_Rate}</td>
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
