"use client";

import React, { useState, useEffect } from "react";
import { Database, Network, Atom, Share2, Layers, CheckCircle } from "lucide-react";

export default function ScientificMemoryPage() {
  const [graphData, setGraphData] = useState<any>({ nodes: [], edges: [], stats: {} });

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/discovery/knowledge-graph")
      .then((res) => res.json())
      .then((data) => setGraphData(data))
      .catch(() => {
        setGraphData({
          nodes: [
            { id: "mat_oxides", type: "Material", label: "Perovskite Oxides" },
            { id: "prop_bandgap", type: "Property", label: "Band Gap (eV)" },
            { id: "hyp_01", type: "Hypothesis", label: "Band gap scales with (EN_delta)^1.5..." },
            { id: "exp_01", type: "Experiment", label: "Exp: exp-01a2b3c4" },
            { id: "evi_01", type: "Evidence", label: "Evidence: supported" },
            { id: "conc_01", type: "Conclusion", label: "Status: supported" }
          ],
          edges: [
            { source: "mat_oxides", target: "prop_bandgap", relation: "exhibits_property" },
            { source: "prop_bandgap", target: "hyp_01", relation: "investigated_by" },
            { source: "hyp_01", target: "exp_01", relation: "tested_by" },
            { source: "exp_01", target: "evi_01", relation: "produces_evidence" },
            { source: "evi_01", target: "conc_01", relation: "leads_to_conclusion" }
          ],
          stats: { total_nodes: 6, total_edges: 5, node_types: ["Material", "Property", "Hypothesis", "Experiment", "Evidence", "Conclusion"] }
        });
      });
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white flex items-center gap-2">
            <Database className="w-6 h-6 text-emerald-400" />
            <span>Persistent Scientific Memory & Knowledge Graph</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Bipartite relational graph linking Materials &rarr; Physical Properties &rarr; Hypotheses &rarr; Experiments &rarr; Evidence.
          </p>
        </div>

        {/* Knowledge Hierarchy Graph Visualization Cards */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Network className="w-4 h-4 text-cyan-400" />
              <span>Relational Scientific Entities</span>
            </h2>
            <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
              <span className="text-cyan-400 font-bold">{graphData.nodes?.length ?? 6}</span> Nodes |{" "}
              <span className="text-indigo-400 font-bold">{graphData.edges?.length ?? 5}</span> Edges
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {graphData.nodes?.map((node: any, i: number) => (
              <div key={i} className="p-4 rounded-xl bg-slate-950/80 border border-slate-800/90 space-y-2">
                <div className="flex items-center justify-between">
                  <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold tracking-wider ${
                    node.type === "Material"
                      ? "bg-blue-500/10 text-blue-400 border border-blue-500/20"
                      : node.type === "Property"
                      ? "bg-purple-500/10 text-purple-400 border border-purple-500/20"
                      : node.type === "Hypothesis"
                      ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20"
                      : node.type === "Experiment"
                      ? "bg-indigo-500/10 text-indigo-400 border border-indigo-500/20"
                      : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                  }`}>
                    {node.type}
                  </span>
                  <span className="font-mono text-[10px] text-slate-400">{node.id}</span>
                </div>
                <div className="text-xs font-semibold text-slate-200">{node.label}</div>
              </div>
            ))}
          </div>

          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80">
            <h3 className="text-xs font-bold text-slate-300 mb-2">Relational Graph Links:</h3>
            <div className="space-y-1.5 font-mono text-[11px] text-slate-400">
              {graphData.edges?.map((e: any, idx: number) => (
                <div key={idx} className="flex items-center gap-2">
                  <span className="text-cyan-400">{e.source}</span>
                  <span className="text-slate-500">&mdash;[{e.relation}]&rarr;</span>
                  <span className="text-indigo-400">{e.target}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
