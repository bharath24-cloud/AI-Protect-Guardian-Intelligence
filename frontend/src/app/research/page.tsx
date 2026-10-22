"use client";

import React, { useState } from "react";
import { FlaskConical, Download, Filter, Sliders, Sparkles, BarChart2 } from "lucide-react";

export default function ResearchLabPage() {
  const [selectedDataset, setSelectedDataset] = useState<string>("wikiart");

  const sampleExperiments = [
    {
      id: "EXP-001",
      model: "ResNet-50",
      epsilon: "8/255",
      iters: 20,
      disruption: "92.4%",
      ssim: "0.964",
      psnr: "38.2 dB",
      lpips: "0.028",
      dataset: "WikiArt",
    },
    {
      id: "EXP-002",
      model: "ViT-B/16",
      epsilon: "8/255",
      iters: 20,
      disruption: "88.1%",
      ssim: "0.968",
      psnr: "39.1 dB",
      lpips: "0.024",
      dataset: "WikiArt",
    },
    {
      id: "EXP-003",
      model: "ConvNeXt-Tiny",
      epsilon: "12/255",
      iters: 30,
      disruption: "95.6%",
      ssim: "0.932",
      psnr: "34.7 dB",
      lpips: "0.045",
      dataset: "CIFAR-10",
    },
    {
      id: "EXP-004",
      model: "CLIP ViT-B/32",
      epsilon: "8/255",
      iters: 20,
      disruption: "84.3%",
      ssim: "0.961",
      psnr: "37.8 dB",
      lpips: "0.031",
      dataset: "ImageNet Subset",
    },
  ];

  const exportCSV = () => {
    const headers = "Experiment,Model,Epsilon,Iterations,AI Disruption,SSIM,PSNR,LPIPS,Dataset\n";
    const rows = sampleExperiments
      .map(
        (e) =>
          `${e.id},${e.model},${e.epsilon},${e.iters},${e.disruption},${e.ssim},${e.psnr},${e.lpips},${e.dataset}`
      )
      .join("\n");
    const blob = new Blob([headers + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ai_protect_experiments.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="space-y-1">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-400 text-xs font-semibold">
            <FlaskConical className="w-3.5 h-3.5" />
            <span>Research & Benchmarking Lab</span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">
            Adversarial Experiments & Reproducibility Matrix
          </h1>
          <p className="text-sm text-slate-400 max-w-2xl">
            Controlled empirical runs evaluating epsilon scaling, architecture cross-transferability, and perceptual trade-offs across standardized datasets.
          </p>
        </div>

        <button
          onClick={exportCSV}
          className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl font-semibold text-white bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs transition-colors"
        >
          <Download className="w-4 h-4 text-sky-400" />
          <span>Export CSV</span>
        </button>
      </div>

      {/* Dataset Filter */}
      <div className="flex items-center space-x-3 text-xs">
        <span className="text-slate-400 font-medium">Dataset Filter:</span>
        {[
          { id: "all", name: "All Datasets" },
          { id: "wikiart", name: "WikiArt (IP Artworks)" },
          { id: "cifar10", name: "CIFAR-10" },
          { id: "imagenet", name: "ImageNet Subset" },
        ].map((ds) => (
          <button
            key={ds.id}
            onClick={() => setSelectedDataset(ds.id)}
            className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
              selectedDataset === ds.id
                ? "bg-sky-500/20 text-sky-300 border border-sky-500/30"
                : "bg-slate-900/60 text-slate-400 border border-slate-800 hover:text-slate-200"
            }`}
          >
            {ds.name}
          </button>
        ))}
      </div>

      {/* Experiment Table */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl overflow-hidden backdrop-blur-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-950/80 text-slate-400 font-semibold uppercase tracking-wider text-[11px] border-b border-slate-800">
              <tr>
                <th className="py-3.5 px-4">Experiment</th>
                <th className="py-3.5 px-4">Dataset</th>
                <th className="py-3.5 px-4">Surrogate Model</th>
                <th className="py-3.5 px-4">Epsilon (&epsilon;)</th>
                <th className="py-3.5 px-4">Iterations</th>
                <th className="py-3.5 px-4">AI Disruption</th>
                <th className="py-3.5 px-4">SSIM</th>
                <th className="py-3.5 px-4">PSNR</th>
                <th className="py-3.5 px-4">LPIPS</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {sampleExperiments.map((exp) => (
                <tr key={exp.id} className="hover:bg-slate-800/30 transition-colors">
                  <td className="py-3.5 px-4 font-mono font-bold text-sky-400">{exp.id}</td>
                  <td className="py-3.5 px-4">{exp.dataset}</td>
                  <td className="py-3.5 px-4 font-medium text-white">{exp.model}</td>
                  <td className="py-3.5 px-4 font-mono">{exp.epsilon}</td>
                  <td className="py-3.5 px-4 font-mono">{exp.iters}</td>
                  <td className="py-3.5 px-4 font-bold text-emerald-400">{exp.disruption}</td>
                  <td className="py-3.5 px-4 font-mono">{exp.ssim}</td>
                  <td className="py-3.5 px-4 font-mono">{exp.psnr}</td>
                  <td className="py-3.5 px-4 font-mono">{exp.lpips}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
