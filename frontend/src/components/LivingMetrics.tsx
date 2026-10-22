"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  Shield,
  Activity,
  Layers,
  HelpCircle,
  TrendingDown,
  Info,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Sparkles,
} from "lucide-react";
import {
  ClassificationDisruptionDetail,
  PerceptualQualityMetrics,
} from "@/lib/api";

interface LivingMetricsProps {
  classification?: ClassificationDisruptionDetail | null;
  qualityMetrics?: PerceptualQualityMetrics | null;
  overallScore?: number | null;
  aiDisruptionScore?: number | null;
  transferabilityScore?: number | null;
  robustnessScore?: number | null;
  runtimeSeconds?: number | null;
  className?: string;
}

export default function LivingMetrics({
  classification,
  qualityMetrics,
  overallScore,
  aiDisruptionScore,
  transferabilityScore,
  robustnessScore,
  runtimeSeconds,
  className = "",
}: LivingMetricsProps) {
  const cls = classification;
  const qm = qualityMetrics;

  // Chart data: compare original vs cloaked Top-5 style probabilities
  const chartData =
    cls?.original_prediction?.top5?.slice(0, 5).map((p) => {
      const cloakedMatch = cls.cloaked_prediction?.top5?.find(
        (cp) => cp.style_name === p.style_name
      );
      const cloakedConf = cloakedMatch ? cloakedMatch.confidence_percent : 0;

      return {
        style: p.style_name.length > 14 ? `${p.style_name.slice(0, 12)}...` : p.style_name,
        fullName: p.style_name,
        original: p.confidence_percent,
        cloaked: cloakedConf,
      };
    }) || [];

  return (
    <div className={`space-y-6 ${className}`}>
      {/* 1. Primary Defense Summary Scores */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Overall Protection Score */}
        <div className="p-4 rounded-2xl bg-[#181C24] border border-[#262C38] text-center space-y-1">
          <div className="flex items-center justify-center space-x-1 text-[#9BA5B4] text-[10px] font-semibold uppercase tracking-wider">
            <span>Overall Protection</span>
          </div>
          <div className="text-2xl font-bold font-mono text-[#5CE1D2]">
            {overallScore !== null && overallScore !== undefined ? `${overallScore}%` : "—"}
          </div>
          <span className="text-[10px] text-[#9BA5B4] block font-mono">Composite Benchmark</span>
        </div>

        {/* AI Disruption Score */}
        <div className="p-4 rounded-2xl bg-[#181C24] border border-[#262C38] text-center space-y-1">
          <div className="flex items-center justify-center space-x-1 text-[#9BA5B4] text-[10px] font-semibold uppercase tracking-wider">
            <span>AI Disruption</span>
          </div>
          <div className="text-2xl font-bold font-mono text-[#5CE1D2]">
            {aiDisruptionScore !== null && aiDisruptionScore !== undefined ? `${aiDisruptionScore}%` : "—"}
          </div>
          <span className="text-[10px] text-[#9BA5B4] block font-mono">Surrogate Embedding Drift</span>
        </div>

        {/* Cross-Model Transferability */}
        <div className="p-4 rounded-2xl bg-[#181C24] border border-[#262C38] text-center space-y-1">
          <div className="flex items-center justify-center space-x-1 text-[#9BA5B4] text-[10px] font-semibold uppercase tracking-wider">
            <span>Transferability</span>
          </div>
          <div className="text-2xl font-bold font-mono text-[#A9A5FF]">
            {transferabilityScore !== null && transferabilityScore !== undefined
              ? `${transferabilityScore}%`
              : "—"}
          </div>
          <span className="text-[10px] text-[#9BA5B4] block font-mono">Black-Box Generalization</span>
        </div>

        {/* Transformation Robustness */}
        <div className="p-4 rounded-2xl bg-[#181C24] border border-[#262C38] text-center space-y-1">
          <div className="flex items-center justify-center space-x-1 text-[#9BA5B4] text-[10px] font-semibold uppercase tracking-wider">
            <span>Robustness</span>
          </div>
          <div className="text-2xl font-bold font-mono text-[#5CE1D2]">
            {robustnessScore !== null && robustnessScore !== undefined
              ? `${robustnessScore}%`
              : "—"}
          </div>
          <span className="text-[10px] text-[#9BA5B4] block font-mono">JPEG & Filter Survival</span>
        </div>
      </div>

      {/* 2. Mathematical Quality Metrics vs Disruption Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left: Perceptual Image Quality & Norms */}
        <div className="p-5 rounded-2xl bg-[#181C24] border border-[#262C38] space-y-4">
          <div className="flex items-center justify-between border-b border-[#262C38] pb-3">
            <div className="flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-[#5CE1D2]" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-[#F4F1E9]">
                Perceptual Fidelity & Norms
              </h3>
            </div>
            <span className="text-[10px] font-mono text-[#9BA5B4]">Target: SSIM ≥ 0.85</span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs font-mono">
            {/* SSIM */}
            <div className="p-3 rounded-xl bg-[#101217] border border-[#262C38] space-y-1">
              <span className="text-[10px] text-[#9BA5B4] uppercase block">SSIM Index</span>
              <div className="text-lg font-bold text-[#5CE1D2]">
                {qm?.ssim !== null && qm?.ssim !== undefined ? qm.ssim.toFixed(4) : "—"}
              </div>
              <span className="text-[9px] text-[#9BA5B4] block">Higher = Better fidelity</span>
            </div>

            {/* PSNR */}
            <div className="p-3 rounded-xl bg-[#101217] border border-[#262C38] space-y-1">
              <span className="text-[10px] text-[#9BA5B4] uppercase block">PSNR (dB)</span>
              <div className="text-lg font-bold text-[#5CE1D2]">
                {qm?.psnr_db !== null && qm?.psnr_db !== undefined ? `${qm.psnr_db.toFixed(1)} dB` : "—"}
              </div>
              <span className="text-[9px] text-[#9BA5B4] block">&gt;30 dB = High fidelity</span>
            </div>

            {/* LPIPS */}
            <div className="p-3 rounded-xl bg-[#101217] border border-[#262C38] space-y-1">
              <span className="text-[10px] text-[#9BA5B4] uppercase block">LPIPS (AlexNet)</span>
              <div className="text-lg font-bold text-[#A9A5FF]">
                {qm?.lpips !== null && qm?.lpips !== undefined ? qm.lpips.toFixed(4) : "—"}
              </div>
              <span className="text-[9px] text-[#9BA5B4] block">Lower = Less perceptual drift</span>
            </div>

            {/* L-infinity */}
            <div className="p-3 rounded-xl bg-[#101217] border border-[#262C38] space-y-1">
              <span className="text-[10px] text-[#9BA5B4] uppercase block">Max |&delta;|&infin;</span>
              <div className="text-lg font-bold text-[#F4F1E9]">
                {qm?.linf_norm !== null && qm?.linf_norm !== undefined ? qm.linf_norm.toFixed(4) : "—"}
              </div>
              <span className="text-[9px] text-[#5CE1D2] block">&le; 8/255 &approx; 0.0314</span>
            </div>
          </div>
        </div>

        {/* Right: Neural Disruption & Embedding Shift */}
        <div className="p-5 rounded-2xl bg-[#181C24] border border-[#262C38] space-y-4">
          <div className="flex items-center justify-between border-b border-[#262C38] pb-3">
            <div className="flex items-center space-x-2">
              <Activity className="w-4 h-4 text-[#A9A5FF]" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-[#F4F1E9]">
                Neural Disruption & Drift
              </h3>
            </div>
            <span className="text-[10px] font-mono text-[#5CE1D2]">Surrogate: ResNet-50</span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs font-mono">
            {/* Classification Flip */}
            <div className="p-3 rounded-xl bg-[#101217] border border-[#262C38] space-y-1">
              <span className="text-[10px] text-[#9BA5B4] uppercase block">Top-1 Disrupted</span>
              <div
                className={`text-lg font-bold ${
                  cls?.top1_prediction_changed ? "text-[#5CE1D2]" : "text-[#FF8C79]"
                }`}
              >
                {cls?.top1_prediction_changed ? "DISRUPTED" : "UNCHANGED"}
              </div>
              <span className="text-[9px] text-[#9BA5B4] block">Model prediction shifted</span>
            </div>

            {/* Confidence Drop */}
            <div className="p-3 rounded-xl bg-[#101217] border border-[#262C38] space-y-1">
              <span className="text-[10px] text-[#9BA5B4] uppercase block">Confidence Drop</span>
              <div className="text-lg font-bold text-[#5CE1D2]">
                {cls?.confidence_drop_percent !== undefined
                  ? `${cls.confidence_drop_percent.toFixed(1)}%`
                  : "—"}
              </div>
              <span className="text-[9px] text-[#9BA5B4] block">Reduction in top confidence</span>
            </div>

            {/* Cosine Distance */}
            <div className="p-3 rounded-xl bg-[#101217] border border-[#262C38] space-y-1">
              <span className="text-[10px] text-[#9BA5B4] uppercase block">Embedding Cosine Drift</span>
              <div className="text-lg font-bold text-[#A9A5FF]">
                {cls?.embedding_cosine_distance !== undefined
                  ? cls.embedding_cosine_distance.toFixed(4)
                  : "—"}
              </div>
              <span className="text-[9px] text-[#9BA5B4] block">Feature space shift</span>
            </div>

            {/* L2 Norm */}
            <div className="p-3 rounded-xl bg-[#101217] border border-[#262C38] space-y-1">
              <span className="text-[10px] text-[#9BA5B4] uppercase block">L2 Norm Energy</span>
              <div className="text-lg font-bold text-[#F4F1E9]">
                {qm?.l2_norm !== null && qm?.l2_norm !== undefined ? qm.l2_norm.toFixed(2) : "—"}
              </div>
              <span className="text-[9px] text-[#9BA5B4] block">Perturbation Euclidean norm</span>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Top-5 Prediction Probability Shift (Recharts Visualization) */}
      {chartData.length > 0 && (
        <div className="p-5 rounded-2xl bg-[#181C24] border border-[#262C38] space-y-4">
          <div className="flex items-center justify-between border-b border-[#262C38] pb-3">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-[#F4F1E9]">
                Top-5 Style Classification Probability Shift
              </h3>
              <p className="text-[11px] text-[#9BA5B4] mt-0.5">
                Compares unperturbed original image probabilities (Teal) against protected cloaked output (Coral/Muted).
              </p>
            </div>
            <div className="flex items-center space-x-3 text-xs font-mono">
              <span className="flex items-center space-x-1.5 text-[#5CE1D2]">
                <span className="w-2.5 h-2.5 rounded-sm bg-[#5CE1D2]" />
                <span>Original</span>
              </span>
              <span className="flex items-center space-x-1.5 text-[#FF8C79]">
                <span className="w-2.5 h-2.5 rounded-sm bg-[#FF8C79]" />
                <span>Cloaked</span>
              </span>
            </div>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <XAxis dataKey="style" tick={{ fill: "#9BA5B4", fontSize: 10 }} />
                <YAxis unit="%" tick={{ fill: "#9BA5B4", fontSize: 10 }} domain={[0, 100]} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-[#181C24] border border-[#262C38] p-2.5 rounded-xl text-xs font-mono shadow-xl space-y-1">
                          <span className="font-bold text-[#F4F1E9] block">{data.fullName}</span>
                          <div className="text-[#5CE1D2]">Original: {data.original.toFixed(1)}%</div>
                          <div className="text-[#FF8C79]">Cloaked: {data.cloaked.toFixed(1)}%</div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="original" fill="#5CE1D2" radius={[4, 4, 0, 0]} />
                <Bar dataKey="cloaked" fill="#FF8C79" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
