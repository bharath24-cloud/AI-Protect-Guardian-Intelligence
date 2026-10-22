"use client";

import React, { useState } from "react";
import { Palette, Image as ImageIcon, UserCheck, Cpu, Sliders, ShieldAlert, Zap, Loader2 } from "lucide-react";
import { ProtectionConfigRequest, ImageUploadResponse } from "@/lib/api";

interface ProtectionConfigFormProps {
  image: ImageUploadResponse;
  onSubmit: (config: ProtectionConfigRequest) => Promise<void>;
  loading: boolean;
}

export default function ProtectionConfigForm({
  image,
  onSubmit,
  loading,
}: ProtectionConfigFormProps) {
  const [protectionMode, setProtectionMode] = useState<"artwork" | "general" | "portrait">("artwork");
  const [surrogateModel, setSurrogateModel] = useState<string>("resnet50");
  const [evalModels, setEvalModels] = useState<string[]>([
    "resnet50",
    "vit_b_16",
    "convnext_tiny",
    "clip_vit_b32",
  ]);
  const [epsilonNorm, setEpsilonNorm] = useState<number>(8); // representing 8/255
  const [iterations, setIterations] = useState<number>(20);
  const [stepSizeNorm, setStepSizeNorm] = useState<number>(2); // 2/255

  const modes = [
    {
      id: "artwork",
      title: "Artwork Protection",
      desc: "Optimized for digital paintings, illustrations, styles & photography IP protection.",
      icon: Palette,
      color: "border-sky-500 text-sky-400",
    },
    {
      id: "general",
      title: "General Image Protection",
      desc: "Standard benchmark mode for broad object classification disruption.",
      icon: ImageIcon,
      color: "border-emerald-500 text-emerald-400",
    },
    {
      id: "portrait",
      title: "Portrait Protection (Exp.)",
      desc: "Feature embedding perturbation for human subject/portrait representation.",
      icon: UserCheck,
      color: "border-purple-500 text-purple-400",
    },
  ];

  const availableModels = [
    { id: "resnet50", name: "ResNet-50", type: "CNN (Residual Network)" },
    { id: "vit_b_16", name: "ViT-B/16", type: "Vision Transformer" },
    { id: "convnext_tiny", name: "ConvNeXt-Tiny", type: "Modern ConvNet" },
    { id: "clip_vit_b32", name: "CLIP ViT-B/32", type: "Multimodal Embedding" },
  ];

  const toggleEvalModel = (id: string) => {
    if (evalModels.includes(id)) {
      if (evalModels.length > 1) {
        setEvalModels(evalModels.filter((m) => m !== id));
      }
    } else {
      setEvalModels([...evalModels, id]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const config: ProtectionConfigRequest = {
      image_id: image.image_id,
      protection_mode: protectionMode,
      surrogate_model: surrogateModel,
      evaluation_models: evalModels,
      epsilon: epsilonNorm / 255.0,
      iterations: iterations,
      step_size: stepSizeNorm / 255.0,
      loss_objective: "classification_disruption",
      random_init: true,
      random_seed: 42,
    };
    await onSubmit(config);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      {/* 1. Protection Mode */}
      <div className="space-y-3">
        <label className="text-sm font-semibold text-white flex items-center space-x-2">
          <Zap className="w-4 h-4 text-sky-400" />
          <span>1. Select Protection Mode</span>
        </label>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {modes.map((mode) => {
            const Icon = mode.icon;
            const isSelected = protectionMode === mode.id;
            return (
              <div
                key={mode.id}
                onClick={() => setProtectionMode(mode.id as any)}
                className={`p-4 rounded-xl cursor-pointer border transition-all ${
                  isSelected
                    ? "bg-slate-900 border-sky-500 shadow-md shadow-sky-500/10 ring-1 ring-sky-500/40"
                    : "bg-slate-900/40 border-slate-800 hover:border-slate-700 hover:bg-slate-900/70"
                }`}
              >
                <div className="flex items-center space-x-2.5 mb-2">
                  <div className={`p-1.5 rounded-lg bg-slate-800/80 ${isSelected ? "text-sky-400" : "text-slate-400"}`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="font-medium text-sm text-white">{mode.title}</span>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">{mode.desc}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* 2. Model Selection */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Surrogate Model */}
        <div className="space-y-3">
          <label className="text-sm font-semibold text-white flex items-center justify-between">
            <span className="flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-sky-400" />
              <span>Surrogate Model (Optimization)</span>
            </span>
            <span className="text-[11px] text-slate-400 font-normal">Generates perturbation</span>
          </label>
          <div className="space-y-2">
            {availableModels.map((m) => (
              <label
                key={m.id}
                className={`flex items-center justify-between p-3 rounded-xl border cursor-pointer transition-colors ${
                  surrogateModel === m.id
                    ? "bg-sky-500/10 border-sky-500/50 text-white"
                    : "bg-slate-900/40 border-slate-800 text-slate-300 hover:bg-slate-900/70"
                }`}
              >
                <div className="flex items-center space-x-3">
                  <input
                    type="radio"
                    name="surrogate"
                    checked={surrogateModel === m.id}
                    onChange={() => setSurrogateModel(m.id)}
                    className="text-sky-500 focus:ring-sky-500/30"
                  />
                  <span className="text-sm font-medium">{m.name}</span>
                </div>
                <span className="text-xs text-slate-400">{m.type}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Evaluation Models */}
        <div className="space-y-3">
          <label className="text-sm font-semibold text-white flex items-center justify-between">
            <span className="flex items-center space-x-2">
              <ShieldAlert className="w-4 h-4 text-emerald-400" />
              <span>Evaluation Models (Transferability)</span>
            </span>
            <span className="text-[11px] text-slate-400 font-normal">Measures cross-defense</span>
          </label>
          <div className="space-y-2">
            {availableModels.map((m) => {
              const checked = evalModels.includes(m.id);
              return (
                <label
                  key={m.id}
                  className={`flex items-center justify-between p-3 rounded-xl border cursor-pointer transition-colors ${
                    checked
                      ? "bg-slate-900/90 border-slate-700 text-white"
                      : "bg-slate-950/40 border-slate-900 text-slate-500"
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleEvalModel(m.id)}
                      className="rounded text-sky-500 focus:ring-sky-500/30"
                    />
                    <span className="text-sm font-medium">{m.name}</span>
                  </div>
                  <span className="text-xs text-slate-400">{m.type}</span>
                </label>
              );
            })}
          </div>
        </div>
      </div>

      {/* 3. Attack Hyperparameters */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-5 backdrop-blur-sm">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center space-x-2">
            <Sliders className="w-4 h-4 text-sky-400" />
            <span className="text-sm font-semibold text-white">Perturbation & Optimization Hyperparameters</span>
          </div>
          <span className="text-xs text-slate-400 font-mono">
            &epsilon; = {epsilonNorm}/255 ({(epsilonNorm / 255).toFixed(4)})
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Epsilon */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 font-medium">Perturbation Bound (&epsilon;)</span>
              <span className="text-sky-400 font-mono font-bold">{epsilonNorm}/255</span>
            </div>
            <input
              type="range"
              min="2"
              max="24"
              step="1"
              value={epsilonNorm}
              onChange={(e) => setEpsilonNorm(parseInt(e.target.value))}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-400"
            />
            <p className="text-[11px] text-slate-400">
              Maximum allowed L-infinity pixel change. Lower = more imperceptible.
            </p>
          </div>

          {/* Iterations */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 font-medium">Optimization Steps</span>
              <span className="text-sky-400 font-mono font-bold">{iterations}</span>
            </div>
            <input
              type="range"
              min="5"
              max="50"
              step="5"
              value={iterations}
              onChange={(e) => setIterations(parseInt(e.target.value))}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-400"
            />
            <p className="text-[11px] text-slate-400">
              PGD / optimization iterations to converge perturbation.
            </p>
          </div>

          {/* Step Size */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-300 font-medium">Step Size (&alpha;)</span>
              <span className="text-sky-400 font-mono font-bold">{stepSizeNorm}/255</span>
            </div>
            <input
              type="range"
              min="1"
              max="8"
              step="1"
              value={stepSizeNorm}
              onChange={(e) => setStepSizeNorm(parseInt(e.target.value))}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-400"
            />
            <p className="text-[11px] text-slate-400">
              Gradient ascent step per iteration for adversarial loss.
            </p>
          </div>
        </div>
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={loading}
        className="w-full py-4 px-6 rounded-xl font-semibold text-white bg-gradient-to-r from-sky-500 via-cyan-500 to-emerald-500 hover:from-sky-400 hover:to-emerald-400 shadow-lg shadow-sky-500/25 transition-all flex items-center justify-center space-x-2 text-base disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            <span>Scheduling Adversarial Cloaking Engine...</span>
          </>
        ) : (
          <>
            <Zap className="w-5 h-5" />
            <span>Generate Adversarial Protection</span>
          </>
        )}
      </button>
    </form>
  );
}
