"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import {
  Shield,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Cpu,
  Layers,
  Lock,
  Loader2,
  Sliders,
} from "lucide-react";
import ImageUploader from "@/components/ImageUploader";
import ProtectionConfigForm from "@/components/ProtectionConfigForm";
import LiquidCanvas from "@/components/LiquidCanvas";
import { ImageUploadResponse, ProtectionConfigRequest, startProtection } from "@/lib/api";

export default function ProtectPage() {
  const router = useRouter();
  const [selectedImage, setSelectedImage] = useState<ImageUploadResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [jobError, setJobError] = useState<string | null>(null);
  const [createdJobId, setCreatedJobId] = useState<string | null>(null);
  const [showCanvasIntro, setShowCanvasIntro] = useState(true);

  const handleImageUploaded = (imageData: ImageUploadResponse) => {
    setSelectedImage(imageData);
    setJobError(null);
    setCreatedJobId(null);
  };

  const handleClear = () => {
    setSelectedImage(null);
    setJobError(null);
    setCreatedJobId(null);
  };

  const handleProtectionSubmit = async (config: ProtectionConfigRequest) => {
    setLoading(true);
    setJobError(null);
    try {
      const job = await startProtection(config);
      setCreatedJobId(job.job_id);
      // Navigate to results page
      router.push(`/results/${job.job_id}`);
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || "Failed to schedule protection job.";
      setJobError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-10 pb-16">
      {/* Studio Header Banner with Liquid Canvas Intro Aura */}
      <div className="relative rounded-3xl bg-[#181C24] border border-[#262C38] overflow-hidden p-6 sm:p-10 shadow-2xl">
        {/* Background Liquid Canvas Particles */}
        {showCanvasIntro && (
          <div className="absolute inset-0 opacity-40 pointer-events-none">
            <LiquidCanvas active={true} onSettled={() => {}} className="w-full h-full" />
          </div>
        )}

        <div className="relative z-10 max-w-2xl space-y-4">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-[#5CE1D2]/10 border border-[#5CE1D2]/20 text-[#5CE1D2] text-xs font-semibold">
            <Shield className="w-3.5 h-3.5" />
            <span>Adversarial Synthesis Studio</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-extrabold text-[#F4F1E9] tracking-tight leading-tight">
            Synthesize Bounded Adversarial Cloaks for{" "}
            <span className="bg-gradient-to-r from-[#5CE1D2] via-[#A9A5FF] to-[#5CE1D2] bg-clip-text text-transparent">
              Digital Artwork
            </span>
          </h1>

          <p className="text-xs sm:text-sm text-[#9BA5B4] leading-relaxed">
            Generate imperceptible, mathematically bounded perturbations (&epsilon; &le; 8/255) at native resolution without downsampling blur. Disrupt unauthorized AI scrapers and feature extractors while preserving human aesthetic fidelity.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-2 text-xs font-mono text-[#9BA5B4]">
            <span className="flex items-center space-x-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#5CE1D2]" />
              <span>Native-Resolution Delta</span>
            </span>
            <span className="flex items-center space-x-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#5CE1D2]" />
              <span>Strict L-&infin; Bound Check</span>
            </span>
            <span className="flex items-center space-x-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#A9A5FF]" />
              <span>Multi-Model Evaluation</span>
            </span>
          </div>
        </div>
      </div>

      {/* Step 1: Upload Asset */}
      <section className="space-y-4">
        <div className="flex items-center justify-between border-b border-[#262C38] pb-3">
          <h2 className="text-base font-bold text-[#F4F1E9] flex items-center space-x-2">
            <span className="w-6 h-6 rounded-lg bg-[#5CE1D2]/15 text-[#5CE1D2] border border-[#5CE1D2]/30 text-xs flex items-center justify-center font-mono font-bold">
              1
            </span>
            <span>Upload Visual Asset or Artwork</span>
          </h2>
          <span className="text-xs text-[#9BA5B4] font-mono">PNG, JPEG, WebP &bull; Max 20MB</span>
        </div>

        <ImageUploader
          onImageUploaded={handleImageUploaded}
          selectedImage={selectedImage}
          onClear={handleClear}
        />
      </section>

      {/* Step 2: Configure & Launch */}
      {selectedImage && (
        <section className="space-y-6 pt-4 border-t border-slate-800">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h2 className="text-base font-bold text-white flex items-center space-x-2">
              <span className="w-6 h-6 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 text-xs flex items-center justify-center font-mono font-bold">
                2
              </span>
              <span>Configure Optimization Hyperparameters</span>
            </h2>
            <span className="text-xs text-cyan-400 font-mono">Differentiable PGD Loop</span>
          </div>

          <ProtectionConfigForm
            image={selectedImage}
            onSubmit={handleProtectionSubmit}
            loading={loading}
          />

          {jobError && (
            <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-400 flex items-center space-x-3 text-xs">
              <AlertTriangle className="w-5 h-5 flex-shrink-0" />
              <span>{jobError}</span>
            </div>
          )}

          {createdJobId && (
            <div className="p-4 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 flex items-center justify-between text-xs">
              <div className="flex items-center space-x-3">
                <CheckCircle2 className="w-5 h-5 flex-shrink-0" />
                <span>Protection job registered successfully (ID: {createdJobId}).</span>
              </div>
              <button
                onClick={() => router.push(`/results/${createdJobId}`)}
                className="inline-flex items-center space-x-1.5 font-bold hover:underline"
              >
                <span>Open Artwork Microscope</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
