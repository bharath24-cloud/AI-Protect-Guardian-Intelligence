import { Shield, Lock, Eye, AlertTriangle, Cpu, CheckCircle2, BookOpen, Layers } from "lucide-react";

export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-12">
      {/* Title */}
      <div className="space-y-3 text-center sm:text-left">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-semibold">
          <BookOpen className="w-3.5 h-3.5" />
          <span>Scientific Methodology & Ethics</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          About AI-Protect
        </h1>
        <p className="text-base text-slate-400 leading-relaxed">
          An adaptive adversarial cloaking research platform developed to evaluate empirical protection mechanisms for digital artworks and visual IP against unauthorized model scraping and ingestion.
        </p>
      </div>

      {/* Scientific Honesty Disclaimer */}
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-2xl p-6 space-y-3">
        <div className="flex items-center space-x-2.5 text-amber-400 font-semibold text-base">
          <AlertTriangle className="w-5 h-5" />
          <span>Scientific & Real-World Limitations</span>
        </div>
        <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
          Adversarial perturbations do <strong>not</strong> provide cryptographic guarantees. They exploit high-dimensional feature geometry in specific neural networks. Protection success depends heavily on the similarity between the surrogate model and the target model, as well as any image preprocessing pipelines (such as resizing or aggressive JPEG recompression) applied prior to model ingestion.
        </p>
      </div>

      {/* Methodology Section */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-white tracking-tight">Research Methodology</h2>
        <div className="space-y-4 text-sm text-slate-300 leading-relaxed">
          <p>
            AI-Protect follows the scientific paradigm:
          </p>
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 font-mono text-xs text-sky-400 text-center font-bold">
            Problem &rarr; Algorithm &rarr; Experiment &rarr; Measurement &rarr; Improvement &rarr; Real-World Application
          </div>
          <p>
            1. <strong>Bounded Optimization:</strong> Perturbations are strictly constrained under an $L_\infty$ ball of radius $\epsilon$, ensuring mathematical guarantees that maximum pixel modification does not exceed the budget.
          </p>
          <p>
            2. <strong>Transferability Matrix:</strong> By generating perturbations on a surrogate model (e.g. ResNet-50) and evaluating on disparate architectures (e.g. Vision Transformer ViT-B/16, ConvNeXt, and CLIP), we empirically measure black-box defense efficacy.
          </p>
          <p>
            3. <strong>Perceptual Quality:</strong> We evaluate image similarity using SSIM (Structural Similarity Index), PSNR (Peak Signal-to-Noise Ratio), and deep perceptual metrics (LPIPS) to ensure human imperceptibility.
          </p>
        </div>
      </section>

      {/* Privacy Guarantees */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-white tracking-tight">Privacy & Content Safety</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
            <div className="flex items-center space-x-2 text-emerald-400 font-semibold text-sm">
              <Lock className="w-4 h-4" />
              <span>Temporary Storage</span>
            </div>
            <p className="text-slate-400 leading-relaxed">
              Uploaded assets are stored in temporary scratch space and automatically cleaned up after the retention window.
            </p>
          </div>
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
            <div className="flex items-center space-x-2 text-sky-400 font-semibold text-sm">
              <CheckCircle2 className="w-4 h-4" />
              <span>SHA-256 Verification</span>
            </div>
            <p className="text-slate-400 leading-relaxed">
              Every asset is verified with a cryptographic SHA-256 digest to ensure reproducibility across experimental iterations without storing redundant copies.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
