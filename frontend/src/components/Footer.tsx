import { AlertTriangle, ShieldCheck } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-surface-border bg-background/60 backdrop-blur-sm mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 mb-8 flex items-start space-x-3 text-xs text-slate-400">
          <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-semibold text-slate-200">Scientific & Empirical Disclaimer:</span>
            <p>
              Adversarial cloaking creates bounded input perturbations designed to alter the output responses of selected AI models.
              AI-Protect does not guarantee 100% prevention against AI scraping, model ingestion, or future neural architectures. Protection effectiveness
              is empirically measured and varies based on model target, surrogate selection, and subsequent transformations (e.g. heavy compression or resizing).
            </p>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <div className="flex items-center space-x-2">
            <ShieldCheck className="w-4 h-4 text-sky-400" />
            <span>AI-Protect Engine &bull; Academic & Creative IP Defense Research</span>
          </div>
          <div className="flex space-x-6">
            <span>Modular PyTorch Pipeline</span>
            <span>Transferability & Robustness Benchmarking</span>
            <span>Privacy-Preserving Temp Storage</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
