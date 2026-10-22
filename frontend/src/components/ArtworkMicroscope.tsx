"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Minimize2,
  RotateCcw,
  Sliders,
  Layers,
  SplitSquareVertical,
  Activity,
  Eye,
  Info,
  Check,
  Compass,
  AlertTriangle,
  Flame,
} from "lucide-react";
import { BACKEND_ROOT_URL } from "@/lib/api";

interface ArtworkMicroscopeProps {
  originalUrl: string;
  cloakedUrl?: string | null;
  differenceMapUrl?: string | null;
  metrics?: {
    ssim?: number | null;
    psnr_db?: number | null;
    lpips?: number | null;
    linf_norm?: number | null;
    l2_norm?: number | null;
    epsilon_passed?: boolean | null;
  };
  className?: string;
}

export default function ArtworkMicroscope({
  originalUrl,
  cloakedUrl,
  differenceMapUrl,
  metrics,
  className = "",
}: ArtworkMicroscopeProps) {
  // Modes: "split" (slider), "original", "cloaked", "xray" (difference)
  const [mode, setMode] = useState<"split" | "original" | "cloaked" | "xray">("split");

  // Transform state: zoom & pan
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Split-slider position (0 to 100%)
  const [sliderPos, setSliderPos] = useState(50);
  const [isSliding, setIsSliding] = useState(false);

  // Difference map amplification & colormap
  const [amplification, setAmplification] = useState<number>(32);
  const [colormap, setColormap] = useState<"heatmap" | "grayscale">("heatmap");

  // Cursor coordinates inside the image
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);

  // Container refs & Fullscreen
  const viewerRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const getFullUrl = (url?: string | null) => {
    if (!url) return "";
    return url.startsWith("http") ? url : `${BACKEND_ROOT_URL}${url}`;
  };

  const origFull = getFullUrl(originalUrl);
  const cloakedFull = getFullUrl(cloakedUrl || originalUrl);
  const diffFull = getFullUrl(differenceMapUrl || "");

  // Zoom handlers
  const handleZoomIn = () => setScale((s) => Math.min(5, Number((s * 1.25).toFixed(2))));
  const handleZoomOut = () => setScale((s) => Math.max(0.4, Number((s * 0.8).toFixed(2))));
  const handleReset = () => {
    setScale(1);
    setPan({ x: 0, y: 0 });
    setSliderPos(50);
  };
  const handleActualSize = () => {
    setScale(1);
    setPan({ x: 0, y: 0 });
  };

  // Wheel zoom
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY * -0.0015;
    setScale((s) => Math.max(0.4, Math.min(5, Number((s + delta).toFixed(2)))));
  };

  // Pan drag handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if (isSliding) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }

    if (viewerRef.current) {
      const rect = viewerRef.current.getBoundingClientRect();
      const x = Math.round(e.clientX - rect.left);
      const y = Math.round(e.clientY - rect.top);
      setCursorPos({ x, y });

      if (isSliding) {
        const pct = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
        setSliderPos(pct);
      }
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setIsSliding(false);
  };

  // Touch handlers for mobile
  const handleTouchMove = (e: React.TouchEvent) => {
    if (viewerRef.current && e.touches.length === 1) {
      const touch = e.touches[0];
      const rect = viewerRef.current.getBoundingClientRect();
      const pct = Math.max(0, Math.min(100, ((touch.clientX - rect.left) / rect.width) * 100));
      setSliderPos(pct);
    }
  };

  const toggleFullscreen = () => {
    if (!viewerRef.current) return;
    if (!document.fullscreenElement) {
      viewerRef.current.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
    }
  };

  return (
    <div
      ref={viewerRef}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        setIsDragging(false);
        setIsSliding(false);
        setCursorPos(null);
      }}
      className={`relative rounded-2xl bg-[#101217] border border-[#262C38] overflow-hidden shadow-2xl flex flex-col select-none ${
        isFullscreen ? "h-screen w-screen rounded-none" : "h-[620px]"
      } ${className}`}
    >
      {/* Top Microscope Header & View Modes Bar */}
      <div className="z-20 flex flex-wrap items-center justify-between px-4 py-3 bg-[#181C24]/90 border-b border-[#262C38] backdrop-blur-md gap-3">
        {/* Left: Mode Switcher */}
        <div className="flex items-center space-x-1.5 p-1 rounded-xl bg-[#101217] border border-[#262C38]">
          {[
            { id: "split", label: "Split Slider", icon: SplitSquareVertical },
            { id: "original", label: "Original Asset", icon: Eye },
            { id: "cloaked", label: "Cloaked Asset", icon: Layers },
            { id: "xray", label: "Difference Heatmap", icon: Activity },
          ].map((tab) => {
            const Icon = tab.icon;
            const active = mode === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setMode(tab.id as any)}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  active
                    ? "bg-[#202631] text-[#5CE1D2] font-semibold shadow-sm"
                    : "text-[#9BA5B4] hover:text-[#F4F1E9] hover:bg-[#202631]/50"
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Center: Live Mathematical Quality Badges */}
        {metrics && (
          <div className="hidden lg:flex items-center space-x-2.5 text-xs font-mono">
            <span className="px-2.5 py-1 rounded-lg bg-[#101217] border border-[#262C38] text-[#9BA5B4]">
              SSIM: <strong className="text-[#5CE1D2]">{metrics.ssim ?? "—"}</strong>
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-[#101217] border border-[#262C38] text-[#9BA5B4]">
              PSNR: <strong className="text-[#5CE1D2]">{metrics.psnr_db ? `${metrics.psnr_db} dB` : "—"}</strong>
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-[#101217] border border-[#262C38] text-[#9BA5B4]">
              LPIPS: <strong className="text-[#A9A5FF]">{metrics.lpips ?? "—"}</strong>
            </span>
            <span
              className={`px-2.5 py-1 rounded-lg border font-semibold ${
                metrics.epsilon_passed
                  ? "bg-[#5CE1D2]/10 text-[#5CE1D2] border-[#5CE1D2]/20"
                  : "bg-[#FF8C79]/10 text-[#FF8C79] border-[#FF8C79]/20"
              }`}
            >
              &epsilon;-Bound: {metrics.epsilon_passed ? "PASSED (≤8/255)" : "FAILED"}
            </span>
          </div>
        )}

        {/* Right: Zoom & Layout Controls */}
        <div className="flex items-center space-x-1.5 text-xs text-[#9BA5B4]">
          <div className="flex items-center space-x-1 bg-[#101217] border border-[#262C38] rounded-xl p-1">
            <button
              onClick={handleZoomOut}
              className="p-1.5 hover:text-[#F4F1E9] hover:bg-[#202631] rounded-lg transition"
              title="Zoom Out (0.8x)"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="px-2 font-mono text-[11px] text-[#F4F1E9]">
              {(scale * 100).toFixed(0)}%
            </span>
            <button
              onClick={handleZoomIn}
              className="p-1.5 hover:text-[#F4F1E9] hover:bg-[#202631] rounded-lg transition"
              title="Zoom In (1.25x)"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
          </div>

          <button
            onClick={handleReset}
            className="p-2 bg-[#101217] border border-[#262C38] hover:bg-[#202631] text-[#9BA5B4] hover:text-[#F4F1E9] rounded-xl transition"
            title="Reset View"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={toggleFullscreen}
            className="p-2 bg-[#101217] border border-[#262C38] hover:bg-[#202631] text-[#9BA5B4] hover:text-[#F4F1E9] rounded-xl transition"
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen Microscope"}
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Main Viewport Container */}
      <div
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onTouchMove={handleTouchMove}
        className="relative flex-1 overflow-hidden bg-[#101217] flex items-center justify-center cursor-grab active:cursor-grabbing"
      >
        <div
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`,
            transition: isDragging ? "none" : "transform 0.1s ease-out",
          }}
          className="relative max-w-full max-h-full flex items-center justify-center pointer-events-none"
        >
          {/* Mode 1: Split Slider */}
          {mode === "split" && (
            <div className="relative overflow-hidden select-none">
              {/* Underneath: Original Image */}
              <img
                src={origFull}
                alt="Original Artwork"
                className="max-h-[500px] w-auto object-contain block shadow-lg pointer-events-none"
              />

              {/* Top overlay: Cloaked Image clipped by sliderPos */}
              <div
                style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
                className="absolute inset-0 select-none pointer-events-none"
              >
                <img
                  src={cloakedFull}
                  alt="Cloaked Artwork"
                  className="max-h-[500px] w-auto object-contain block shadow-lg"
                />
              </div>

              {/* Interactive Divider Line with Dual Handle */}
              <div
                style={{ left: `${sliderPos}%` }}
                onMouseDown={(e) => {
                  e.stopPropagation();
                  setIsSliding(true);
                }}
                className="absolute top-0 bottom-0 w-0.5 bg-[#5CE1D2] pointer-events-auto cursor-ew-resize shadow-lg flex items-center justify-center"
              >
                <div className="w-7 h-7 -ml-3.5 rounded-full bg-[#181C24] border-2 border-[#5CE1D2] flex items-center justify-center text-[#5CE1D2] shadow-xl hover:scale-110 transition-transform">
                  <SplitSquareVertical className="w-3.5 h-3.5" />
                </div>
              </div>
            </div>
          )}

          {/* Mode 2: Original Asset Only */}
          {mode === "original" && (
            <img
              src={origFull}
              alt="Original Artwork Baseline"
              className="max-h-[500px] w-auto object-contain shadow-2xl"
            />
          )}

          {/* Mode 3: Cloaked Asset Only */}
          {mode === "cloaked" && (
            <img
              src={cloakedFull}
              alt="Adversarially Cloaked Artwork"
              className="max-h-[500px] w-auto object-contain shadow-2xl"
            />
          )}

          {/* Mode 4: Perturbation Difference Heatmap */}
          {mode === "xray" && (
            <div className="relative">
              {diffFull ? (
                <img
                  src={diffFull}
                  alt="Perturbation Heatmap"
                  style={{
                    filter:
                      colormap === "grayscale"
                        ? `grayscale(100%) contrast(${amplification / 10})`
                        : `contrast(${amplification / 12}) saturate(1.8)`,
                  }}
                  className="max-h-[500px] w-auto object-contain shadow-2xl transition-all"
                />
              ) : (
                <div className="p-12 text-center text-[#9BA5B4] bg-[#181C24] border border-[#262C38] rounded-2xl max-w-md">
                  <Activity className="w-8 h-8 text-[#FF8C79] mx-auto mb-2" />
                  <p className="font-semibold text-[#F4F1E9]">Difference Map Unavailable</p>
                  <p className="text-xs text-[#9BA5B4] mt-1">
                    Difference heatmap is generated automatically during job processing.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Floating Indicator Labels in Split Mode */}
        {mode === "split" && (
          <>
            <div className="absolute top-4 left-4 z-10 px-2.5 py-1 rounded-lg bg-[#181C24]/80 border border-[#262C38] text-[10px] font-mono text-[#5CE1D2] font-semibold pointer-events-none backdrop-blur-sm">
              PROTECTED (CLOAKED)
            </div>
            <div className="absolute top-4 right-4 z-10 px-2.5 py-1 rounded-lg bg-[#181C24]/80 border border-[#262C38] text-[10px] font-mono text-[#9BA5B4] font-semibold pointer-events-none backdrop-blur-sm">
              ORIGINAL ASSET
            </div>
          </>
        )}
      </div>

      {/* Bottom Microscope Telemetry & X-Ray Controls Bar */}
      <div className="z-20 px-4 py-2.5 bg-[#181C24]/90 border-t border-[#262C38] backdrop-blur-md flex flex-wrap items-center justify-between text-xs text-[#9BA5B4] font-mono gap-3">
        {/* Left: Cursor Position & Scientific Disclaimer */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1.5">
            <Compass className="w-3.5 h-3.5 text-[#5CE1D2]" />
            <span>Cursor: {cursorPos ? `${cursorPos.x}px, ${cursorPos.y}px` : "Outside Canvas"}</span>
          </div>
          <span className="text-[#262C38]">&bull;</span>
          <span className="text-[10px] text-[#9BA5B4]/80 hidden md:inline">
            Notice: Visible differences are non-destructive and strictly bounded (&epsilon; &le; 8/255).
          </span>
        </div>

        {/* Right: Difference Amplification & Colormap Controls (Active in X-Ray Mode) */}
        {mode === "xray" && (
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-1.5 bg-[#101217] px-2 py-1 rounded-lg border border-[#262C38]">
              <span className="text-[10px] text-[#9BA5B4]">Amplification:</span>
              <input
                type="range"
                min="4"
                max="64"
                step="4"
                value={amplification}
                onChange={(e) => setAmplification(Number(e.target.value))}
                className="w-20 accent-[#5CE1D2] cursor-pointer"
              />
              <span className="text-[10px] text-[#5CE1D2] font-bold w-6">{amplification}x</span>
            </div>

            <div className="flex items-center space-x-1 bg-[#101217] p-0.5 rounded-lg border border-[#262C38]">
              <button
                onClick={() => setColormap("heatmap")}
                className={`px-2 py-0.5 rounded text-[10px] font-semibold transition ${
                  colormap === "heatmap" ? "bg-[#202631] text-[#5CE1D2]" : "text-[#9BA5B4]"
                }`}
              >
                Heatmap
              </button>
              <button
                onClick={() => setColormap("grayscale")}
                className={`px-2 py-0.5 rounded text-[10px] font-semibold transition ${
                  colormap === "grayscale" ? "bg-[#202631] text-[#5CE1D2]" : "text-[#9BA5B4]"
                }`}
              >
                Grayscale
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
