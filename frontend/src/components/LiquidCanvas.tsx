"use client";

import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Sparkles, EyeOff, FastForward } from "lucide-react";

interface LiquidCanvasProps {
  active?: boolean;
  onSettled?: () => void;
  className?: string;
  showControls?: boolean;
}

export default function LiquidCanvas({
  active = true,
  onSettled,
  className = "",
  showControls = false,
}: LiquidCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [isSkipped, setIsSkipped] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
      setReducedMotion(mediaQuery.matches);
    }
  }, []);

  useEffect(() => {
    if (!containerRef.current || reducedMotion || isSkipped || !active) {
      if (onSettled && (reducedMotion || isSkipped)) {
        onSettled();
      }
      return;
    }

    const container = containerRef.current;
    const width = container.clientWidth || 400;
    const height = container.clientHeight || 300;

    // 1. Scene, Camera, Renderer
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.z = 24;

    const renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      powerPreference: "low-power",
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // 2. Particle Geometry
    const particleCount = 260;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const initialRadii = new Float32Array(particleCount);
    const targetRadii = new Float32Array(particleCount);
    const angles = new Float32Array(particleCount);
    const speeds = new Float32Array(particleCount);

    // AI-Protect Palette: Electric Teal (#5CE1D2), Soft Coral (#FF8C79), Muted Violet (#A9A5FF)
    const colorTeal = new THREE.Color(0x5ce1d2);
    const colorCoral = new THREE.Color(0xff8c79);
    const colorViolet = new THREE.Color(0xa9a5ff);

    for (let i = 0; i < particleCount; i++) {
      const r = 5 + Math.random() * 9;
      const angle = Math.random() * Math.PI * 2;
      const z = (Math.random() - 0.5) * 5;

      positions[i * 3] = Math.cos(angle) * r;
      positions[i * 3 + 1] = Math.sin(angle) * r;
      positions[i * 3 + 2] = z;

      initialRadii[i] = r;
      // Target radius for outline halo
      targetRadii[i] = 3.2 + (Math.random() - 0.5) * 0.4;
      angles[i] = angle;
      speeds[i] = 0.008 + Math.random() * 0.016;

      // Color distribution: 60% Electric Teal, 25% Muted Violet, 15% Soft Coral
      const rand = Math.random();
      const chosenColor = rand > 0.4 ? colorTeal : rand > 0.15 ? colorViolet : colorCoral;
      colors[i * 3] = chosenColor.r;
      colors[i * 3 + 1] = chosenColor.g;
      colors[i * 3 + 2] = chosenColor.b;
    }

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    // Circular particle texture
    const canvas = document.createElement("canvas");
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      const grad = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
      grad.addColorStop(0, "rgba(255, 255, 255, 1)");
      grad.addColorStop(0.4, "rgba(255, 255, 255, 0.7)");
      grad.addColorStop(1, "rgba(255, 255, 255, 0)");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 32, 32);
    }
    const texture = new THREE.CanvasTexture(canvas);

    const material = new THREE.PointsMaterial({
      size: 0.45,
      vertexColors: true,
      map: texture,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const particles = new THREE.Points(geometry, material);
    scene.add(particles);

    // Animation Loop with 4 Phases:
    // Phase 1 (0-1.5s): Particles gather inward towards target radii
    // Phase 2 (1.5-2.5s): Form outline halo
    // Phase 3 (2.5-3.5s): Settle gracefully
    // Phase 4 (>3.5s): Trigger onSettled
    let animationFrameId: number;
    const startTime = performance.now();
    let hasNotifiedSettled = false;

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      const elapsed = (performance.now() - startTime) / 1000;
      const positionAttr = geometry.attributes.position as THREE.BufferAttribute;
      const posArray = positionAttr.array as Float32Array;

      // Gathering progression factor [0 to 1]
      const gatherFactor = Math.min(1.0, elapsed / 2.2);
      const ease = 1 - Math.pow(1 - gatherFactor, 3); // Cubic ease out

      for (let i = 0; i < particleCount; i++) {
        angles[i] += speeds[i] * (1 - ease * 0.4);
        const currentR = initialRadii[i] + (targetRadii[i] - initialRadii[i]) * ease;

        posArray[i * 3] = Math.cos(angles[i]) * currentR;
        posArray[i * 3 + 1] = Math.sin(angles[i]) * currentR;
      }
      positionAttr.needsUpdate = true;

      // Slow orbital rotation
      particles.rotation.z += 0.0015;

      renderer.render(scene, camera);

      if (elapsed >= 3.0 && !hasNotifiedSettled) {
        hasNotifiedSettled = true;
        if (onSettled) onSettled();
      }
    };

    animate();

    const handleResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth || 400;
      const h = containerRef.current.clientHeight || 300;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationFrameId);
      geometry.dispose();
      material.dispose();
      texture.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [reducedMotion, isSkipped, active]);

  if (reducedMotion || isSkipped) {
    return (
      <div className={`flex items-center justify-center text-xs text-[#9BA5B4] ${className}`}>
        <span className="font-mono text-[10px] uppercase tracking-wider">Ambient Intro Settled</span>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      <div ref={containerRef} className="w-full h-full" />
      {showControls && (
        <button
          onClick={() => {
            setIsSkipped(true);
            if (onSettled) onSettled();
          }}
          className="absolute bottom-2 right-2 px-2 py-1 rounded bg-[#181C24]/80 hover:bg-[#202631] border border-[#262C38] text-[10px] font-mono text-[#9BA5B4] hover:text-[#F4F1E9] transition flex items-center gap-1 z-10"
          title="Skip decorative particle intro"
        >
          <FastForward className="w-3 h-3 text-[#5CE1D2]" />
          <span>Skip</span>
        </button>
      )}
    </div>
  );
}
