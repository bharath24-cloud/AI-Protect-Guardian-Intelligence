"use client";

import React, { useState, useRef } from "react";
import { UploadCloud, Image as ImageIcon, CheckCircle2, AlertCircle, Loader2, X } from "lucide-react";
import { uploadImage, ImageUploadResponse, BACKEND_ROOT_URL } from "@/lib/api";

interface ImageUploaderProps {
  onImageUploaded: (imageData: ImageUploadResponse) => void;
  selectedImage: ImageUploadResponse | null;
  onClear: () => void;
}

export default function ImageUploader({
  onImageUploaded,
  selectedImage,
  onClear,
}: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setError(null);
    if (!file) return;

    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type)) {
      setError(`Unsupported file type (${file.type}). Please upload a JPEG, PNG, or WebP image.`);
      return;
    }

    if (file.size > 15 * 1024 * 1024) {
      setError("File size exceeds the 15MB limit.");
      return;
    }

    try {
      setUploading(true);
      const res = await uploadImage(file);
      onImageUploaded(res);
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || "Failed to upload image.";
      setError(msg);
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  if (selectedImage) {
    return (
      <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 relative overflow-hidden backdrop-blur-sm">
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-slate-800">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <CheckCircle2 className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-semibold text-white text-sm sm:text-base">
                Image Validated & Ready
              </h3>
              <p className="text-xs text-slate-400">{selectedImage.original_filename}</p>
            </div>
          </div>
          <button
            onClick={onClear}
            className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors"
            title="Remove image"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-slate-950 flex items-center justify-center max-h-[300px]">
            <img
              src={`${BACKEND_ROOT_URL}${selectedImage.preview_url}`}
              alt="Uploaded source"
              className="max-h-[300px] w-auto object-contain rounded-lg"
            />
          </div>

          <div className="space-y-3 text-xs sm:text-sm">
            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-slate-400 block text-[11px] uppercase tracking-wider">Dimensions</span>
              <span className="text-slate-200 font-mono font-medium">
                {selectedImage.width} &times; {selectedImage.height} px
              </span>
            </div>

            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-slate-400 block text-[11px] uppercase tracking-wider">File Size & MIME</span>
              <span className="text-slate-200 font-mono font-medium">
                {(selectedImage.file_size_bytes / 1024).toFixed(1)} KB &bull; {selectedImage.mime_type}
              </span>
            </div>

            <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
              <span className="text-slate-400 block text-[11px] uppercase tracking-wider">SHA-256 Digest</span>
              <span className="text-sky-400 font-mono text-[11px] break-all">
                {selectedImage.sha256_hash}
              </span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <input
        type="file"
        ref={fileInputRef}
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
      />

      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-2xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-200 backdrop-blur-sm ${
          isDragging
            ? "border-sky-400 bg-sky-500/10 scale-[1.01]"
            : "border-slate-700/80 hover:border-sky-500/60 bg-slate-900/40 hover:bg-slate-900/60"
        }`}
      >
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="w-16 h-16 rounded-2xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center text-sky-400">
            {uploading ? (
              <Loader2 className="w-8 h-8 animate-spin" />
            ) : (
              <UploadCloud className="w-8 h-8" />
            )}
          </div>

          <div className="space-y-1">
            <h4 className="text-base sm:text-lg font-semibold text-white">
              {uploading ? "Analyzing & Validating Image..." : "Upload Image to Protect"}
            </h4>
            <p className="text-xs sm:text-sm text-slate-400">
              Drag & drop your artwork or photo, or click to browse
            </p>
          </div>

          <div className="flex items-center space-x-2 text-[11px] text-slate-500">
            <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">JPEG</span>
            <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">PNG</span>
            <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">WEBP</span>
            <span>&bull; Up to 15MB</span>
          </div>
        </div>
      </div>

      {error && (
        <div className="mt-4 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 flex items-center space-x-3 text-sm">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
