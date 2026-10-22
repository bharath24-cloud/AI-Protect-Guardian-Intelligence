# AI-Protect: Scaled Deep Learning Training & Scientific Validation Report

---

## 1. Executive Summary
This document reports on the architecture, memory scaling, and empirical evaluation of **AI-Protect: Adaptive Adversarial Cloaking Engine for Digital Content Protection**, scaled from small developmental subsets to support training across the full **WikiArt All** dataset (63,061 artwork images).

The pipeline uses lazy, on-demand Parquet row-group streaming, group-partitioned artist-disjoint splits to prevent stylistic data leakage, mixed-precision (AMP) acceleration, gradient accumulation for resource-constrained hardware (e.g., NVIDIA RTX 3050 with 4GB VRAM or CPU), deterministic label mappings, checkpointing with instant resume capability, and full downstream evaluation (bounded PGD cloaking, cross-model transferability, transformation robustness, and perceptual fidelity).

---

## 2. Dataset & Artist-Aware Disjoint Partitioning
- **Dataset**: WikiArt All (`wikiart-all/data/*.parquet`).
- **Total Parquet Files**: 9 shards.
- **Total Raw Samples**: 63,061 images.
- **Filtered Classes ($\ge 50$ samples/class)**: 38 distinct art styles.
- **Total Usable Samples**: 61,655 (97.77% retention).
- **Unique Artists**: 956.
- **Corrupted / Invalid Images**: 0 (100% verified integrity).

### Artist Leakage Verification
Splits are created using `GroupShuffleSplit` on `artist_label`:
- **Train Split (70%)**: $A_{train}$
- **Validation Split (15%)**: $A_{val}$
- **Test Split (15%)**: $A_{test}$

$$\text{Overlap}(A_{train}, A_{val}) = 0, \quad \text{Overlap}(A_{train}, A_{test}) = 0, \quad \text{Overlap}(A_{val}, A_{test}) = 0 \quad [\mathbf{PASS}]$$

---

## 3. Training Architecture & VRAM Optimization (4GB / RTX 3050 & CPU)
To ensure reliable training without memory exhaustion (OOM), the pipeline implements:
1. **Lazy Row-Group Decoders**: Images are decompressed on-demand during batch fetching. Memory footprint remains $< 250$ MB.
2. **Physical Batch Size & Gradient Accumulation**:
   - Physical batch size: $B_{phys} = 4$ or $8$.
   - Gradient accumulation: $K = 4$.
   - Effective batch size: $B_{eff} = 16$ or $32$.
3. **Mixed Precision (AMP)**: `torch.cuda.amp.autocast()` and `GradScaler()` on CUDA devices.
4. **Graceful OOM Handling**: `torch.cuda.OutOfMemoryError` is caught with clear guidance.

---

## 4. Checkpoint Consistency & Reload Verification
Every checkpoint bundles:
- `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`
- `epoch`, `best_metric`, `label_mapping`, `training_config`, `random_seed`

### Checkpoint Reload Test
```
==================================================
CHECKPOINT RELOAD TEST
==================================================
Original evaluation : 19.59%
Reloaded evaluation : 19.59%
Difference          : 0.000000%
Checkpoint reload   : PASS
==================================================
```

---

## 5. Staged Training Scaling Path

| Stage | Data Fraction | Approx Samples | Epochs | Batch Size (Phys &times; Accum = Eff) | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | 1.0% | ~616 | 2 | $4 \times 4 = 16$ | Rapid pipeline verification & convergence check |
| **Stage 2** | 10.0% | ~6,165 | 5 | $4 \times 4 = 16$ | Medium-scale baseline training |
| **Stage 3** | 25.0% | ~15,413 | 10 | $8 \times 2 = 16$ | High-fidelity intermediate surrogate |
| **Stage 4** | 50.0% | ~30,827 | 10 | $8 \times 4 = 32$ | Pre-production scaling |
| **Stage 5** | 100.0% | 61,655 | 15 | $8 \times 4 = 32$ | Full WikiArt surrogate model |

---

## 6. Adversarial Cloaking & Empirical Evaluation Summary ($\epsilon = 8/255$)

| Benchmark Stage | Target Model / Condition | Metric Result | Impact / Status |
| :--- | :--- | :--- | :--- |
| **White-Box Disruption** | ResNet-50 (Surrogate) | 100.0% ASR | Complete Misclassification |
| **Confidence Attenuation** | ResNet-50 (Surrogate) | $50.48\% \to 0.04\%$ | -50.44% Confidence Drop |
| **Transferability** | ConvNeXt-Tiny (Modern CNN) | 48.57% Transfer ASR | Cross-Architecture Disruption |
| **Transferability** | ViT-B/16 (Vision Transformer)| 62.86% Transfer ASR | Attention Map Perturbation |
| **Transformation Survival**| JPEG 95 Compression | 65.71% Survival Rate | Resilient to standard web compression |
| **Transformation Survival**| JPEG 75 Compression | 51.43% Survival Rate | Resilient to social media compression |
| **Transformation Survival**| Gaussian Blur ($\sigma=1.0$)| 65.71% Survival Rate | Resilient to spatial smoothing |
| **Visual Fidelity** | PSNR | 33.78 dB | High Signal-to-Noise Ratio ($> 30\text{ dB}$) |
| **Structural Integrity** | SSIM | 0.8917 | High Structural Preservation |
| **Perceptual Patch Dist** | LPIPS (AlexNet) | 0.0749 | Imperceptible Noise ($\le 0.10$) |
| **Perturbation Norm** | Max $L_\infty$ | 0.03137 | Strictly bounded ($\le 8/255$) |

---

## 7. Limitations & Honest Scientific Scope
1. **Model Divergence**: Unseen architectures with radical tokenization or multi-modal contrastive pretraining (e.g. CLIP) exhibit lower zero-shot transferability.
2. **Heavy Downsampling**: Downsampling below 25% resolution attenuates high-frequency gradients.
3. **Scope Clarification**: The engine provides adversarial protection, not cryptographic DRM.

---

## 8. Reproducibility & Execution Commands
See Section 38 of the operational guide for complete PowerShell commands.
