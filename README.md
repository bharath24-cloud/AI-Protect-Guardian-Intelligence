# AI-Protect: Adaptive Adversarial Cloaking Engine for Digital Content Protection

A complete, production-grade deep-learning and adversarial protection system designed to protect visual artwork from unauthorized AI scraping, vision embedding models, and unauthorized model training by generating mathematically bounded, imperceptible adversarial perturbations.

---

## 1. System Overview & Pipeline

```
# 1. System Overview & Pipeline

```text
IMAGE UPLOAD (Web UI / API)
        |
        v
[Image Validation & Preprocessing (Integrity, EXIF Stripping, Bounds)]
        |
        v
[Surrogate Vision Model Inference (WikiArt ResNet-50 / ViT / ConvNeXt / CLIP)]
        |
        v
[Adversarial Cloaking Engine (PGD, L∞ ≤ ε, Feature/Classification Disruption)]
        |
        +--------------------------------------------------+
        |                         |                        |
        v                         v                        v
[Perceptual Quality       [Multi-Model             [Transformation
 Evaluation]               Transferability Matrix]  Robustness Suite]
(SSIM, PSNR, LPIPS, L2,    (ViT, ConvNeXt,           (JPEG 95/75/50,
 L∞)                        ResNet, CLIP)             Resizing, Crop,
                                                     Blur, Brightness)
        |                         |                        |
        +-------------------------+------------------------+
                                  |
                                  v
[Download Protected Image & Comprehensive Protection Audit Report (PDF/JSON)]
```

---

## 2. Key Modules & Directory Structure

```
project/ (c:\AI Protect\)
 wikiart-all/
 data/
 test-00000-of-00009.parquet ... test-00008-of-00009.parquet
 models/
 checkpoints/         # Saved PyTorch surrogate models (best_model.pt)
 embeddings/          # Extracted feature vectors (test_embeddings.pt)
 src/
 dataset.py           # Parquet streaming dataset, metadata indexing, class filtering
 preprocessing.py     # Image transforms, normalization, artist-aware train/val/test splits
 models.py            # Surrogate vision architectures (ResNet-50, ConvNeXt, ViT, CLIP)
 train.py             # AdamW, Cosine scheduler, AMP mixed-precision, checkpointing
 evaluate.py          # Top-1/balanced accuracy, F1, precision, recall, confusion matrix
 embeddings.py        # get_embedding(img) interface, test set embeddings extraction & comparison
 adversarial_cloak.py # Bounded PGD cloaking engine (L_inf <= eps, feature/classification disruption)
 transferability.py   # Multi-model cross-transferability matrix (ResNet -> ViT, ConvNeXt, CLIP)
 robustness.py        # Transformation survival pipeline (JPEG 95/75, resize, crop, blur, brightness)
 metrics.py           # SSIM, PSNR, LPIPS, L_inf, L2 verification tables
 verify_pipeline.py   # Complete end-to-end pipeline verification check
 validation/          # Rigorous scientific audit scripts (1, 2, 3)
 experiments/
 results.csv          # Experiment outputs & benchmark metrics
 plots/               # Training curves, confusion matrices, difference maps, transferability heatmaps
 reports/             # Detailed evaluation logs & summaries
 backend/                 # FastAPI backend server
 frontend/                # Next.js 14 web application
```

---

## 3. Quick Start & Execution

### 1. Run Complete End-to-End Pipeline Verification
```powershell
.venv\Scripts\python.exe src/verify_pipeline.py
```

### 2. Run Comprehensive Scientific Validation Audits
```powershell
# Step 1: Full dataset scan (63,061 rows) & artist-aware split leakage check
.venv\Scripts\python.exe src/validation/1_dataset_and_split_audit.py

# Step 2: Surrogate training, checkpoint serialization & single-image classification audit
.venv\Scripts\python.exe src/validation/2_train_and_validate_models.py

# Step 3: Adversarial attack, transferability, robustness, metrics & epsilon ablation
.venv\Scripts\python.exe src/validation/3_adversarial_transfer_robustness_audit.py
```

### 3. Run Backend Unit & API Tests
```powershell
.venv\Scripts\pytest.exe backend/tests/ -v
```

### 4. Start the FastAPI Backend
```powershell
.venv\Scripts\python.exe backend/run.py
```

### 5. Start the Next.js Frontend
```powershell
cd frontend
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to access the Protect Studio dashboard.

---

## 4. Scientific Disclaimer & Transparency

> **Important**: Adversarial cloaking provides empirical, non-cryptographic protection by altering vision model activations and deep feature embeddings. Effectiveness varies across target model architectures, attack hyper-parameters, and subsequent image transformations (such as aggressive re-compression or spatial cropping). AI-Protect is designed for responsible content defense and transparent evaluation.
