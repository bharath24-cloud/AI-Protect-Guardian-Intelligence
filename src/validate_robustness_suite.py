import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime, timezone
import numpy as np
import torch
from PIL import Image

from src.models import SurrogateVisionModel
from src.classifier_eval import load_evaluation_model, get_model_predictions
from src.adversarial_cloak import AdversarialCloakingEngine
from src.robustness_eval import ImageTransformations, evaluate_asset_robustness
from src.evaluation_metrics import calculate_image_quality_metrics

def validate_robustness_transformations():
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 85)
    print("      AI-PROTECT: EMPIRICAL TRANSFORMATION ROBUSTNESS BENCHMARKS")
    print("=" * 85)

    # 1. Load Surrogate Model
    model, load_msg = load_evaluation_model("resnet50", device=device)
    print(f"\n[1/4] Loaded Evaluation Model: {load_msg}")

    # 2. Select Test Visual Asset
    test_img_dir = Path("backend/data/uploads")
    sample_images = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    test_path = sample_images[0] if sample_images else None
    if test_path and test_path.exists():
        pil_orig = Image.open(test_path).convert("RGB")
        print(f"[2/4] Testing with asset: {test_path.name} ({pil_orig.size[0]}x{pil_orig.size[1]} px)")
    else:
        pil_orig = Image.new("RGB", (512, 512), (180, 140, 100))
        print("[2/4] Testing with synthetic asset: 512x512 px")

    # 3. Generate Adversarial Cloak
    print("\n[3/4] Synthesizing Native Adversarial Perturbations (eps=8/255, iters=20)...")
    engine = AdversarialCloakingEngine(model=model, device=device)
    x_adv, delta, stats = engine.generate_cloak(
        image=pil_orig,
        epsilon=8.0 / 255.0,
        iterations=20,
        step_size=2.0 / 255.0,
        objective="classification_disruption",
        native_resolution=True
    )
    adv_np = (x_adv.squeeze(0).detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_adv = Image.fromarray(adv_np)

    # Baseline predictions
    orig_p = get_model_predictions(model, pil_orig, device=device)
    adv_p = get_model_predictions(model, pil_adv, device=device)
    orig_style = orig_p["top1_style"]
    orig_idx = orig_p["top1_class_idx"]
    adv_style = adv_p["top1_style"]

    print(f"      Baseline Original: {orig_style} ({orig_p['top1_confidence']*100:.1f}%)")
    print(f"      Baseline Cloaked : {adv_style} ({adv_p['top1_confidence']*100:.1f}%)")

    # 4. Comprehensive Suite of 10 Reproducible Transformations
    print("\n[4/4] Executing Robustness Transformation Suite...")
    transformations = [
        {"name": "JPEG Compression (Q=75)", "params": {"format": "JPEG", "quality": 75}, "fn": lambda im: ImageTransformations.jpeg_compression(im, 75)},
        {"name": "JPEG Compression (Q=50)", "params": {"format": "JPEG", "quality": 50}, "fn": lambda im: ImageTransformations.jpeg_compression(im, 50)},
        {"name": "JPEG Compression (Q=25)", "params": {"format": "JPEG", "quality": 25}, "fn": lambda im: ImageTransformations.jpeg_compression(im, 25)},
        {"name": "Resizing Down/Up (0.8x)", "params": {"scale": 0.8, "interpolation": "bilinear"}, "fn": lambda im: ImageTransformations.resize_restore(im, 0.8)},
        {"name": "Resizing Down/Up (0.5x)", "params": {"scale": 0.5, "interpolation": "bilinear"}, "fn": lambda im: ImageTransformations.resize_restore(im, 0.5)},
        {"name": "Center Crop (90%)", "params": {"crop_ratio": 0.9}, "fn": lambda im: ImageTransformations.center_crop(im, 0.9)},
        {"name": "Center Crop (80%)", "params": {"crop_ratio": 0.8}, "fn": lambda im: ImageTransformations.center_crop(im, 0.8)},
        {"name": "Gaussian Spatial Blur (r=1.0)", "params": {"radius": 1.0}, "fn": lambda im: ImageTransformations.gaussian_blur(im, 1.0)},
        {"name": "Gaussian Spatial Blur (r=2.0)", "params": {"radius": 2.0}, "fn": lambda im: ImageTransformations.gaussian_blur(im, 2.0)},
        {"name": "Brightness Adjustment (+15%)", "params": {"factor": 1.15}, "fn": lambda im: ImageTransformations.brightness(im, 1.15)},
        {"name": "Contrast Adjustment (+15%)", "params": {"factor": 1.15}, "fn": lambda im: ImageTransformations.contrast(im, 1.15)}
    ]

    robustness_records = []
    print("-" * 95)
    print(f"{'Transformation':<30} | {'Transformed Prediction':<22} | {'Conf':<6} | {'Disrupted':<10} | {'Status':<16}")
    print("-" * 95)

    for item in transformations:
        t_name = item["name"]
        t_params = item["params"]
        t_fn = item["fn"]

        try:
            # Transform separate copy of cloaked image
            pil_transformed = t_fn(pil_adv.copy())
            trans_pred = get_model_predictions(model, pil_transformed, device=device)

            trans_idx = trans_pred["top1_class_idx"]
            trans_style = trans_pred["top1_style"]
            trans_conf = trans_pred["top1_confidence"]

            # Protection survived if model does NOT revert back to original true class
            is_disrupted = (trans_idx != orig_idx)
            retained_rate = 100.0 if is_disrupted else 0.0
            status_desc = "Robust (Disrupted)" if is_disrupted else "Reverted to True Style"

            # Measure perceptual fidelity of transformed image relative to original uncloaked asset
            from skimage.metrics import structural_similarity as ssim_fn
            np_orig = np.array(pil_orig).astype(np.float32) / 255.0
            np_trans = np.array(pil_transformed).astype(np.float32) / 255.0
            ssim_score = round(float(ssim_fn(np_orig, np_trans, channel_axis=2, data_range=1.0)), 4)

            record = {
                "transformation_name": t_name,
                "parameters": t_params,
                "top1_style_after_transform": trans_style,
                "top1_confidence_after_transform": round(trans_conf, 4),
                "prediction_disrupted": is_disrupted,
                "retained_protection_rate": retained_rate,
                "status": status_desc,
                "ssim_relative_to_original": ssim_score,
                "original_baseline_style": orig_style,
                "cloaked_baseline_style": adv_style
            }
            robustness_records.append(record)
            print(f"{t_name:<30} | {trans_style:<22} | {trans_conf*100:4.1f}% | {'YES' if is_disrupted else 'NO':<10} | {status_desc:<16}")

        except Exception as e:
            print(f"{t_name:<30} | ERROR: {e}")
            robustness_records.append({
                "transformation_name": t_name,
                "parameters": t_params,
                "status": f"Evaluation Error: {str(e)}",
                "prediction_disrupted": False,
                "retained_protection_rate": 0.0
            })

    print("-" * 95)

    # Compute overall robustness statistics
    survived_count = sum(1 for r in robustness_records if r.get("prediction_disrupted", False))
    total_transforms = len(robustness_records)
    robustness_survival_rate = round((survived_count / max(1, total_transforms)) * 100.0, 1)

    summary_artifact = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_transformations_tested": total_transforms,
        "transformations_survived": survived_count,
        "empirical_robustness_retention_rate_percent": robustness_survival_rate,
        "scientific_conclusion": (
            f"Under empirical testing with epsilon=8/255, the adversarial perturbation survived {survived_count}/{total_transforms} "
            f"({robustness_survival_rate}%) of common digital image transformations. High-frequency filtering operations such as JPEG "
            "compression and blur tend to attenuate gradient-based perturbations, causing models to revert toward baseline visual semantics."
        ),
        "transformation_results": robustness_records
    }

    out_json = results_dir / "robustness_validation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_artifact, f, indent=2)
    print(f"\n[Artifact] Saved robustness validation report to: {out_json}")

    print("=" * 85)
    print(f"      TRANSFORMATION ROBUSTNESS VALIDATION COMPLETE (Survival Rate: {robustness_survival_rate}%)")
    print("=" * 85 + "\n")
    return summary_artifact

if __name__ == "__main__":
    validate_robustness_transformations()
