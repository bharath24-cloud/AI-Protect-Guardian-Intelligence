from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from PIL import Image

from src.models import get_model
from src.classifier_eval import load_evaluation_model, compare_original_versus_cloaked
from src.adversarial_cloak import AdversarialCloakingEngine
from src.evaluation_metrics import calculate_image_quality_metrics

def validate_adversarial_cloaking():
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("      AI-PROTECT: ADVERSARIAL CLOAKING ENGINE VALIDATION")
    print("=" * 80)

    # 1. Load Surrogate Model with trained checkpoint
    model_name = "resnet50"
    surrogate_model, load_msg = load_evaluation_model(model_name, device=device)
    print(f"\n[1/5] Loaded Surrogate Model ({model_name}): {load_msg}")

    # 2. Prepare high-resolution test artwork image
    # Use an actual artwork from uploads or synthetic artwork
    test_img_dir = Path("backend/data/uploads")
    sample_images = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    if sample_images:
        test_path = sample_images[0]
        pil_orig = Image.open(test_path).convert("RGB")
        print(f"[2/5] Using test image from disk: {test_path.name} ({pil_orig.size[0]}x{pil_orig.size[1]} px)")
    else:
        # Create a realistic test image with geometric patterns and gradients
        w, h = 512, 512
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                arr[i, j, 0] = int(255 * (i / h))
                arr[i, j, 1] = int(255 * (j / w))
                arr[i, j, 2] = int(128 + 127 * np.sin(i / 20.0) * np.cos(j / 20.0))
        pil_orig = Image.fromarray(arr)
        print(f"[2/5] Synthesized test artwork: {w}x{h} px")

    # Record original image checksum and immutability hash
    orig_bytes = pil_orig.tobytes()
    orig_sha256 = hashlib.sha256(orig_bytes).hexdigest()

    # 3. Initialize Engine and run cloaking
    engine = AdversarialCloakingEngine(model=surrogate_model, device=device)
    epsilon = 8.0 / 255.0
    iterations = 20
    step_size = 2.0 / 255.0
    objective = "classification_disruption"

    print(f"\n[3/5] Executing Native-Resolution PGD Optimization...")
    print(f"      Epsilon Budget : {epsilon:.5f} ({epsilon*255:.1f}/255)")
    print(f"      Iterations     : {iterations}")
    print(f"      Step Size      : {step_size:.5f}")
    print(f"      Loss Objective : {objective}")

    t_start = time.time()
    x_adv, delta, stats = engine.generate_cloak(
        image=pil_orig,
        epsilon=epsilon,
        iterations=iterations,
        step_size=step_size,
        objective=objective,
        random_init=True,
        native_resolution=True
    )
    runtime = time.time() - t_start

    # Verify Original Image Immutability
    post_orig_bytes = pil_orig.tobytes()
    post_orig_sha256 = hashlib.sha256(post_orig_bytes).hexdigest()
    assert orig_sha256 == post_orig_sha256, "CRITICAL ERROR: Original image was mutated during cloaking!"
    print(f"\n[4/5] Verification Checks:")
    print(f"      [PASS] Original Image Immutability Verified (SHA256: {orig_sha256[:16]}...)")

    # Extract cloaked PIL image
    adv_tensor = x_adv.squeeze(0).detach().cpu()
    adv_np = (adv_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil_adv = Image.fromarray(adv_np)
    cloaked_sha256 = hashlib.sha256(pil_adv.tobytes()).hexdigest()

    # Check non-trivial perturbation
    delta_np = delta.squeeze(0).detach().cpu().numpy()
    max_delta = float(np.max(np.abs(delta_np)))
    mean_delta = float(np.mean(np.abs(delta_np)))
    assert max_delta > 0.001, "CRITICAL ERROR: Cloaking produced zero or trivial perturbation!"
    print(f"      [PASS] Non-trivial perturbation generated: max=|{max_delta:.5f}|, mean=|{mean_delta:.5f}|")

    # Check L-infinity constraint strictly
    linf_measured = stats["linf_norm_measured"]
    assert linf_measured <= epsilon + 1e-5, f"L-infinity constraint violated: {linf_measured} > {epsilon}"
    print(f"      [PASS] L-infinity constraint strictly bounded: {linf_measured:.5f} <= {epsilon:.5f}")

    # Check valid pixel range [0, 1]
    assert float(adv_tensor.min()) >= 0.0 and float(adv_tensor.max()) <= 1.0, "Cloaked pixel values outside [0, 1] range!"
    print(f"      [PASS] Pixel value range valid: [{float(adv_tensor.min()):.4f}, {float(adv_tensor.max()):.4f}]")

    # Check dimensions match original
    assert pil_adv.size == pil_orig.size, f"Dimensions mismatch: {pil_adv.size} vs {pil_orig.size}"
    print(f"      [PASS] Output dimensions preserved at native resolution: {pil_adv.size[0]}x{pil_adv.size[1]} px")

    # 4. Measure Quality Metrics
    print(f"\n[5/5] Computing Quality & Transferability Benchmarks...")
    qm = calculate_image_quality_metrics(pil_orig, pil_adv, epsilon_bound=epsilon, device=device)
    print(f"      SSIM : {qm['ssim']}")
    print(f"      PSNR : {qm['psnr_db']} dB")
    print(f"      LPIPS: {qm['lpips']} (Status: {qm['lpips_status']})")
    print(f"      L-inf: {qm['linf_norm']:.5f}")
    print(f"      L2   : {qm['l2_norm']:.4f}")

    # 5. Measure Classifier Disruption
    cmp = compare_original_versus_cloaked(pil_orig, pil_adv, surrogate_model, device=device)
    orig_p = cmp["original_prediction"]
    adv_p = cmp["cloaked_prediction"]
    print(f"\n      Original Prediction: {orig_p['style_name']} ({orig_p['confidence_percent']}%)")
    print(f"      Cloaked Prediction : {adv_p['style_name']} ({adv_p['confidence_percent']}%)")
    print(f"      Top-1 Disrupted    : {cmp['top1_prediction_changed']}")
    print(f"      Confidence Drop    : {cmp['confidence_drop_percent']}%")
    print(f"      Embedding Cos Dist : {cmp['embedding_cosine_distance']}")

    # Save comprehensive validation record
    record = {
        "validation_id": "cloak_validation_001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "Projected Gradient Descent (PGD)",
        "optimization_objective": objective,
        "target_model": model_name,
        "checkpoint_status": load_msg,
        "native_resolution": True,
        "input_dimensions": list(pil_orig.size),
        "runtime_seconds": round(runtime, 3),
        "random_init": True,
        "parameters": {
            "epsilon": epsilon,
            "epsilon_255": "8/255",
            "iterations": iterations,
            "step_size": step_size,
            "step_size_255": "2/255"
        },
        "checksums": {
            "original_sha256": orig_sha256,
            "cloaked_sha256": cloaked_sha256,
            "original_immutable": (orig_sha256 == post_orig_bytes)
        },
        "perturbation_statistics": {
            "max_delta_measured": max_delta,
            "mean_delta_measured": mean_delta,
            "linf_norm": qm["linf_norm"],
            "l2_norm": qm["l2_norm"],
            "rmse": qm["rmse"],
            "mae": qm["mae"],
            "epsilon_constraint_passed": qm["epsilon_passed"]
        },
        "perceptual_quality": {
            "ssim": qm["ssim"],
            "psnr_db": qm["psnr_db"],
            "lpips": qm["lpips"],
            "lpips_status": qm["lpips_status"]
        },
        "classification_disruption": {
            "original_style": orig_p["style_name"],
            "original_confidence": orig_p["confidence"],
            "cloaked_style": adv_p["style_name"],
            "cloaked_confidence": adv_p["confidence"],
            "cloaked_confidence_in_orig_class": adv_p["cloaked_confidence_in_orig_class"],
            "confidence_drop": cmp["confidence_drop"],
            "top1_prediction_changed": cmp["top1_prediction_changed"],
            "embedding_cosine_distance": cmp["embedding_cosine_distance"],
            "embedding_l2_distance": cmp["embedding_l2_distance"]
        },
        "validation_status": "PASS"
    }

    out_file = results_dir / "adversarial_cloaking_validation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"\n[Artifact] Saved validation record to: {out_file}")

    print("=" * 80)
    print("      ADVERSARIAL CLOAKING VALIDATION COMPLETE: ALL PASS")
    print("=" * 80 + "\n")
    return record

if __name__ == "__main__":
    validate_adversarial_cloaking()
