import argparse
import io
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms

from src.models import get_model, SurrogateVisionModel
from src.adversarial_cloak import AdversarialCloakingEngine
from src.dataset import WikiArtDataset
from src.preprocessing import create_artist_aware_splits


class TransformationPipeline:
    """
    Simulates real-world image processing transformations that protected assets
    encounter online, evaluating perturbation survival rates.
    """
    @staticmethod
    def apply_jpeg_compression(img_tensor: torch.Tensor, quality: int = 95) -> torch.Tensor:
        """Applies real PIL JPEG compression and decompression."""
        img_np = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        decomp_img = Image.open(buf).convert("RGB")

        to_tensor = transforms.ToTensor()
        return to_tensor(decomp_img).unsqueeze(0).to(img_tensor.device)

    @staticmethod
    def apply_resize(img_tensor: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
        """Downsamples and upsamples back to simulate social media thumbnailing."""
        _, _, h, w = img_tensor.shape
        down = F.interpolate(img_tensor, size=(int(h * scale), int(w * scale)), mode="bilinear", align_corners=False)
        up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
        return torch.clamp(up, 0.0, 1.0)

    @staticmethod
    def apply_center_crop(img_tensor: torch.Tensor, crop_ratio: float = 0.9) -> torch.Tensor:
        """Crops center region and resizes back to original dimensions."""
        _, _, h, w = img_tensor.shape
        crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        cropped = img_tensor[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
        resized = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
        return torch.clamp(resized, 0.0, 1.0)

    @staticmethod
    def apply_gaussian_blur(img_tensor: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
        """Applies Gaussian spatial filtering."""
        img_np = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img_np).filter(ImageFilter.GaussianBlur(radius=radius))
        to_tensor = transforms.ToTensor()
        return to_tensor(pil_img).unsqueeze(0).to(img_tensor.device)

    @staticmethod
    def apply_brightness(img_tensor: torch.Tensor, factor: float = 1.15) -> torch.Tensor:
        """Adjusts luminance/brightness."""
        img_np = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        pil_img = ImageEnhance.Brightness(Image.fromarray(img_np)).enhance(factor)
        to_tensor = transforms.ToTensor()
        return to_tensor(pil_img).unsqueeze(0).to(img_tensor.device)

    @staticmethod
    def apply_contrast(img_tensor: torch.Tensor, factor: float = 1.15) -> torch.Tensor:
        """Adjusts image contrast."""
        img_np = (img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        pil_img = ImageEnhance.Contrast(Image.fromarray(img_np)).enhance(factor)
        to_tensor = transforms.ToTensor()
        return to_tensor(pil_img).unsqueeze(0).to(img_tensor.device)


def evaluate_robustness(
    model: SurrogateVisionModel,
    test_samples: list,
    epsilon: float = 8.0 / 255.0,
    iterations: int = 20,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Evaluates adversarial protection survival rates across standard image transformations.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = AdversarialCloakingEngine(model=model, device=device)

    transform_suite = {
        "Original Protected": lambda x: x,
        "JPEG (Q=95)": lambda x: TransformationPipeline.apply_jpeg_compression(x, 95),
        "JPEG (Q=75)": lambda x: TransformationPipeline.apply_jpeg_compression(x, 75),
        "JPEG (Q=50)": lambda x: TransformationPipeline.apply_jpeg_compression(x, 50),
        "Resizing (50%)": lambda x: TransformationPipeline.apply_resize(x, 0.5),
        "Center Crop (90%)": lambda x: TransformationPipeline.apply_center_crop(x, 0.9),
        "Gaussian Blur": lambda x: TransformationPipeline.apply_gaussian_blur(x, 1.0),
        "Brightness (+15%)": lambda x: TransformationPipeline.apply_brightness(x, 1.15),
        "Contrast (+15%)": lambda x: TransformationPipeline.apply_contrast(x, 1.15),
    }

    results = {t_name: {
        "disrupted_count": 0,
        "total_count": 0,
        "avg_confidence_drop": 0.0,
        "avg_embedding_distance": 0.0
    } for t_name in transform_suite.keys()}

    print(f"\n[Robustness] Testing {len(test_samples)} assets against transformation suite...")

    for img_tensor, true_label, meta in test_samples:
        x_orig = img_tensor.unsqueeze(0).to(device) if img_tensor.ndim == 3 else img_tensor.to(device)

        # 1. Baseline prediction on original un-cloaked image
        with torch.no_grad():
            orig_norm = engine.normalize(x_orig)
            orig_logits, orig_emb = model(orig_norm, return_embedding=True)
            orig_prob = F.softmax(orig_logits, dim=-1)
            orig_pred = orig_logits.argmax(dim=-1).item()
            orig_conf = orig_prob[0, orig_pred].item()

        # 2. Synthesize cloaked image
        x_adv, delta, stats = engine.generate_cloak(
            image=x_orig,
            epsilon=epsilon,
            iterations=iterations,
            objective="classification_disruption"
        )

        # 3. Apply transformations and measure survival
        for t_name, t_fn in transform_suite.items():
            x_trans = t_fn(x_adv)

            with torch.no_grad():
                trans_norm = engine.normalize(x_trans)
                trans_logits, trans_emb = model(trans_norm, return_embedding=True)
                trans_prob = F.softmax(trans_logits, dim=-1)
                trans_pred = trans_logits.argmax(dim=-1).item()
                trans_conf_orig = trans_prob[0, orig_pred].item()

                is_disrupted = (trans_pred != orig_pred)
                conf_drop = orig_conf - trans_conf_orig
                cos_dist = (1.0 - F.cosine_similarity(trans_emb, orig_emb, dim=-1)).item()

                res = results[t_name]
                if is_disrupted:
                    res["disrupted_count"] += 1
                res["avg_confidence_drop"] += conf_drop
                res["avg_embedding_distance"] += cos_dist
                res["total_count"] += 1

    # Format summary table
    summary_list = []
    print("\n" + "=" * 75)
    print("                 TRANSFORMATION ROBUSTNESS BENCHMARKS")
    print("=" * 75)
    print(f"{'Transformation':<22} | {'Survival Rate (%)':<18} | {'Conf Drop (%)':<16} | {'Emb Dist':<10}")
    print("-" * 75)

    for t_name, res in results.items():
        total = max(1, res["total_count"])
        survival_rate = (res["disrupted_count"] / total) * 100.0
        avg_conf_drop = (res["avg_confidence_drop"] / total) * 100.0
        avg_emb_dist = res["avg_embedding_distance"] / total

        print(f"{t_name:<22} | {survival_rate:<18.1f} | {avg_conf_drop:<16.2f} | {avg_emb_dist:<10.4f}")
        summary_list.append({
            "transformation": t_name,
            "survival_rate": round(survival_rate, 2),
            "confidence_drop": round(avg_conf_drop, 2),
            "embedding_distance": round(avg_emb_dist, 4)
        })
    print("=" * 75)

    return summary_list


def plot_robustness_chart(summary_list: List[Dict[str, Any]], save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    names = [s["transformation"] for s in summary_list]
    rates = [s["survival_rate"] for s in summary_list]

    plt.figure(figsize=(10, 5))
    bars = plt.barh(names, rates, color="#0ea5e9", edgecolor="#0284c7")

    plt.xlim(0, 105)
    plt.xlabel("Adversarial Disruption Survival Rate (%)", fontweight="bold")
    plt.title("Empirical Robustness Under Image Transformations", fontsize=12, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)

    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Plots] Saved robustness degradation chart to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Robustness Benchmark Suite")
    parser.add_argument("--data_dir", type=str, default="wikiart-all")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=8.0/255.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = WikiArtDataset(data_dir=args.data_dir, fast_dev_mode=True, fast_dev_samples=args.samples * 2)
    _, _, test_ds = create_artist_aware_splits(ds, artist_aware=True)
    test_samples = [test_ds[i] for i in range(min(args.samples, len(test_ds)))]

    model = get_model("resnet50", num_classes=ds.num_classes, pretrained=True, device=device)

    summary = evaluate_robustness(model, test_samples, epsilon=args.epsilon, device=device)
    plot_robustness_chart(summary, Path("experiments/plots/robustness_curves.png"))
