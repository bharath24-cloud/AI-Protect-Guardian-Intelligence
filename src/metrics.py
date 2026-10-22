import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

# Optional LPIPS perceptual distance model
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


class PerceptualQualityEvaluator:
    """
    Evaluates empirical mathematical and perceptual fidelity metrics
    between original visual assets and their adversarially cloaked counterparts.
    """
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips_model = None

        if LPIPS_AVAILABLE:
            try:
                self.lpips_model = lpips.LPIPS(net="alex", verbose=False).to(self.device)
                self.lpips_model.eval()
            except Exception as e:
                print(f"[Warning] Could not initialize LPIPS network: {e}")

    def _to_unit_tensor_and_np(self, img: Union[torch.Tensor, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, np.ndarray]:
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")
        if isinstance(img, Image.Image):
            np_arr = np.array(img).astype(np.float32) / 255.0
            t = torch.from_numpy(np_arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            return t, np_arr
        if isinstance(img, torch.Tensor):
            t = img.clone().to(self.device)
            if t.ndim == 3:
                t = t.unsqueeze(0)
            # Unnormalize if pre-normalized
            if t.min() < -0.01 or t.max() > 1.01:
                mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                t = torch.clamp(t * std + mean, 0.0, 1.0)
            t = torch.clamp(t, 0.0, 1.0)
            np_arr = t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            return t, np_arr
        if isinstance(img, np.ndarray):
            np_arr = img.astype(np.float32)
            if np_arr.max() > 1.5:
                np_arr = np_arr / 255.0
            t = torch.from_numpy(np_arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            return t, np_arr
        raise TypeError(f"Unsupported image input type: {type(img)}")

    def evaluate_pair(
        self,
        img_orig: Union[torch.Tensor, np.ndarray, Image.Image],
        img_adv: Union[torch.Tensor, np.ndarray, Image.Image]
    ) -> Dict[str, float]:
        """
        Computes PSNR, SSIM, LPIPS, L-inf, and L2 norms between original and protected images.
        """
        t_orig, np_orig = self._to_unit_tensor_and_np(img_orig)
        t_adv, np_adv = self._to_unit_tensor_and_np(img_adv)

        # 1. PSNR
        psnr_val = float(psnr_fn(np_orig, np_adv, data_range=1.0))

        # 2. SSIM (multichannel RGB)
        ssim_val = float(ssim_fn(
            np_orig, np_adv,
            data_range=1.0,
            channel_axis=2 if np_orig.ndim == 3 else None
        ))

        # 3. L-infinity and L2 perturbation norms
        diff = np_adv - np_orig
        linf_val = float(np.max(np.abs(diff)))
        l2_val = float(np.linalg.norm(diff))

        # 4. LPIPS perceptual distance
        lpips_val = None
        if self.lpips_model is not None:
            try:
                # LPIPS expects tensors scaled in [-1, 1]
                t1 = (t_orig * 2.0) - 1.0
                t2 = (t_adv * 2.0) - 1.0
                with torch.no_grad():
                    dist = self.lpips_model(t1, t2)
                    lpips_val = float(dist.item())
            except Exception as e:
                pass

        return {
            "psnr_db": round(psnr_val, 2),
            "ssim": round(ssim_val, 4),
            "lpips": round(lpips_val, 4) if lpips_val is not None else -1.0,
            "linf_norm": round(linf_val, 5),
            "l2_norm": round(l2_val, 4),
        }


def build_evaluation_table(
    evaluator: PerceptualQualityEvaluator,
    samples: list,
    engine,
    epsilons: list = [4.0/255.0, 8.0/255.0, 16.0/255.0]
) -> pd.DataFrame:
    records = []
    print(f"\n[Metrics] Computing comprehensive perceptual & disruption benchmarks...")

    for eps in epsilons:
        for idx, (img_tensor, true_label, meta) in enumerate(samples):
            x_adv, delta, stats = engine.generate_cloak(
                image=img_tensor,
                epsilon=eps,
                iterations=20,
                objective="classification_disruption"
            )

            q_metrics = evaluator.evaluate_pair(img_tensor, x_adv)

            surr_acc = 0.0 if stats["classification_disrupted"] else 100.0
            target_acc = round(stats["cloaked_orig_class_confidence"] * 100.0, 1)

            records.append({
                "Image": meta.get("title", f"Image_{idx}")[:25],
                "Epsilon": f"{eps*255:.0f}/255 ({eps:.4f})",
                "PSNR (dB)": q_metrics["psnr_db"],
                "SSIM": q_metrics["ssim"],
                "LPIPS": q_metrics["lpips"] if q_metrics["lpips"] >= 0 else "N/A",
                "L2": q_metrics["l2_norm"],
                "L_inf": q_metrics["linf_norm"],
                "Surrogate Acc (%)": surr_acc,
                "Orig Conf (%)": target_acc
            })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    from src.dataset import WikiArtDataset
    from src.preprocessing import create_artist_aware_splits
    from src.models import get_model
    from src.adversarial_cloak import AdversarialCloakingEngine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = PerceptualQualityEvaluator(device=device)

    ds = WikiArtDataset(fast_dev_mode=True, fast_dev_samples=20)
    _, _, test_ds = create_artist_aware_splits(ds, artist_aware=True)
    sample_list = [test_ds[i] for i in range(min(5, len(test_ds)))]

    model = get_model("resnet50", num_classes=ds.num_classes, pretrained=True, device=device)
    engine = AdversarialCloakingEngine(model=model, device=device)

    table_df = build_evaluation_table(evaluator, sample_list, engine, epsilons=[4.0/255.0, 8.0/255.0])

    print("\n" + "=" * 95)
    print("                 PERCEPTUAL QUALITY & PROTECTION MATRIX")
    print("=" * 95)
    print(table_df.to_string(index=False))
    print("=" * 95)
