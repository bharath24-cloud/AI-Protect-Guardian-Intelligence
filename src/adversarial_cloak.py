import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models import get_model, SurrogateVisionModel
from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD


class AdversarialCloakingEngine:
    """
    Optimization-based Adversarial Cloaking Engine for Digital Content Protection.
    
    Generates bounded adversarial perturbations under strict L-infinity constraints
    to disrupt vision model classification and deep embedding representations
    while preserving human perceptual fidelity.
    """
    def __init__(
        self,
        model: SurrogateVisionModel,
        device: Optional[torch.device] = None,
        mean: list = IMAGENET_MEAN,
        std: list = IMAGENET_STD
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes an un-normalized image tensor in [0, 1] for model forward pass."""
        return (x - self.mean) / self.std

    def unnormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Un-normalizes a tensor back to [0, 1] range."""
        return torch.clamp(x_norm * self.std + self.mean, 0.0, 1.0)

    def generate_cloak(
        self,
        image: Union[Image.Image, torch.Tensor, str, Path],
        epsilon: float = 8.0 / 255.0,
        iterations: int = 20,
        step_size: float = 2.0 / 255.0,
        objective: str = "classification_disruption",
        target_class: Optional[int] = None,
        random_init: bool = True,
        image_size: int = 224,
        native_resolution: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Synthesizes a bounded adversarial perturbation for the input image.
        Supports native resolution cloaking via differentiable spatial interpolation,
        preserving fine artwork brushstrokes and high-frequency sharpness without downsampling blur.
        """
        # 1. Prepare input tensor in [0, 1]
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            w, h = pil_img.size
            if native_resolution:
                # Cap max dimension to 1024 if image is huge to avoid OOM
                max_dim = max(w, h)
                if max_dim > 1024:
                    scale = 1024.0 / max_dim
                    pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                x_orig = transforms.ToTensor()(pil_img).unsqueeze(0).to(self.device)
            else:
                resize_tf = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()
                ])
                x_orig = resize_tf(pil_img).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            x_orig = image.clone().to(self.device)
            if x_orig.ndim == 3:
                x_orig = x_orig.unsqueeze(0)
            if x_orig.min() < -0.01 or x_orig.max() > 1.01:
                x_orig = self.unnormalize(x_orig)
            x_orig = torch.clamp(x_orig, 0.0, 1.0)
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        # 2. Extract baseline predictions & embeddings (using model input dimensions)
        with torch.no_grad():
            if x_orig.shape[-2:] != (image_size, image_size):
                x_orig_eval = F.interpolate(x_orig, size=(image_size, image_size), mode="bilinear", align_corners=False)
            else:
                x_orig_eval = x_orig
            x_orig_norm = self.normalize(x_orig_eval)
            orig_logits, orig_emb = self.model(x_orig_norm, return_embedding=True)
            orig_probs = F.softmax(orig_logits, dim=-1)
            orig_pred_class = orig_logits.argmax(dim=-1).item()
            orig_confidence = orig_probs[0, orig_pred_class].item()

        # 3. Initialize perturbation delta at native resolution
        if random_init:
            delta = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
            delta = torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig
        else:
            delta = torch.zeros_like(x_orig)

        delta.requires_grad = True

        # 4. Optimization Loop (Projected Gradient Descent)
        criterion_ce = nn.CrossEntropyLoss()

        for step in range(iterations):
            x_curr = torch.clamp(x_orig + delta, 0.0, 1.0)
            if x_curr.shape[-2:] != (image_size, image_size):
                x_curr_resized = F.interpolate(x_curr, size=(image_size, image_size), mode="bilinear", align_corners=False)
            else:
                x_curr_resized = x_curr

            x_curr_norm = self.normalize(x_curr_resized)
            curr_logits, curr_emb = self.model(x_curr_norm, return_embedding=True)

            if objective == "classification_disruption":
                target_tensor = torch.tensor([orig_pred_class], device=self.device)
                loss = criterion_ce(curr_logits, target_tensor)

            elif objective == "embedding_disruption":
                cos_sim = F.cosine_similarity(curr_emb, orig_emb, dim=-1)
                loss = -cos_sim

            elif objective == "targeted_disruption" and target_class is not None:
                target_tensor = torch.tensor([target_class], device=self.device)
                loss = -criterion_ce(curr_logits, target_tensor)

            else:
                target_tensor = torch.tensor([orig_pred_class], device=self.device)
                loss = criterion_ce(curr_logits, target_tensor)

            loss.backward()

            # Gradient ascent step
            with torch.no_grad():
                grad_sign = delta.grad.sign()
                delta += step_size * grad_sign

                # Project back to L-infinity ball [-eps, eps]
                delta.clamp_(-epsilon, epsilon)

                # Ensure pixel values stay within valid RGB range [0, 1]
                delta.copy_(torch.clamp(x_orig + delta, 0.0, 1.0) - x_orig)

            delta.grad.zero_()

        # 5. Final cloaked image at native resolution
        x_adv = torch.clamp(x_orig + delta.detach(), 0.0, 1.0)

        # 6. Evaluate protection metrics
        with torch.no_grad():
            if x_adv.shape[-2:] != (image_size, image_size):
                x_adv_eval = F.interpolate(x_adv, size=(image_size, image_size), mode="bilinear", align_corners=False)
            else:
                x_adv_eval = x_adv

            x_adv_norm = self.normalize(x_adv_eval)
            adv_logits, adv_emb = self.model(x_adv_norm, return_embedding=True)
            adv_probs = F.softmax(adv_logits, dim=-1)
            adv_pred_class = adv_logits.argmax(dim=-1).item()
            adv_confidence = adv_probs[0, adv_pred_class].item()
            orig_class_adv_conf = adv_probs[0, orig_pred_class].item()

            embedding_cosine_dist = (1.0 - F.cosine_similarity(adv_emb, orig_emb, dim=-1)).item()
            embedding_l2_dist = torch.norm(adv_emb - orig_emb, p=2, dim=-1).item()

        linf_measured = torch.max(torch.abs(x_adv - x_orig)).item()
        l2_measured = torch.norm(x_adv - x_orig, p=2).item()

        stats = {
            "objective": objective,
            "epsilon_configured": float(epsilon),
            "iterations": iterations,
            "step_size": float(step_size),
            "linf_norm_measured": float(linf_measured),
            "l2_norm_measured": float(l2_measured),
            "original_class_idx": int(orig_pred_class),
            "original_confidence": float(orig_confidence),
            "cloaked_pred_class_idx": int(adv_pred_class),
            "cloaked_top1_confidence": float(adv_confidence),
            "cloaked_orig_class_confidence": float(orig_class_adv_conf),
            "confidence_drop": float(orig_confidence - orig_class_adv_conf),
            "classification_disrupted": bool(orig_pred_class != adv_pred_class),
            "embedding_cosine_distance": float(embedding_cosine_dist),
            "embedding_l2_distance": float(embedding_l2_dist),
        }

        return x_adv, delta.detach(), stats

    def save_comparison_plot(
        self,
        x_orig: torch.Tensor,
        x_adv: torch.Tensor,
        delta: torch.Tensor,
        save_path: Path,
        title: str = "AI-Protect Adversarial Cloaking Assessment"
    ):
        """Generates a side-by-side visual comparison with an amplified difference map."""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        orig_np = x_orig.squeeze(0).permute(1, 2, 0).cpu().numpy()
        adv_np = x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Compute difference and amplify x10 for human visibility
        diff_np = np.clip(np.abs(adv_np - orig_np) * 10.0, 0.0, 1.0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(orig_np)
        axes[0].set_title("Original Visual Asset", fontweight="bold", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(adv_np)
        axes[1].set_title("Adversarially Cloaked Asset", fontweight="bold", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(diff_np)
        axes[2].set_title("Perturbation Difference (x10)", fontweight="bold", fontsize=12)
        axes[2].axis("off")

        plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[Plots] Saved cloaking visual comparison to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Protect Adversarial Cloak Generator")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--epsilon", type=float, default=8.0/255.0, help="L-infinity perturbation bound")
    parser.add_argument("--iterations", type=int, default=20, help="PGD optimization iterations")
    parser.add_argument("--step_size", type=float, default=2.0/255.0, help="Step size per iteration")
    parser.add_argument("--objective", type=str, default="classification_disruption",
                        choices=["classification_disruption", "embedding_disruption", "targeted_disruption"])
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model = get_model(ckpt.get("model_name", "resnet50"), num_classes=ckpt.get("num_classes", 20), pretrained=False, device=device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = get_model("resnet50", num_classes=10, pretrained=True, device=device)

    engine = AdversarialCloakingEngine(model=model, device=device)

    # Use sample from dataset if no image provided
    if args.image and Path(args.image).exists():
        img_input = args.image
    else:
        from src.dataset import WikiArtDataset
        ds = WikiArtDataset(fast_dev_mode=True, fast_dev_samples=10)
        img_tensor, _, _ = ds[0]
        img_input = img_tensor

    x_adv, delta, stats = engine.generate_cloak(
        image=img_input,
        epsilon=args.epsilon,
        iterations=args.iterations,
        step_size=args.step_size,
        objective=args.objective
    )

    print("\n" + "=" * 70)
    print("                 ADVERSARIAL CLOAKING RESULTS")
    print("=" * 70)
    print(f"Objective Target        : {stats['objective']}")
    print(f"Configured Epsilon (eps): {stats['epsilon_configured']:.5f} ({stats['epsilon_configured']*255:.1f}/255)")
    print(f"Measured L-inf Delta    : {stats['linf_norm_measured']:.5f} (Satisfies bound: {stats['linf_norm_measured'] <= stats['epsilon_configured'] + 1e-6})")
    print(f"Measured L2 Norm        : {stats['l2_norm_measured']:.4f}")
    print(f"Original Top-1 Conf     : {stats['original_confidence']*100:.2f}% (Class {stats['original_class_idx']})")
    print(f"Cloaked Original Conf   : {stats['cloaked_orig_class_confidence']*100:.2f}% (Drop: {stats['confidence_drop']*100:.2f}%)")
    print(f"Classification Disrupted: {stats['classification_disrupted']} (New Top-1: Class {stats['cloaked_pred_class_idx']})")
    print(f"Embedding Cosine Dist   : {stats['embedding_cosine_distance']:.4f}")
    print("=" * 70)

    # Save visual comparison
    if isinstance(img_input, torch.Tensor):
        x_orig_plot = img_input.unsqueeze(0) if img_input.ndim == 3 else img_input
    else:
        x_orig_plot = x_adv - delta
    engine.save_comparison_plot(x_orig_plot, x_adv, delta, Path("experiments/plots/adversarial_cloak_comparison.png"))
