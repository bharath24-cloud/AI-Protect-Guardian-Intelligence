import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models import get_model, SurrogateVisionModel
from src.adversarial_cloak import AdversarialCloakingEngine
from src.dataset import WikiArtDataset
from src.preprocessing import create_artist_aware_splits


class TransferabilityBenchmark:
    """
    Evaluates empirical cross-architecture transferability of adversarial perturbations.
    """
    def __init__(
        self,
        model_names: List[str] = None,
        num_classes: int = 10,
        device: torch.device = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_names = model_names or ["resnet50", "vit_b_16", "convnext_tiny", "clip_vit_b32"]
        self.num_classes = num_classes

        print(f"[Transferability] Initializing model pool for cross-benchmarking: {self.model_names}")
        self.models: Dict[str, SurrogateVisionModel] = {}
        for name in self.model_names:
            try:
                m = get_model(name, num_classes=self.num_classes, pretrained=True, device=self.device)
                m.eval()
                self.models[name] = m
            except Exception as e:
                print(f"[Warning] Failed to initialize model '{name}': {e}")

    def evaluate_transferability(
        self,
        test_samples: list,
        surrogate_name: str = "resnet50",
        epsilon: float = 8.0 / 255.0,
        iterations: int = 20,
        step_size: float = 2.0 / 255.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Generates perturbations on surrogate_name, then evaluates disruption
        across all other models in the evaluation pool.
        """
        if surrogate_name not in self.models:
            raise ValueError(f"Surrogate model '{surrogate_name}' not available in model pool.")

        surrogate_model = self.models[surrogate_name]
        engine = AdversarialCloakingEngine(model=surrogate_model, device=self.device)

        results = {m_name: {
            "disruption_rate": 0.0,
            "avg_confidence_drop": 0.0,
            "avg_embedding_distance": 0.0,
            "total_evaluated": 0
        } for m_name in self.models.keys()}

        print(f"\n[Transferability] Generating perturbations with surrogate: {surrogate_name.upper()}...")

        total_samples = len(test_samples)
        for idx, (img_tensor, true_label, meta) in enumerate(test_samples):
            # 1. Generate adversarial perturbation on surrogate
            x_adv, delta, stats = engine.generate_cloak(
                image=img_tensor,
                epsilon=epsilon,
                iterations=iterations,
                step_size=step_size,
                objective="classification_disruption"
            )

            # 2. Evaluate across all models
            for eval_name, eval_model in self.models.items():
                with torch.no_grad():
                    # Baseline
                    orig_norm = engine.normalize(img_tensor.unsqueeze(0).to(self.device))
                    orig_logits, orig_emb = eval_model(orig_norm, return_embedding=True)
                    orig_prob = F.softmax(orig_logits, dim=-1)
                    orig_pred = orig_logits.argmax(dim=-1).item()
                    orig_conf = orig_prob[0, orig_pred].item()

                    # Protected
                    adv_norm = engine.normalize(x_adv)
                    adv_logits, adv_emb = eval_model(adv_norm, return_embedding=True)
                    adv_prob = F.softmax(adv_logits, dim=-1)
                    adv_pred = adv_logits.argmax(dim=-1).item()
                    adv_conf_orig = adv_prob[0, orig_pred].item()

                    # Metrics
                    is_disrupted = (orig_pred != adv_pred)
                    conf_drop = orig_conf - adv_conf_orig
                    cos_dist = (1.0 - F.cosine_similarity(adv_emb, orig_emb, dim=-1)).item()

                    res = results[eval_name]
                    if is_disrupted:
                        res["disruption_rate"] += 1
                    res["avg_confidence_drop"] += conf_drop
                    res["avg_embedding_distance"] += cos_dist
                    res["total_evaluated"] += 1

        # Normalize averages
        for m_name, res in results.items():
            cnt = max(1, res["total_evaluated"])
            res["disruption_rate"] = (res["disruption_rate"] / cnt) * 100.0
            res["avg_confidence_drop"] = (res["avg_confidence_drop"] / cnt) * 100.0
            res["avg_embedding_distance"] = res["avg_embedding_distance"] / cnt

        return results

    def build_full_matrix(
        self,
        test_samples: list,
        epsilon: float = 8.0 / 255.0
    ) -> pd.DataFrame:
        """
        Builds the full transferability matrix where each model serves as surrogate.
        """
        matrix_data = {}
        for surrogate in self.models.keys():
            res = self.evaluate_transferability(test_samples, surrogate_name=surrogate, epsilon=epsilon)
            matrix_data[surrogate] = {target: round(res[target]["disruption_rate"], 1) for target in self.models.keys()}

        matrix_df = pd.DataFrame(matrix_data).T  # Rows = Surrogate, Columns = Target
        return matrix_df

    def plot_transferability_matrix(self, matrix_df: pd.DataFrame, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 6))

        data = matrix_df.values
        labels = [col.replace("_", "-").upper() for col in matrix_df.columns]

        plt.imshow(data, cmap="Blues", interpolation="nearest", vmin=0, vmax=100)
        plt.colorbar(label="Disruption Transferability Rate (%)")

        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontweight="bold")
        plt.yticks(np.arange(len(labels)), labels, fontweight="bold")

        for i in range(len(labels)):
            for j in range(len(labels)):
                val = data[i, j]
                color = "white" if val > 50 else "black"
                plt.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight="bold")

        plt.title("Cross-Architecture Adversarial Transferability Matrix", fontsize=12, fontweight="bold", pad=15)
        plt.xlabel("Evaluation Target Model", fontweight="bold")
        plt.ylabel("Surrogate Optimization Model", fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[Plots] Saved transferability matrix heatmap to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Architecture Transferability Benchmark")
    parser.add_argument("--data_dir", type=str, default="wikiart-all")
    parser.add_argument("--samples", type=int, default=15, help="Number of test samples for evaluation")
    parser.add_argument("--epsilon", type=float, default=8.0/255.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    ds = WikiArtDataset(data_dir=args.data_dir, fast_dev_mode=True, fast_dev_samples=args.samples * 2)
    _, _, test_ds = create_artist_aware_splits(ds, artist_aware=True)

    test_samples = [test_ds[i] for i in range(min(args.samples, len(test_ds)))]

    benchmark = TransferabilityBenchmark(
        model_names=["resnet50", "convnext_tiny", "vit_b_16"],
        num_classes=ds.num_classes,
        device=device
    )

    matrix_df = benchmark.build_full_matrix(test_samples, epsilon=args.epsilon)

    print("\n" + "=" * 70)
    print("           EMPIRICAL TRANSFERABILITY MATRIX (Disruption %)")
    print("=" * 70)
    print(matrix_df.to_string())
    print("=" * 70)

    benchmark.plot_transferability_matrix(matrix_df, Path("experiments/plots/transferability_matrix.png"))
