import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.train import run_training


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: Path):
    """
    Generates a grouped bar chart comparing Baseline, Exp A, Exp B, and Exp C
    across Validation Accuracy, Test Accuracy, Balanced Accuracy, Macro F1, and Weighted F1.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    metrics = ["best_val_accuracy", "test_accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    metric_labels = ["Val Acc (%)", "Test Acc (%)", "Balanced Acc (%)", "Macro F1 (%)", "Weighted F1 (%)"]

    exp_names = comparison_df["experiment_id"].tolist()
    n_exp = len(exp_names)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_exp

    colors = ["#78909c", "#42a5f5", "#26a69a", "#ab47bc"]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    for i, exp_name in enumerate(exp_names):
        row = comparison_df[comparison_df["experiment_id"] == exp_name].iloc[0]
        vals = [row[m] for m in metrics]
        offset = (i - (n_exp - 1) / 2) * width
        rects = ax.bar(x + offset, vals, width, label=exp_name, color=colors[i % len(colors)], alpha=0.9, edgecolor="black", linewidth=0.8)
        
        # Add values on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Performance (%)", fontsize=12, fontweight="bold")
    ax.set_title("Generalization Benchmark: Baseline vs Improved Configurations (25% WikiArt)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()
    print(f"[Comparison Plot] Saved benchmark chart to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AI-Protect Generalization Benchmark Suite")
    parser.add_argument("--data_fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--split_dir", type=str, default="experiments/results/splits/split_25pct")
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path("experiments/results")
    res_dir.mkdir(parents=True, exist_ok=True)
    plt_dir = Path("experiments/plots")
    plt_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "id": "Baseline",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "label_smoothing": 0.0,
            "freeze_backbone": False,
            "unfreeze_epoch": 0,
            "augmentation": "basic",
            "class_weights": "none"
        },
        {
            "id": "Experiment_A",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "label_smoothing": 0.0,
            "freeze_backbone": False,
            "unfreeze_epoch": 0,
            "augmentation": "enhanced",
            "class_weights": "none"
        },
        {
            "id": "Experiment_B",
            "lr": 5e-5,
            "weight_decay": 1e-3,
            "label_smoothing": 0.1,
            "freeze_backbone": False,
            "unfreeze_epoch": 0,
            "augmentation": "enhanced",
            "class_weights": "none"
        },
        {
            "id": "Experiment_C",
            "lr": 5e-5,
            "weight_decay": 1e-3,
            "label_smoothing": 0.1,
            "freeze_backbone": True,
            "unfreeze_epoch": 3,
            "augmentation": "enhanced",
            "class_weights": "none"
        },
    ]

    all_summaries = []

    for exp in experiments:
        print("\n" + "#" * 80)
        print(f"  STARTING {exp['id'].upper()}")
        print(f"  LR: {exp['lr']} | WD: {exp['weight_decay']} | LS: {exp['label_smoothing']} | Freeze: {exp['freeze_backbone']} (Unfreeze: {exp['unfreeze_epoch']}) | Aug: {exp['augmentation']}")
        print("#" * 80 + "\n")

        _, summary = run_training(
            data_dir="wikiart-all",
            model_name="resnet50",
            target_field="style",
            data_fraction=args.data_fraction,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lr=exp["lr"],
            weight_decay=exp["weight_decay"],
            label_smoothing=exp["label_smoothing"],
            freeze_backbone=exp["freeze_backbone"],
            unfreeze_epoch=exp["unfreeze_epoch"],
            augmentation_level=exp["augmentation"],
            class_weights_strategy=exp["class_weights"],
            split_cache_dir=str(split_dir),
            experiment_id=exp["id"],
            early_stopping_patience=4
        )
        all_summaries.append(summary)

    # Save model_comparison.csv
    comp_df = pd.DataFrame(all_summaries)
    comp_csv_path = res_dir / "model_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\n[Comparison] Saved model comparison CSV to: {comp_csv_path}")

    # Generate model_comparison.png
    comp_plot_path = plt_dir / "model_comparison.png"
    plot_model_comparison(comp_df, comp_plot_path)

    print("\n" + "=" * 90)
    print("                    EXPERIMENT SUITE SUMMARY TABLE")
    print("=" * 90)
    print(comp_df[["experiment_id", "learning_rate", "weight_decay", "label_smoothing", "freeze_backbone", "best_epoch", "best_val_loss", "best_val_accuracy", "test_accuracy", "macro_f1", "balanced_accuracy"]].to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
