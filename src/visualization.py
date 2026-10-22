import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_training_curves(history, output_dir="experiments/plots"):
    """
    Plots training and validation loss and accuracy curves individually and combined.
    history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'epochs'
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    epochs = history.get("epochs", list(range(1, len(history["train_loss"]) + 1)))

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Training Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], marker="o", color="#d32f2f", label="Train Loss", linewidth=2)
    ax.set_title("Training Loss vs Epochs", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path / "training_loss.png", dpi=300)
    plt.close()

    # 2. Validation Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["val_loss"], marker="s", color="#1976d2", label="Val Loss", linewidth=2)
    ax.set_title("Validation Loss vs Epochs", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path / "validation_loss.png", dpi=300)
    plt.close()

    # 3. Training Accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [a * 100 if a <= 1.0 else a for a in history["train_acc"]], marker="o", color="#388e3c", label="Train Acc", linewidth=2)
    ax.set_title("Training Accuracy vs Epochs", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path / "training_accuracy.png", dpi=300)
    plt.close()

    # 4. Validation Accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [a * 100 if a <= 1.0 else a for a in history["val_acc"]], marker="s", color="#f57c00", label="Val Acc", linewidth=2)
    ax.set_title("Validation Accuracy vs Epochs", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path / "validation_accuracy.png", dpi=300)
    plt.close()

    # 5. Combined Dual-Axis Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history["train_loss"], "o-", color="#d32f2f", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", color="#1976d2", label="Val Loss", linewidth=2)
    ax1.set_title("Convergence: Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    train_acc_pct = [a * 100 if a <= 1.0 else a for a in history["train_acc"]]
    val_acc_pct = [a * 100 if a <= 1.0 else a for a in history["val_acc"]]
    ax2.plot(epochs, train_acc_pct, "o-", color="#388e3c", label="Train Acc", linewidth=2)
    ax2.plot(epochs, val_acc_pct, "s--", color="#f57c00", label="Val Acc", linewidth=2)
    ax2.set_title("Generalization: Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path / "training_curves.png", dpi=300)
    plt.close()
    print(f"Training curve plots saved in {out_path}")

def plot_confusion_matrix(cm, class_names, output_path="experiments/plots/confusion_matrix.png"):
    """
    Plots normalized confusion matrix with high visual clarity.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={"label": "Normalized Recall"}, ax=ax)
    ax.set_title("Normalized Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Style", fontsize=12)
    ax.set_ylabel("True Style", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {out_path}")

def plot_adversarial_comparison(orig_img_np, cloaked_img_np, orig_label, orig_conf, 
                                cloaked_label, cloaked_conf, epsilon=8/255, 
                                output_path="experiments/plots/original_vs_cloaked_comparison.png"):
    """
    Plots 3-panel comparison: Original Image, Amplified Perturbation (x10), Cloaked Image.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    diff = np.abs(cloaked_img_np - orig_img_np)
    diff_amp = np.clip(diff * 10.0, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig_img_np)
    axes[0].set_title(f"Original Artwork\nPred: {orig_label} ({orig_conf:.1%})", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(diff_amp)
    axes[1].set_title(f"Amplified Perturbation (x10)\nBound: ε = {epsilon:.4f} (8/255)", fontsize=11, fontweight="bold", color="#d32f2f")
    axes[1].axis("off")

    axes[2].imshow(cloaked_img_np)
    axes[2].set_title(f"Cloaked Artwork (AI-Protect)\nPred: {cloaked_label} ({cloaked_conf:.1%})", fontsize=11, fontweight="bold", color="#1976d2")
    axes[2].axis("off")

    plt.suptitle("AI-Protect: Adaptive Adversarial Perturbation Disruption", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.parent / "adversarial_cloak_comparison.png", dpi=300)
    plt.close()
    print(f"Adversarial comparison saved to {out_path}")

def plot_epsilon_ablation(df_ablation, output_dir="experiments/plots"):
    """
    Plots Epsilon Ablation Tradeoff curves: ASR, PSNR, SSIM, LPIPS vs Epsilon.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epsilons = df_ablation["epsilon"]
    asr = df_ablation["attack_success_rate"]
    psnr = df_ablation["avg_psnr"]
    ssim = df_ablation["avg_ssim"]
    lpips = df_ablation["avg_lpips"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Epsilon vs ASR
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epsilons, asr * 100, marker="o", color="#d32f2f", linewidth=2.2)
    ax.set_title("Perturbation Budget vs Attack Success Rate (ASR)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Perturbation Budget (ε)", fontsize=11)
    ax.set_ylabel("ASR (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / "epsilon_vs_asr.png", dpi=300)
    plt.close()

    # 2. Epsilon vs PSNR
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epsilons, psnr, marker="s", color="#1976d2", linewidth=2.2)
    ax.set_title("Perturbation Budget vs PSNR (dB)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Perturbation Budget (ε)", fontsize=11)
    ax.set_ylabel("PSNR (dB)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / "epsilon_vs_psnr.png", dpi=300)
    plt.close()

    # 3. Epsilon vs SSIM
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epsilons, ssim, marker="^", color="#388e3c", linewidth=2.2)
    ax.set_title("Perturbation Budget vs Structural Similarity (SSIM)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Perturbation Budget (ε)", fontsize=11)
    ax.set_ylabel("SSIM Index", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / "epsilon_vs_ssim.png", dpi=300)
    plt.close()

    # 4. Epsilon vs LPIPS
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epsilons, lpips, marker="D", color="#7b1fa2", linewidth=2.2)
    ax.set_title("Perturbation Budget vs Perceptual Distance (LPIPS)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Perturbation Budget (ε)", fontsize=11)
    ax.set_ylabel("LPIPS Distance (Lower is closer)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / "epsilon_vs_lpips.png", dpi=300)
    plt.close()

    # 5. Combined Multi-Metric Ablation Figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
    ax1.plot(epsilons, asr * 100, "o-", color="#d32f2f", linewidth=2)
    ax1.set_title("Attack Success Rate (%)", fontweight="bold")
    ax1.set_xlabel("ε")
    ax1.set_ylabel("ASR (%)")
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2.plot(epsilons, psnr, "s-", color="#1976d2", linewidth=2)
    ax2.set_title("Peak Signal-to-Noise Ratio (dB)", fontweight="bold")
    ax2.set_xlabel("ε")
    ax2.set_ylabel("PSNR (dB)")
    ax2.grid(True, linestyle="--", alpha=0.7)

    ax3.plot(epsilons, ssim, "^-", color="#388e3c", linewidth=2)
    ax3.set_title("Structural Similarity Index (SSIM)", fontweight="bold")
    ax3.set_xlabel("ε")
    ax3.set_ylabel("SSIM")
    ax3.grid(True, linestyle="--", alpha=0.7)

    ax4.plot(epsilons, lpips, "D-", color="#7b1fa2", linewidth=2)
    ax4.set_title("Perceptual Distance (LPIPS)", fontweight="bold")
    ax4.set_xlabel("ε")
    ax4.set_ylabel("LPIPS")
    ax4.grid(True, linestyle="--", alpha=0.7)

    plt.suptitle("Adversarial Epsilon Tradeoff Curves (Budget vs Fidelity & Efficacy)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "epsilon_ablation_summary.png", dpi=300)
    plt.close()
    print(f"Epsilon ablation plots saved in {out_dir}")

def plot_transferability_heatmap(matrix_df, output_path="experiments/plots/transferability_heatmap.png"):
    """
    Plots cross-model adversarial transferability heatmap.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(matrix_df, annot=True, fmt=".1f", cmap="YlOrRd", 
                cbar_kws={"label": "Transfer Disruption Rate (%)"}, 
                vmin=0, vmax=100, linewidths=1.0, linecolor="gray", ax=ax)
    ax.set_title("Cross-Model Adversarial Transferability Heatmap", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Target Architecture Evaluated", fontsize=11)
    ax.set_ylabel("Source Surrogate Model (Perturbation Generator)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.parent / "transferability_matrix.png", dpi=300)
    plt.close()
    print(f"Transferability heatmap saved to {out_path}")

def plot_robustness_summary(robustness_df, output_path="experiments/plots/robustness_summary.png"):
    """
    Plots transformation survival and robustness bar chart.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(robustness_df["transformation"][::-1], 
                   robustness_df["cloaked_disruption_rate"][::-1], 
                   color="#3949ab", edgecolor="black", alpha=0.85, label="Cloaked Disruption")
    
    # Add clean baseline line/bars if present
    if "clean_disruption_rate" in robustness_df.columns:
        ax.barh(robustness_df["transformation"][::-1], 
                robustness_df["clean_disruption_rate"][::-1], 
                color="#9e9e9e", alpha=0.4, edgecolor="black", label="Clean Baseline Misclassification")

    ax.set_title("Adversarial Cloak Survival Under Common Image Transformations", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Disruption Rate (%)", fontsize=11)
    ax.set_ylabel("Image Transformation Applied", fontsize=11)
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    for i, v in enumerate(robustness_df["cloaked_disruption_rate"][::-1]):
        ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.parent / "robustness_curves.png", dpi=300)
    plt.close()
    print(f"Robustness summary plot saved to {out_path}")
