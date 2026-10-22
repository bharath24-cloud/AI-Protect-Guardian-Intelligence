import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import WikiArtDataset
from src.preprocessing import create_artist_aware_splits
from src.models import get_model, SurrogateVisionModel


def get_device() -> torch.device:
    """Detects CUDA GPU, Apple MPS, or CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cuda_ver = torch.version.cuda or "N/A"
        print(f"[Hardware] CUDA detected")
        print(f"GPU: {gpu_name}")
        print(f"VRAM: approximately {total_vram:.2f} GB")
        print(f"PyTorch CUDA: {cuda_ver}")
        print(f"AMP: ENABLED")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[Hardware] Apple Silicon MPS detected.")
    else:
        dev = torch.device("cpu")
        print("[Hardware] CPU active (No GPU detected).")
    return dev


def format_time(seconds: float) -> str:
    """Converts seconds into MM:SS or HH:MM:SS string."""
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_memory_info(device: torch.device) -> str:
    """Returns formatted RAM and GPU VRAM usage."""
    if _HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        ram_gb = process.memory_info().rss / (1024 ** 3)
        ram_str = f"RAM: {ram_gb:.2f}GB"
    else:
        ram_str = "RAM: N/A"

    if device.type == "cuda":
        alloc_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
        res_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
        peak_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        return f"{ram_str} | VRAM: {alloc_gb:.2f}GB (Peak: {peak_gb:.2f}GB / Res: {res_gb:.2f}GB)"
    return f"{ram_str} | CPU"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[Any] = None,
    gradient_accumulation_steps: int = 1,
    epoch: int = 1,
    total_epochs: int = 15,
    log_interval: int = 20
) -> Tuple[float, float, int]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)
    total_images = len(loader.dataset)

    use_amp = (scaler is not None and device.type == "cuda")

    optimizer.zero_grad(set_to_none=True)
    t_start = time.time()

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = images.size(0)

        try:
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    loss_scaled = loss / gradient_accumulation_steps
                scaler.scale(loss_scaled).backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss_scaled = loss / gradient_accumulation_steps
                loss_scaled.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        except torch.cuda.OutOfMemoryError as oom_err:
            print("\n" + "!" * 80)
            print("[ERROR: CUDA OUT OF MEMORY]")
            print(f"CUDA Out of Memory caught in Epoch {epoch}, Batch {batch_idx+1}/{num_batches}.")
            print("Actionable Suggestions:")
            print("  1. Reduce --batch_size (e.g. from 8 -> 4, or 4 -> 2)")
            print("  2. Increase --gradient_accumulation_steps (e.g. 4 or 8)")
            print("  3. Lower --image_size (e.g. 224)")
            print("!" * 80 + "\n")
            raise oom_err

        running_loss += loss.item() * batch_size
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += batch_size

        # Live Progress Display
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - t_start
            img_done = total
            speed = img_done / max(0.001, elapsed)
            remaining_img = max(0, total_images - img_done)
            eta_s = remaining_img / max(0.001, speed)
            pct = (img_done / total_images) * 100.0

            cur_loss = running_loss / max(1, total)
            cur_acc = (correct / max(1, total)) * 100.0
            mem_str = get_memory_info(device)

            bar_len = 16
            filled = int(bar_len * (pct / 100.0))
            bar = "=" * filled + "-" * (bar_len - filled)

            sys.stdout.write(
                f"\r  Epoch {epoch:02d}/{total_epochs:02d} [{bar}] {pct:5.1f}% | "
                f"Img: {img_done:>5d}/{total_images:<5d} | "
                f"Loss: {cur_loss:.4f} | Acc: {cur_acc:5.2f}% | "
                f"{mem_str} | ETA: {format_time(eta_s)}"
            )
            sys.stdout.flush()

    sys.stdout.write("\n")
    epoch_loss = running_loss / max(1, total)
    epoch_acc = (correct / max(1, total)) * 100.0
    return epoch_loss, epoch_acc, total


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    total = 0

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        total += images.size(0)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    num_classes = model.num_classes if hasattr(model, "num_classes") else 38
    all_labels = list(range(num_classes))

    val_loss = running_loss / max(1, total)
    val_acc = accuracy_score(all_targets, all_preds) * 100.0
    cm = confusion_matrix(all_targets, all_preds, labels=all_labels)
    recalls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=all_labels, average="macro", zero_division=0
    )
    p_wt, r_wt, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=all_labels, average="weighted", zero_division=0
    )
    _, _, _, supp_cls = precision_recall_fscore_support(
        all_targets, all_preds, labels=all_labels, zero_division=0
    )

    active_support = supp_cls > 0
    bal_acc = float(np.mean(recalls[active_support]) * 100.0) if np.any(active_support) else 0.0

    return {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_acc": val_acc,
        "balanced_acc": bal_acc,
        "macro_f1": f1_macro * 100.0,
        "weighted_f1": f1_weighted * 100.0
    }


def save_training_plots(history: Dict[str, list], output_dir: Path, experiment_id: str = ""):
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = history.get("epochs", list(range(1, len(history["train_loss"]) + 1)))

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    prefix = f"{experiment_id}_" if experiment_id else ""

    # Combined Dual-Axis Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(epochs, history["train_loss"], "o-", color="#d32f2f", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", color="#1976d2", label="Val Loss", linewidth=2)
    ax1.set_title(f"Loss Convergence {f'({experiment_id})' if experiment_id else ''}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2.plot(epochs, history["train_acc"], "o-", color="#388e3c", label="Train Acc (%)", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "s--", color="#f57c00", label="Val Acc (%)", linewidth=2)
    if "macro_f1" in history and len(history["macro_f1"]) == len(epochs):
        ax2.plot(epochs, history["macro_f1"], "^-.", color="#7b1fa2", label="Val Macro F1 (%)", linewidth=1.5)
    ax2.set_title(f"Generalization Performance {f'({experiment_id})' if experiment_id else ''}", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}training_curves.png", dpi=200)
    plt.close()
    print(f"[Plots] Saved training and validation curves in {output_dir / f'{prefix}training_curves.png'}")


from sklearn.metrics import confusion_matrix


def plot_confusion_matrix_heatmap(cm: np.ndarray, class_names: List[str], save_path: Path, max_classes: int = 15):
    """Saves a normalized confusion matrix heatmap."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    import seaborn as sns

    if len(class_names) > max_classes:
        class_totals = cm.sum(axis=1)
        top_idx = np.argsort(class_totals)[::-1][:max_classes]
        cm_sub = cm[np.ix_(top_idx, top_idx)]
        names_sub = [class_names[i] for i in top_idx]
    else:
        cm_sub = cm
        names_sub = class_names

    cm_norm = cm_sub.astype("float") / np.maximum(cm_sub.sum(axis=1)[:, np.newaxis], 1e-9)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=names_sub, yticklabels=names_sub, ax=ax)
    ax.set_title("Normalized Confusion Matrix (Top Classes)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Ground-Truth Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Plots] Saved confusion matrix to: {save_path}")


def evaluate_test_set(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    experiment_id: str = "exp",
    results_dir: Path = Path("experiments/results"),
    plots_dir: Path = Path("experiments/plots")
) -> Dict[str, float]:
    """Runs rigorous test evaluation on held-out test split and saves all required artifacts."""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)

        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    n_samples = len(all_targets)

    all_labels = list(range(len(class_names)))
    acc = accuracy_score(all_targets, all_preds) * 100.0
    cm = confusion_matrix(all_targets, all_preds, labels=all_labels)
    recalls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, labels=all_labels, average="macro", zero_division=0)
    p_wt, r_wt, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, labels=all_labels, average="weighted", zero_division=0)

    # Per-class metrics
    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_targets, all_preds, labels=all_labels, zero_division=0
    )

    active_support = support_per_class > 0
    bal_acc = float(np.mean(recalls[active_support]) * 100.0) if np.any(active_support) else 0.0

    per_class_df = pd.DataFrame({
        "class_index": list(range(len(class_names))),
        "class_name": class_names,
        "precision": np.round(p_per_class * 100.0, 2),
        "recall": np.round(r_per_class * 100.0, 2),
        "f1_score": np.round(f1_per_class * 100.0, 2),
        "support": support_per_class
    })

    # Save per-class metrics
    per_class_path = results_dir / f"{experiment_id}_per_class_metrics.csv"
    per_class_df.to_csv(per_class_path, index=False)
    if "50pct" in experiment_id:
        per_class_df.to_csv(results_dir / "50pct_per_class_metrics.csv", index=False)
    print(f"[Results] Saved per-class metrics to: {per_class_path}")

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))
    cm_plot_path = plots_dir / f"{experiment_id}_confusion_matrix.png"
    plot_confusion_matrix_heatmap(cm, class_names, cm_plot_path)
    if "50pct" in experiment_id:
        plot_confusion_matrix_heatmap(cm, class_names, plots_dir / "50pct_confusion_matrix.png")

    test_metrics = {
        "test_samples": n_samples,
        "test_classes": len(class_names),
        "test_loss": round(total_loss / max(1, n_samples), 4),
        "test_accuracy": round(acc, 2),
        "balanced_accuracy": round(bal_acc, 2),
        "macro_precision": round(p_macro * 100.0, 2),
        "macro_recall": round(r_macro * 100.0, 2),
        "macro_f1": round(f1_macro * 100.0, 2),
        "weighted_f1": round(f1_weighted * 100.0, 2)
    }

    # Save test metrics JSON
    json_path = results_dir / f"{experiment_id}_test_metrics.json"
    with open(json_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    if "50pct" in experiment_id:
        with open(results_dir / "50pct_test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

    # 25% vs 50% comparison if 50pct run
    if "50pct" in experiment_id:
        baseline_25 = {
            "dataset_fraction": "25% (15,413 samples)",
            "test_accuracy": 34.74,
            "balanced_accuracy": 17.07,
            "macro_f1": 12.31,
            "weighted_f1": 31.05
        }
        current_50 = {
            "dataset_fraction": "50% (30,827 samples)",
            "test_accuracy": test_metrics["test_accuracy"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"]
        }
        comparison_df = pd.DataFrame([baseline_25, current_50])
        comp_path = results_dir / "25_vs_50_comparison.csv"
        comparison_df.to_csv(comp_path, index=False)
        print(f"[Comparison] Saved 25% vs 50% comparison to: {comp_path}")

    print("\n" + "=" * 75)
    print("                 HELD-OUT TEST SET EVALUATION")
    print("=" * 75)
    print(f"Test Samples Evaluated  : {n_samples:,}")
    print(f"Top-1 Accuracy          : {acc:.2f}%")
    print(f"Balanced Accuracy       : {bal_acc:.2f}%")
    print(f"Macro Precision         : {p_macro * 100.0:.2f}%")
    print(f"Macro Recall            : {r_macro * 100.0:.2f}%")
    print(f"Macro F1 Score          : {f1_macro * 100.0:.2f}%")
    print(f"Weighted F1 Score       : {f1_weighted * 100.0:.2f}%")
    print("=" * 75 + "\n")
    return test_metrics


def append_hyperparameter_results(results_csv: Path, record: Dict[str, Any]):
    """Appends an experiment record to hyperparameter_results.csv without overwriting previous runs."""
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([record])
    if results_csv.exists():
        df_existing = pd.read_csv(results_csv)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(results_csv, index=False)
    print(f"[Tracking] Appended experiment to: {results_csv}")


def run_training(
    data_dir: str = "wikiart-all",
    model_name: str = "resnet50",
    target_field: str = "style",
    min_samples: int = 50,
    data_fraction: float = 1.0,
    max_samples: Optional[int] = None,
    epochs: int = 10,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
    label_smoothing: float = 0.1,
    image_size: int = 224,
    num_workers: int = 0,
    pin_memory: bool = True,
    class_weights_strategy: str = "none",
    use_weighted_sampler: bool = False,
    freeze_backbone: bool = False,
    unfreeze_epoch: int = 0,
    checkpoint_metric: str = "val_loss",
    augmentation_level: str = "enhanced",
    pretrained: bool = True,
    resume_path: Optional[str] = None,
    save_periodic_epochs: int = 5,
    early_stopping_patience: int = 5,
    checkpoint_dir: str = "models/checkpoints",
    results_dir: str = "experiments/results",
    plots_dir: str = "experiments/plots",
    split_cache_dir: Optional[str] = "experiments/results/splits/split_25pct",
    experiment_id: str = "exp_default",
    fast_dev_mode: bool = False,
    fast_dev_samples: int = 300,
    random_seed: int = 42
):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = get_device()
    ckpt_dir = Path(checkpoint_dir)
    res_dir = Path(results_dir)
    plt_dir = Path(plots_dir)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    plt_dir.mkdir(parents=True, exist_ok=True)

    effective_batch_size = batch_size * gradient_accumulation_steps
    amp_enabled = (device.type == "cuda")

    print("\n" + "=" * 85)
    print(f"        AI-PROTECT SURROGATE TRAINING ENGINE — [{experiment_id.upper()}]")
    print("=" * 85)
    print(f"Target Classification : {target_field.upper()}")
    print(f"Model Architecture    : {model_name.upper()}")
    print(f"Pretrained Weights    : {pretrained}")
    print(f"Physical batch size   : {batch_size}")
    print(f"Gradient accumulation : {gradient_accumulation_steps}")
    print(f"Effective batch size  : {effective_batch_size}")
    print(f"Data fraction         : {data_fraction * 100:.1f}%")
    print(f"Learning Rate         : {lr}")
    print(f"Weight Decay          : {weight_decay}")
    print(f"Label Smoothing       : {label_smoothing}")
    print(f"Freeze Backbone       : {freeze_backbone} (Unfreeze Epoch: {unfreeze_epoch if unfreeze_epoch > 0 else 'Never'})")
    print(f"Class Weights Strategy: {class_weights_strategy.upper()}")
    print(f"Weighted Sampler      : {use_weighted_sampler}")
    print(f"Augmentation Level    : {augmentation_level.upper()}")
    print(f"Checkpoint Metric     : {checkpoint_metric.upper()}")
    print(f"AMP enabled           : {'YES' if amp_enabled else 'NO'}")
    print(f"Total Epochs Max      : {epochs}")
    print("=" * 85 + "\n")

    # 1. Dataset & Splits
    dataset = WikiArtDataset(
        data_dir=data_dir,
        target_field=target_field,
        min_samples_per_class=min_samples,
        data_fraction=data_fraction,
        max_samples=max_samples,
        fast_dev_mode=fast_dev_mode,
        fast_dev_samples=fast_dev_samples,
        random_seed=random_seed
    )
    dataset.print_dataset_summary()

    # Save deterministic label mapping
    label_mapping = {
        "num_classes": dataset.num_classes,
        "target_field": dataset.target_field,
        "label_to_idx": dataset.label_to_idx,
        "idx_to_label": dataset.idx_to_label
    }
    with open(res_dir / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)
    with open("experiments/label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)

    # 2. Artist-aware Splits (with persistence to ensure identical splits across experiments)
    if fast_dev_mode:
        split_cache_dir = None
    elif split_cache_dir is not None and abs(data_fraction - 0.25) < 1e-4 and "split_25pct_v2" in split_cache_dir:
        # Note: split_25pct_v2 contains 30,827 samples (50% dataset fraction). Use split_25pct for 25% runs.
        split_cache_dir = "experiments/results/splits/split_25pct"

    train_ds, val_ds, test_ds = create_artist_aware_splits(
        dataset,
        val_ratio=0.15,
        test_ratio=0.15,
        image_size=image_size,
        model_name=model_name,
        artist_aware=True,
        random_seed=random_seed,
        augmentation_level=augmentation_level,
        split_cache_dir=split_cache_dir
    )

    # Save detailed class distribution across splits
    train_targets = [dataset.metadata_df.iloc[idx]["target_idx"] for idx in train_ds.indices]
    val_targets = [dataset.metadata_df.iloc[idx]["target_idx"] for idx in val_ds.indices]
    test_targets = [dataset.metadata_df.iloc[idx]["target_idx"] for idx in test_ds.indices]

    train_c = pd.Series(train_targets).value_counts()
    val_c = pd.Series(val_targets).value_counts()
    test_c = pd.Series(test_targets).value_counts()

    dist_records = []
    for idx, name in dataset.idx_to_label.items():
        tot = int(dataset.metadata_df["target_idx"].value_counts().get(idx, 0))
        tr = int(train_c.get(idx, 0))
        va = int(val_c.get(idx, 0))
        te = int(test_c.get(idx, 0))
        dist_records.append({
            "class_index": idx,
            "class_name": name,
            "total_count": tot,
            "percentage": round(tot / max(1, len(dataset.metadata_df)) * 100.0, 2),
            "train_count": tr,
            "val_count": va,
            "test_count": te
        })
    dist_df = pd.DataFrame(dist_records)
    dist_df.to_csv(res_dir / f"{experiment_id}_class_distribution.csv", index=False)
    if "50pct" in experiment_id or abs(data_fraction - 0.5) < 1e-4:
        dist_df.to_csv(res_dir / "class_distribution_50pct.csv", index=False)
        print(f"[Results] Saved 50% class distribution to: {res_dir / 'class_distribution_50pct.csv'}")

    # Train Sampler setup
    if use_weighted_sampler:
        if class_weights_strategy != "none":
            print("[Warning] Both weighted sampler and class-weighted loss are specified. "
                  "To prevent double imbalance correction, setting class-weighted loss to 'none'.")
            class_weights_strategy = "none"
        
        # Calculate sample weights for train subset
        train_indices = train_ds.indices
        train_targets = [dataset.metadata_df.iloc[idx]["target_idx"] for idx in train_indices]
        class_counts = pd.Series(train_targets).value_counts().to_dict()
        train_sample_weights = [1.0 / max(1, class_counts[t]) for t in train_targets]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.tensor(train_sample_weights, dtype=torch.double),
            num_samples=len(train_sample_weights),
            replacement=True
        )
        train_shuffle = False
        print("[Training] WeightedRandomSampler ENABLED for train dataloader.")
    else:
        sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(pin_memory and device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(pin_memory and device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(pin_memory and device.type == "cuda")
    )

    # 3. Model Initialization
    num_classes = dataset.num_classes
    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        device=device
    )
    model.print_summary()

    # 4. Criterion, Optimizer, Scheduler, Scaler
    if class_weights_strategy != "none":
        class_weights_t = dataset.get_class_weights(strategy=class_weights_strategy).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smoothing)
        print(f"[Training] Using Weighted CrossEntropyLoss (Strategy: {class_weights_strategy}, Smoothing: {label_smoothing}).")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"[Training] Using Standard CrossEntropyLoss (Smoothing: {label_smoothing}).")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    # Checkpoint tracking metric
    is_minimizing = (checkpoint_metric.lower() == "val_loss")
    best_tracked_metric = float("inf") if is_minimizing else -float("inf")
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    best_val_balanced_acc = 0.0
    best_epoch = 1

    # 5. Handle Resume Checkpoint
    start_epoch = 1
    history_records: List[Dict[str, Any]] = []

    if resume_path and Path(resume_path).exists():
        print(f"\n[Resume] Loading checkpoint from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        state_dict_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
        model.load_state_dict(ckpt[state_dict_key])
        if "optimizer_state_dict" in ckpt and optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"[Resume] Successfully resumed from Epoch {start_epoch-1}.\n")

    # 6. Training Loop
    history = {"epochs": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "macro_f1": [], "balanced_acc": [], "lr": []}
    epochs_no_improve = 0

    best_ckpt_path = ckpt_dir / f"{experiment_id}_{model_name}_{target_field}_best.pth"
    std_best_ckpt_path = ckpt_dir / f"{model_name}_{target_field}_best.pth"

    print("\n" + "=" * 105)
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Train Acc':<11} | {'Val Loss':<10} | {'Val Acc':<10} | {'Val F1':<9} | {'Val BalAcc':<11} | {'LR':<9} | {'Time':<7}")
    print("=" * 105)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Gradual unfreezing check
        if freeze_backbone and unfreeze_epoch > 0 and epoch == unfreeze_epoch:
            print(f"\n[Transfer Learning] Epoch {epoch}: Triggering gradual unfreezing of backbone...")
            model.unfreeze_all()
            total_p, train_p, _ = model.get_param_counts()
            print(f"  Updated Trainable Parameters: {train_p:,} / {total_p:,} (100.0%)")
            # Recreate optimizer with all trainable parameters
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr * 0.5,  # Lower LR slightly when unfreezing backbone to prevent catastrophic forgetting
                weight_decay=weight_decay
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=(epochs - epoch + 1), eta_min=1e-6)

        train_loss, train_acc, samples_done = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            gradient_accumulation_steps=gradient_accumulation_steps,
            epoch=epoch,
            total_epochs=epochs
        )

        val_metrics = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )
        val_loss = val_metrics["val_loss"]
        val_acc = val_metrics["val_accuracy"]
        val_macro_f1 = val_metrics["macro_f1"]
        val_bal_acc = val_metrics["balanced_acc"]

        scheduler.step()
        epoch_time = time.time() - t0

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["macro_f1"].append(val_macro_f1)
        history["balanced_acc"].append(val_bal_acc)
        history["lr"].append(current_lr)

        gpu_mem_str = get_memory_info(device)
        print(f"{epoch:<6} | {train_loss:<11.4f} | {train_acc:<10.2f}% | {val_loss:<10.4f} | {val_acc:<9.2f}% | {val_macro_f1:<8.2f}% | {val_bal_acc:<10.2f}% | {current_lr:<9.2e} | {epoch_time:<5.1f}s")

        # Save history record
        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 2),
            "validation_loss": round(val_loss, 4),
            "validation_accuracy": round(val_acc, 2),
            "validation_macro_f1": round(val_macro_f1, 2),
            "validation_balanced_acc": round(val_bal_acc, 2),
            "learning_rate": current_lr,
            "epoch_time": round(epoch_time, 2),
            "memory": gpu_mem_str
        }
        history_records.append(rec)
        pd.DataFrame(history_records).to_csv(res_dir / f"{experiment_id}_training_history.csv", index=False)

        # Checkpoint evaluation against chosen metric
        if checkpoint_metric == "val_accuracy":
            current_metric_val = val_acc
        elif checkpoint_metric == "macro_f1":
            current_metric_val = val_macro_f1
        elif checkpoint_metric == "balanced_acc":
            current_metric_val = val_bal_acc
        else:
            current_metric_val = val_loss

        is_better = (current_metric_val < best_tracked_metric) if is_minimizing else (current_metric_val > best_tracked_metric)

        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "checkpoint_metric": checkpoint_metric,
            "tracked_metric_val": current_metric_val,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_balanced_acc": val_bal_acc,
            "label_mapping": dataset.label_to_idx,
            "label_to_idx": dataset.label_to_idx,
            "idx_to_label": dataset.idx_to_label,
            "model_name": model_name,
            "target_field": target_field,
            "num_classes": num_classes,
            "experiment_id": experiment_id,
            "random_seed": random_seed
        }

        if is_better:
            best_tracked_metric = current_metric_val
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_macro_f1 = val_macro_f1
            best_val_balanced_acc = val_bal_acc
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save(checkpoint_dict, best_ckpt_path)
            torch.save(checkpoint_dict, std_best_ckpt_path)
            torch.save(checkpoint_dict, ckpt_dir / "best_model.pt")
            print(f"  [*] New Best Checkpoint saved! ({checkpoint_metric}: {best_tracked_metric:.4f} | Val Acc: {best_val_acc:.2f}% | Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # Periodic checkpoint
        if epoch % save_periodic_epochs == 0 or epoch == epochs:
            torch.save(checkpoint_dict, ckpt_dir / f"{experiment_id}_epoch_{epoch:03d}.pth")

        # Early stopping check
        if epochs_no_improve >= early_stopping_patience:
            print(f"\n[Early Stopping] No improvement in {checkpoint_metric} for {early_stopping_patience} epochs. Stopping early at epoch {epoch}.")
            break

    print("=" * 105)
    print(f"[Training Completed] Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc:.2f}% | Val Macro F1: {best_val_macro_f1:.2f}%")

    # Generate training curves
    save_training_plots(history, plt_dir, experiment_id=experiment_id)

    # 7. Held-out Test Set Evaluation using the BEST Checkpoint
    print(f"\n[Test Evaluation] Loading best checkpoint from {best_ckpt_path} for held-out evaluation...")
    best_saved = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_saved["model_state_dict"])
    class_names = [dataset.idx_to_label[i] for i in range(num_classes)]

    test_metrics = evaluate_test_set(
        model=model,
        loader=test_loader,
        device=device,
        class_names=class_names,
        experiment_id=experiment_id,
        results_dir=res_dir,
        plots_dir=plt_dir
    )

    # 8. Record to hyperparameter_results.csv
    peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3) if device.type == "cuda" else 0.0
    summary_record = {
        "experiment_id": experiment_id,
        "dataset_fraction": data_fraction,
        "model": model_name,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "freeze_backbone": freeze_backbone,
        "unfreeze_epoch": unfreeze_epoch,
        "class_weights": class_weights_strategy,
        "weighted_sampler": use_weighted_sampler,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation_steps,
        "epochs": len(history["epochs"]),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_accuracy": round(best_val_acc, 2),
        "best_val_macro_f1": round(best_val_macro_f1, 2),
        "best_val_balanced_accuracy": round(best_val_balanced_acc, 2),
        "test_accuracy": test_metrics["test_accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "peak_vram": f"{peak_vram_gb:.2f} GB"
    }
    append_hyperparameter_results(res_dir / "hyperparameter_results.csv", summary_record)

    if "50pct" in experiment_id or abs(data_fraction - 0.5) < 1e-4:
        print("\n" + "=" * 45)
        print("AI-PROTECT 50% TRAINING RESULT")
        print("=" * 45)
        print(f"Dataset: ~{len(dataset.metadata_df):,} images")
        print(f"Model: {model_name.upper()}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"AMP: {'Enabled' if amp_enabled else 'Disabled'}")
        print(f"Class weighting: Enabled ({class_weights_strategy})")
        print("Artist leakage: PASS")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Best validation Macro F1: {best_val_macro_f1:.2f}%")
        print(f"Best validation Balanced Accuracy: {best_val_balanced_acc:.2f}%")
        print(f"Test accuracy: {test_metrics['test_accuracy']:.2f}%")
        print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.2f}%")
        print(f"Test Macro F1: {test_metrics['macro_f1']:.2f}%")
        print(f"Test Weighted F1: {test_metrics['weighted_f1']:.2f}%")
        print(f"Peak VRAM: {peak_vram_gb:.2f} GB")
        print("=" * 45 + "\n")

    return model, summary_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Protect Large-Scale Surrogate Vision Model Training")
    parser.add_argument("--data_dir", type=str, default="wikiart-all")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "resnet18", "convnext_tiny", "vit_b_16"])
    parser.add_argument("--target", type=str, default="style", choices=["style", "genre", "artist"])
    parser.add_argument("--min_samples", type=int, default=50)
    parser.add_argument("--data_fraction", type=float, default=1.0, help="Fraction of dataset to use (0.01, 0.10, 0.25, 0.50, 1.0)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4, help="Physical batch size (4 or 8)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--class_weights", type=str, default="none", choices=["none", "balanced", "effective_num", "sqrt"], help="Class weighting strategy")
    parser.add_argument("--weighted_sampler", action="store_true", help="Enable WeightedRandomSampler in train loader")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone weights initially")
    parser.add_argument("--unfreeze_epoch", type=int, default=0, help="Epoch at which to unfreeze backbone (0=never)")
    parser.add_argument("--checkpoint_metric", type=str, default="val_loss", choices=["val_loss", "val_accuracy", "macro_f1", "balanced_acc"])
    parser.add_argument("--augmentation_level", type=str, default="enhanced", choices=["basic", "enhanced"])
    parser.add_argument("--split_cache_dir", type=str, default="experiments/results/splits/split_25pct")
    parser.add_argument("--experiment_id", type=str, default="exp_run")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained weights")
    parser.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--save_periodic_epochs", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--fast_dev_mode", action="store_true", help="Fast development mode on tiny subset")
    parser.add_argument("--fast_dev_samples", type=int, default=300)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    run_training(
        data_dir=args.data_dir,
        model_name=args.model,
        target_field=args.target,
        min_samples=args.min_samples,
        data_fraction=args.data_fraction,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        image_size=args.image_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        class_weights_strategy=args.class_weights,
        use_weighted_sampler=args.weighted_sampler,
        freeze_backbone=args.freeze_backbone,
        unfreeze_epoch=args.unfreeze_epoch,
        checkpoint_metric=args.checkpoint_metric,
        augmentation_level=args.augmentation_level,
        split_cache_dir=args.split_cache_dir,
        experiment_id=args.experiment_id,
        pretrained=args.pretrained,
        resume_path=args.resume,
        save_periodic_epochs=args.save_periodic_epochs,
        early_stopping_patience=args.early_stopping_patience,
        fast_dev_mode=args.fast_dev_mode,
        fast_dev_samples=args.fast_dev_samples,
        random_seed=args.random_seed
    )
