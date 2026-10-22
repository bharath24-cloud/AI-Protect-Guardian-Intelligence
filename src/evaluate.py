import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from src.dataset import WikiArtDataset
from src.preprocessing import create_artist_aware_splits
from src.models import get_model, SurrogateVisionModel


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    max_classes_to_show: int = 15
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # If too many classes, select top-N most populated classes for readable visualization
    total_classes = len(class_names)
    if total_classes > max_classes_to_show:
        class_totals = cm.sum(axis=1)
        top_idx = np.argsort(class_totals)[::-1][:max_classes_to_show]
        cm_sub = cm[np.ix_(top_idx, top_idx)]
        names_sub = [class_names[i] for i in top_idx]
    else:
        cm_sub = cm
        names_sub = class_names

    # Normalized confusion matrix
    cm_norm = cm_sub.astype("float") / np.maximum(cm_sub.sum(axis=1)[:, np.newaxis], 1e-9)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=names_sub, yticklabels=names_sub,
        cbar_kws={"label": "Normalized Recall"}, ax=ax
    )
    ax.set_title(f"Normalized Confusion Matrix (Top {len(names_sub)} Classes)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Ground-Truth Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Plots] Saved confusion matrix plot to: {save_path}")


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    test_artists_count: int = 0,
    save_report: bool = True
) -> Dict[str, Any]:
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
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

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=all_labels, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=all_labels, average="weighted", zero_division=0
    )
    _, _, _, supp_cls = precision_recall_fscore_support(
        all_targets, all_preds, labels=all_labels, zero_division=0
    )

    active_support = supp_cls > 0
    bal_acc = float(np.mean(recalls[active_support]) * 100.0) if np.any(active_support) else 0.0
    mean_loss = total_loss / max(1, n_samples)

    metrics = {
        "test_samples": n_samples,
        "test_classes": len(class_names),
        "test_artists": test_artists_count,
        "test_loss": round(mean_loss, 4),
        "top1_accuracy": round(acc, 2),
        "balanced_accuracy": round(bal_acc, 2),
        "macro_precision": round(p_macro * 100.0, 2),
        "macro_recall": round(r_macro * 100.0, 2),
        "macro_f1": round(f1_macro * 100.0, 2),
        "weighted_f1": round(f1_weighted * 100.0, 2)
    }

    print("\n" + "=" * 70)
    print("                 HELD-OUT TEST SET EVALUATION")
    print("=" * 70)
    print(f"Test Samples Evaluated  : {n_samples:,}")
    print(f"Test Artists (Disjoint) : {test_artists_count}")
    print(f"Active Test Classes     : {len(class_names)}")
    print(f"Top-1 Accuracy          : {acc:.2f}%")
    print(f"Balanced Accuracy       : {bal_acc:.2f}%")
    print(f"Macro Precision         : {p_macro * 100.0:.2f}%")
    print(f"Macro Recall            : {r_macro * 100.0:.2f}%")
    print(f"Macro F1 Score          : {f1_macro * 100.0:.2f}%")
    print(f"Weighted F1 Score       : {f1_weighted * 100.0:.2f}%")
    print("=" * 70)

    if save_report:
        # Save confusion matrix plot
        plot_confusion_matrix(cm, class_names, Path("experiments/plots/confusion_matrix.png"))

        # Save JSON report
        out_json = Path("experiments/results/final_test_metrics.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Results] Saved final test metrics to: {out_json}")

    return metrics


def run_checkpoint_reload_test(
    checkpoint_path: str,
    test_loader: DataLoader,
    class_names: List[str],
    device: torch.device
) -> bool:
    """
    Evaluates original checkpoint, reinstantiates a clean model, reloads weights,
    and asserts metric equivalence.
    """
    print("\n" + "=" * 50)
    print("CHECKPOINT RELOAD TEST")
    print("=" * 50)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", "resnet50")
    num_classes = ckpt.get("num_classes", len(class_names))
    state_dict_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"

    # 1. Model A evaluation
    model_a = get_model(model_name, num_classes=num_classes, pretrained=False, device=device)
    model_a.load_state_dict(ckpt[state_dict_key])
    metrics_a = evaluate_model(model_a, test_loader, device, class_names, save_report=False)

    # 2. Re-instantiate fresh Model B and reload
    model_b = get_model(model_name, num_classes=num_classes, pretrained=False, device=device)
    model_b.load_state_dict(torch.load(checkpoint_path, map_location=device)[state_dict_key])
    metrics_b = evaluate_model(model_b, test_loader, device, class_names, save_report=False)

    diff_acc = abs(metrics_a["top1_accuracy"] - metrics_b["top1_accuracy"])
    passed = (diff_acc < 1e-4)

    print(f"Original evaluation : {metrics_a['top1_accuracy']:.2f}%")
    print(f"Reloaded evaluation : {metrics_b['top1_accuracy']:.2f}%")
    print(f"Difference          : {diff_acc:.6f}%")
    print(f"Checkpoint reload   : {'PASS' if passed else 'FAIL'}")
    print("=" * 50 + "\n")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Surrogate Model on Test Set")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="wikiart-all")
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--fast_dev_mode", action="store_true")
    parser.add_argument("--reload_test", action="store_true", help="Run checkpoint reload consistency test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # Fallback to alternate checkpoint paths
        alt_paths = [
            Path("models/checkpoints/resnet50_style_best.pth"),
            Path("models/checkpoints/resnet50_style_latest.pth")
        ]
        for alt in alt_paths:
            if alt.exists():
                ckpt_path = alt
                break

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model_name = ckpt.get("model_name", "resnet50")
        num_classes = ckpt.get("num_classes", 20)
        target_field = ckpt.get("target_field", "style")
        classes = list(ckpt.get("label_to_idx", ckpt.get("label_mapping", {})).keys())

        model = get_model(model_name, num_classes=num_classes, pretrained=False, device=device)
        state_dict_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
        model.load_state_dict(ckpt[state_dict_key])
        print(f"[Evaluate] Loaded model checkpoint from: {ckpt_path}")
    else:
        print("[Evaluate] No checkpoint found, initializing standard ResNet-50 for benchmark testing.")
        model_name = "resnet50"
        target_field = "style"
        model = get_model(model_name, num_classes=10, pretrained=True, device=device)
        classes = [f"Class_{i}" for i in range(10)]

    dataset = WikiArtDataset(
        data_dir=args.data_dir,
        target_field=target_field,
        data_fraction=args.data_fraction,
        fast_dev_mode=args.fast_dev_mode,
        fast_dev_samples=300
    )
    _, _, test_ds = create_artist_aware_splits(dataset, artist_aware=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    test_artists_cnt = len(set(dataset.metadata_df.iloc[test_ds.indices]["artist_label"]))
    evaluate_model(model, test_loader, device, classes, test_artists_count=test_artists_cnt, save_report=True)

    if args.reload_test and ckpt_path.exists():
        run_checkpoint_reload_test(str(ckpt_path), test_loader, classes, device)
