import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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


def run_training_and_checkpoint_validation():
    print("=" * 80)
    print("           STEP 4: SURROGATE MODEL TRAINING & CONVERGENCE AUDIT")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type.upper()}")

    # 1. Dataset loading: Top 5 major art styles with 1,200 samples for solid statistical power
    dataset = WikiArtDataset(
        data_dir="wikiart-all",
        target_field="style",
        min_samples_per_class=100,
        fast_dev_mode=True,
        fast_dev_samples=1200,
        random_seed=42
    )

    # Filter to top 5 prominent styles for clear, high-confidence convergence on CPU
    top5_styles = dataset.metadata_df["target"].value_counts().head(5).index.tolist()
    filtered_df = dataset.metadata_df[dataset.metadata_df["target"].isin(top5_styles)].copy().reset_index(drop=True)
    
    label_to_idx = {lbl: i for i, lbl in enumerate(sorted(top5_styles))}
    idx_to_label = {i: lbl for i, lbl in enumerate(sorted(top5_styles))}
    filtered_df["target_idx"] = filtered_df["target"].map(label_to_idx)
    
    dataset.metadata_df = filtered_df
    dataset.label_to_idx = label_to_idx
    dataset.idx_to_label = idx_to_label
    dataset.num_classes = len(top5_styles)

    print(f"Target Classes ({dataset.num_classes}): {list(dataset.label_to_idx.keys())}")
    print(f"Total Selected Dataset Size: {len(dataset)} samples")

    # 2. Artist-aware splits
    train_ds, val_ds, test_ds = create_artist_aware_splits(
        dataset,
        val_ratio=0.15,
        test_ratio=0.15,
        artist_aware=True,
        random_seed=42
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Test samples: {len(test_ds)}")

    # 3. Model: Pretrained ResNet-50
    print("\nInitializing ResNet-50 with ImageNet pretrained backbone...")
    model = get_model(
        model_name="resnet50",
        num_classes=dataset.num_classes,
        pretrained=True,
        device=device
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-2)
    epochs = 4
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print("\n" + "=" * 90)
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Train Acc (%)':<14} | {'Val Loss':<10} | {'Val Acc (%)':<12} | {'LR':<10} | {'Time':<8}")
    print("=" * 90)

    best_val_acc = 0.0
    best_checkpoint = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, targets, _ in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(targets).sum().item()
            train_total += imgs.size(0)

        epoch_train_loss = train_loss / max(1, train_total)
        epoch_train_acc = (train_correct / max(1, train_total)) * 100.0

        # Val
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, targets, _ in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(targets).sum().item()
                val_total += imgs.size(0)

        epoch_val_loss = val_loss / max(1, val_total)
        epoch_val_acc = (val_correct / max(1, val_total)) * 100.0
        scheduler.step()
        epoch_time = time.time() - t0

        print(f"{epoch:<6} | {epoch_train_loss:<12.4f} | {epoch_train_acc:<14.2f} | {epoch_val_loss:<10.4f} | {epoch_val_acc:<12.2f} | {current_lr:<10.2e} | {epoch_time:<6.1f}s")

        if epoch_val_acc > best_val_acc or best_checkpoint is None:
            best_val_acc = epoch_val_acc
            best_checkpoint = {
                "epoch": epoch,
                "model_name": "resnet50",
                "target_field": "style",
                "num_classes": dataset.num_classes,
                "label_to_idx": dataset.label_to_idx,
                "idx_to_label": dataset.idx_to_label,
                "state_dict": model.state_dict(),
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
            }

    # Save checkpoint
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "best_model.pt"
    torch.save(best_checkpoint, checkpoint_file)
    print(f"\n[Checkpoint Saved] -> {checkpoint_file} (Best Val Acc: {best_val_acc:.2f}%)")

    # 4. Evaluate in-memory model on held-out test set
    print("\n" + "=" * 80)
    print("           STEP 5: CHECKPOINT SERIALIZATION & RELOADING AUDIT")
    print("=" * 80)

    def evaluate_loader(eval_model, loader):
        eval_model.eval()
        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for imgs, targets, _ in loader:
                imgs = imgs.to(device)
                logits = eval_model(imgs)
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        acc = accuracy_score(y_true, y_pred) * 100.0
        bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        wf1 = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]

        return {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": f1 * 100.0,
            "weighted_f1": wf1 * 100.0,
            "macro_precision": p * 100.0,
            "macro_recall": r * 100.0,
            "y_true": y_true,
            "y_pred": y_pred,
            "probs": np.array(all_probs)
        }

    # In-memory evaluation
    orig_res = evaluate_loader(model, test_loader)
    print(f"Original In-Memory Model Evaluation on Test Set ({len(test_ds)} samples):")
    print(f"  - Top-1 Accuracy    : {orig_res['accuracy']:.2f}%")
    print(f"  - Balanced Accuracy : {orig_res['balanced_accuracy']:.2f}%")
    print(f"  - Macro F1 Score    : {orig_res['macro_f1']:.2f}%")
    print(f"  - Weighted F1 Score : {orig_res['weighted_f1']:.2f}%")

    # Reload checkpoint into a fresh instance
    print("\nReloading checkpoint into a fresh, isolated model instance...")
    reloaded_ckpt = torch.load(checkpoint_file, map_location=device)
    reloaded_model = get_model("resnet50", num_classes=dataset.num_classes, pretrained=False, device=device)
    reloaded_model.load_state_dict(reloaded_ckpt["state_dict"])

    reloaded_res = evaluate_loader(reloaded_model, test_loader)
    print(f"Reloaded Checkpoint Evaluation on Test Set ({len(test_ds)} samples):")
    print(f"  - Top-1 Accuracy    : {reloaded_res['accuracy']:.2f}%")
    print(f"  - Balanced Accuracy : {reloaded_res['balanced_accuracy']:.2f}%")
    print(f"  - Macro F1 Score    : {reloaded_res['macro_f1']:.2f}%")
    print(f"  - Weighted F1 Score : {reloaded_res['weighted_f1']:.2f}%")

    # Exact numeric verification
    acc_diff = abs(orig_res['accuracy'] - reloaded_res['accuracy'])
    assert acc_diff < 1e-4, f"Mismatch between in-memory and reloaded model: diff={acc_diff}"
    print(f"  - Checkpoint Reload Verification -> [PASS] (Difference = {acc_diff:.6f})")

    # 5. Step 6: Single-Image Classification Audit (Top 20 Correct Test Samples)
    print("\n" + "=" * 80)
    print("           STEP 6: SINGLE-IMAGE CLASSIFICATION & CONFIDENCE AUDIT")
    print("=" * 80)

    correct_indices = np.where(orig_res["y_true"] == orig_res["y_pred"])[0]
    print(f"Total correctly classified test images: {len(correct_indices)} out of {len(test_ds)}")
    
    audit_count = min(20, len(correct_indices))
    selected_correct = correct_indices[:audit_count]

    print(f"\nAudit of {audit_count} Correctly Classified Images:")
    print(f"{'#':<3} | {'Image Title / ID':<25} | {'Ground-Truth':<16} | {'Predicted':<16} | {'Top-1 Conf':<12} | {'Top-5 Predictions (Class: Probability)'}")
    print("-" * 115)

    for i, idx in enumerate(selected_correct):
        _, true_idx, meta = test_ds[idx]
        probs = orig_res["probs"][idx]
        top5_idx = np.argsort(probs)[::-1][:5]
        
        top1_conf = probs[top5_idx[0]] * 100.0
        gt_name = dataset.idx_to_label[true_idx]
        pred_name = dataset.idx_to_label[top5_idx[0]]
        
        top5_str = ", ".join([f"{dataset.idx_to_label[k]}: {probs[k]*100:.1f}%" for k in top5_idx])
        img_name = meta.get("title", f"Image_{idx}")[:23]
        if not img_name.strip():
            img_name = meta.get("id", f"ID_{idx}")[:23]

        print(f"{i+1:<3} | {img_name:<25} | {gt_name:<16} | {pred_name:<16} | {top1_conf:>6.2f}%     | {top5_str}")

    print("=" * 80)

if __name__ == "__main__":
    run_training_and_checkpoint_validation()
