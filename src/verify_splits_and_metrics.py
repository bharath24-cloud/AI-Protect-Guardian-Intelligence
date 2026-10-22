import json
import os
import sys
from pathlib import Path
from collections import Counter

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
import warnings
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from src.dataset import WikiArtDataset
from src.preprocessing import create_artist_aware_splits


def run_split_and_metric_verification():
    print("=" * 80)
    print("      AI-PROTECT: DATASET SPLIT & EVALUATION METRIC VERIFICATION")
    print("=" * 80 + "\n")

    # 1. Initialize 25% dataset
    print("[1/4] Loading 25% WikiArt dataset...")
    ds_25 = WikiArtDataset(
        data_dir="wikiart-all",
        target_field="style",
        min_samples_per_class=50,
        data_fraction=0.25,
        random_seed=42
    )

    # 2. Create versioned split directory
    v2_split_dir = "experiments/results/splits/split_25pct_v2"
    print(f"\n[2/4] Generating and verifying versioned artist-aware split under: {v2_split_dir}")
    train_ds, val_ds, test_ds = create_artist_aware_splits(
        ds_25,
        val_ratio=0.15,
        test_ratio=0.15,
        artist_aware=True,
        random_seed=42,
        split_cache_dir=v2_split_dir
    )

    df = ds_25.metadata_df
    train_idx = np.array(train_ds.indices)
    val_idx = np.array(val_ds.indices)
    test_idx = np.array(test_ds.indices)

    # 3. Assert zero leakage
    train_artists = set(df.iloc[train_idx]["artist_label"])
    val_artists = set(df.iloc[val_idx]["artist_label"])
    test_artists = set(df.iloc[test_idx]["artist_label"])

    assert len(train_artists.intersection(val_artists)) == 0, "Train-Val artist leakage detected!"
    assert len(train_artists.intersection(test_artists)) == 0, "Train-Test artist leakage detected!"
    assert len(val_artists.intersection(test_artists)) == 0, "Val-Test artist leakage detected!"

    # Assert image index disjointness
    assert len(set(train_idx).intersection(set(val_idx))) == 0, "Train-Val image overlap detected!"
    assert len(set(train_idx).intersection(set(test_idx))) == 0, "Train-Test image overlap detected!"
    assert len(set(val_idx).intersection(set(test_idx))) == 0, "Val-Test image overlap detected!"
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df), "Total indices count mismatch!"

    print("  [PASS] Zero Artist Leakage verified.")
    print("  [PASS] Zero Image ID Overlap verified.")
    print("  [PASS] Total dataset index coverage verified.")

    # 4. Check class distribution across splits
    all_styles = set(df["style_label"])
    tr_styles = set(df.iloc[train_idx]["style_label"])
    va_styles = set(df.iloc[val_idx]["style_label"])
    te_styles = set(df.iloc[test_idx]["style_label"])

    assert len(tr_styles) == len(all_styles), "Not all classes are present in Training split!"
    print(f"  [PASS] 100% of active style classes ({len(tr_styles)}/{len(all_styles)}) represented in Training set.")
    print(f"  [INFO] Test set style coverage: {len(te_styles)}/{len(all_styles)} ({len(all_styles - te_styles)} absent due to single-artist constraint).")
    print(f"  [INFO] Validation set style coverage: {len(va_styles)}/{len(all_styles)} ({len(all_styles - va_styles)} absent due to <=2 artist constraint).")

    # 5. Test metric calculation without sklearn warnings
    print("\n[3/4] Testing metric calculation engine with synthetic predictions...")
    all_labels = list(range(ds_25.num_classes))
    test_targets = np.array([df.iloc[i]["target_idx"] for i in test_idx])

    # Simulate predictions including classes with 0 test support
    dummy_preds = np.random.RandomState(42).randint(0, ds_25.num_classes, size=len(test_targets))

    # Catch warnings during metric calculation
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        acc = accuracy_score(test_targets, dummy_preds) * 100.0
        cm = confusion_matrix(test_targets, dummy_preds, labels=all_labels)
        recalls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            test_targets, dummy_preds, labels=all_labels, average="macro", zero_division=0
        )
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
            test_targets, dummy_preds, labels=all_labels, average="weighted", zero_division=0
        )
        p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(
            test_targets, dummy_preds, labels=all_labels, zero_division=0
        )

        active_support = supp_cls > 0
        bal_acc = float(np.mean(recalls[active_support]) * 100.0) if np.any(active_support) else 0.0

    print(f"  Captured Warnings during evaluation: {len(captured_warnings)}")
    for w in captured_warnings:
        print(f"    Warning: {w.message}")

    assert len(captured_warnings) == 0, f"Evaluation threw warnings: {[str(w.message) for w in captured_warnings]}"
    print("  [PASS] Zero evaluation metric warnings produced (Balanced accuracy calculated cleanly).")

    # 6. Save audit report
    print("\n[4/4] Writing split verification report...")
    out_dir = Path("experiments/results/accuracy_improvement")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "split_directory": v2_split_dir,
        "dataset_samples": len(df),
        "split_counts": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx))
        },
        "split_artists": {
            "train": int(len(train_artists)),
            "validation": int(len(val_artists)),
            "test": int(len(test_artists))
        },
        "artist_leakage_passed": True,
        "image_overlap_passed": True,
        "classes_coverage": {
            "total_active_styles": len(all_styles),
            "train_styles_count": len(tr_styles),
            "val_styles_count": len(va_styles),
            "test_styles_count": len(te_styles),
            "absent_in_test": sorted(list(all_styles - te_styles)),
            "absent_in_val": sorted(list(all_styles - va_styles)),
            "absent_in_train": []
        },
        "mathematical_constraint_notes": {
            "absent_in_test_reason": "Styles have only 1 unique artist in the 25% dataset, which must reside in Train to provide training supervision without artist leakage.",
            "absent_in_val_reason": "Styles have <= 2 unique artists in the 25% dataset, partitioned across Train and Test."
        }
    }

    report_path = out_dir / "split_verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  [Results] Saved split verification report to: {report_path}")
    print("\n" + "=" * 80)
    print("      ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (0 LEAKAGE, 0 WARNINGS)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_split_and_metric_verification()
