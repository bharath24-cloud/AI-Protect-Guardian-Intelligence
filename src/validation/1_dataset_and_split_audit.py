import io
import json
import glob
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from src.dataset import WikiArtDataset
from src.preprocessing import create_artist_aware_splits

def run_dataset_and_split_audit():
    print("=" * 80)
    print("           STEP 1: COMPREHENSIVE DATASET VALIDATION (FULL SCAN)")
    print("=" * 80)

    parquet_files = sorted(glob.glob("wikiart-all/data/*.parquet"))
    print(f"Total Parquet files found: {len(parquet_files)}")

    # 1. Full metadata scan across all files
    dfs = []
    file_row_counts = {}
    
    for f in parquet_files:
        table = pq.read_table(f, columns=[
            "id", "title", "artistName", "artistId",
            "genres", "styles", "media", "width", "height"
        ])
        df = table.to_pandas()
        file_row_counts[Path(f).name] = len(df)
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    total_rows = len(full_df)
    print(f"\nRow counts per Parquet file:")
    for fname, cnt in file_row_counts.items():
        print(f"  {fname:<30}: {cnt:>6} rows")
    print(f"  {'TOTAL ROWS ACROSS ALL FILES':<30}: {total_rows:>6} rows")

    # Helper for extracting primary label
    def extract_primary(val):
        if val is None:
            return "Unknown"
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) > 0 and val[0] is not None and str(val[0]).strip() != "":
                return str(val[0]).strip()
            return "Unknown"
        if isinstance(val, str) and val.strip() != "":
            return val.strip()
        return "Unknown"

    full_df["style_label"] = full_df["styles"].apply(extract_primary)
    full_df["genre_label"] = full_df["genres"].apply(extract_primary)
    full_df["artist_label"] = full_df["artistName"].fillna("Unknown").astype(str).str.strip()

    num_unique_artists = full_df["artist_label"].nunique()
    num_unique_styles = full_df["style_label"].nunique()
    num_unique_genres = full_df["genre_label"].nunique()

    print(f"\nUnique Entities in Full Dataset:")
    print(f"  - Unique Artists : {num_unique_artists}")
    print(f"  - Unique Styles  : {num_unique_styles}")
    print(f"  - Unique Genres  : {num_unique_genres}")

    # Style class distribution before filtering
    style_counts_all = full_df["style_label"].value_counts()
    classes_before = len(style_counts_all[style_counts_all.index != "Unknown"])
    print(f"\nTotal style classes before filtering: {classes_before}")

    # Min samples filtering (>= 50 samples)
    min_samples = 50
    valid_classes = style_counts_all[
        (style_counts_all.index != "Unknown") & (style_counts_all >= min_samples)
    ]
    classes_after = len(valid_classes)
    usable_samples = valid_classes.sum()

    print(f"Total style classes after filtering (min_samples >= {min_samples}): {classes_after}")
    print(f"Total usable artwork samples after filtering: {usable_samples} (out of {total_rows})")

    print(f"\nTop 15 Style Classes (Filtered):")
    for s_name, s_cnt in valid_classes.head(15).items():
        print(f"  - {s_name:<30}: {s_cnt:>6} samples ({s_cnt/usable_samples*100:5.2f}%)")

    # Verify image decodability on a sample of 250 images across files
    print(f"\nVerifying image decodability on sample of 250 images across all 9 Parquet files...")
    decoded_success = 0
    decoded_failed = 0

    for f in parquet_files:
        table = pq.read_table(f, columns=["image"]).slice(0, 30)
        for i in range(len(table)):
            img_val = table.column("image")[i].as_py()
            if isinstance(img_val, dict) and "bytes" in img_val:
                try:
                    img = Image.open(io.BytesIO(img_val["bytes"]))
                    img.verify()
                    decoded_success += 1
                except Exception:
                    decoded_failed += 1
            else:
                decoded_failed += 1

    print(f"  - Successfully decoded: {decoded_success}")
    print(f"  - Failed / Corrupted  : {decoded_failed}")

    print("\n" + "=" * 80)
    print("           STEP 2: ARTIST-AWARE SPLIT VALIDATION & DATA LEAKAGE AUDIT")
    print("=" * 80)

    # Initialize WikiArtDataset on 1,500 samples for split verification
    ds = WikiArtDataset(
        data_dir="wikiart-all",
        target_field="style",
        min_samples_per_class=50,
        fast_dev_mode=True,
        fast_dev_samples=1500,
        random_seed=42
    )

    train_ds, val_ds, test_ds = create_artist_aware_splits(
        ds,
        val_ratio=0.15,
        test_ratio=0.15,
        artist_aware=True,
        random_seed=42
    )

    train_df = ds.metadata_df.iloc[train_ds.indices]
    val_df = ds.metadata_df.iloc[val_ds.indices]
    test_df = ds.metadata_df.iloc[test_ds.indices]

    print(f"\nSample Counts:")
    print(f"  - Train set : {len(train_ds):>5} samples ({len(train_ds)/len(ds)*100:.1f}%)")
    print(f"  - Val set   : {len(val_ds):>5} samples ({len(val_ds)/len(ds)*100:.1f}%)")
    print(f"  - Test set  : {len(test_ds):>5} samples ({len(test_ds)/len(ds)*100:.1f}%)")

    train_artists = set(train_df["artist_label"])
    val_artists = set(val_df["artist_label"])
    test_artists = set(test_df["artist_label"])

    print(f"\nUnique Artists in each split:")
    print(f"  - Unique Train Artists : {len(train_artists)}")
    print(f"  - Unique Val Artists   : {len(val_artists)}")
    print(f"  - Unique Test Artists  : {len(test_artists)}")

    # Leakage checks
    inter_train_val = train_artists.intersection(val_artists)
    inter_train_test = train_artists.intersection(test_artists)
    inter_val_test = val_artists.intersection(test_artists)

    print(f"\nArtist Disjointness Leakage Check:")
    print(f"  - intersection(train_artists, val_artists)  = {len(inter_train_val)} (Must be 0) -> {'[PASS]' if len(inter_train_val)==0 else '[FAIL]'}")
    print(f"  - intersection(train_artists, test_artists) = {len(inter_train_test)} (Must be 0) -> {'[PASS]' if len(inter_train_test)==0 else '[FAIL]'}")
    print(f"  - intersection(val_artists, test_artists)   = {len(inter_val_test)} (Must be 0) -> {'[PASS]' if len(inter_val_test)==0 else '[FAIL]'}")

    # Image ID uniqueness check
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])

    id_overlap_tv = train_ids.intersection(val_ids)
    id_overlap_tt = train_ids.intersection(test_ids)
    id_overlap_vt = val_ids.intersection(test_ids)

    print(f"\nImage ID Duplication Check:")
    print(f"  - Overlap(train_ids, val_ids)  = {len(id_overlap_tv)} -> {'[PASS]' if len(id_overlap_tv)==0 else '[FAIL]'}")
    print(f"  - Overlap(train_ids, test_ids) = {len(id_overlap_tt)} -> {'[PASS]' if len(id_overlap_tt)==0 else '[FAIL]'}")
    print(f"  - Overlap(val_ids, test_ids)   = {len(id_overlap_vt)} -> {'[PASS]' if len(id_overlap_vt)==0 else '[FAIL]'}")

    print("\n" + "=" * 80)
    print("           STEP 3: DETERMINISTIC LABEL MAPPING & RANDOM SAMPLES")
    print("=" * 80)

    label_mapping = {
        "num_classes": ds.num_classes,
        "target_field": ds.target_field,
        "label_to_idx": ds.label_to_idx,
        "idx_to_label": ds.idx_to_label
    }
    
    mapping_path = Path("experiments/label_mapping.json")
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Saved deterministic label mapping to: {mapping_path}")

    # Print 20 random samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(ds), size=20, replace=False)

    print(f"\n{'#':<3} | {'Image ID':<26} | {'Artist':<22} | {'Style':<20} | {'Assigned Label':<20} | {'Idx':<4}")
    print("-" * 105)

    for i, idx in enumerate(sample_indices):
        _, target_idx, meta = ds[idx]
        print(f"{i+1:<3} | {meta['id']:<26} | {meta['artistName'][:20]:<22} | {meta['style'][:18]:<20} | {meta['label_name'][:18]:<20} | {target_idx:<4}")

    print("=" * 80)

if __name__ == "__main__":
    run_dataset_and_split_audit()
