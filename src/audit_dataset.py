import json
import os
import sys
import io
import glob
from pathlib import Path
from collections import Counter, defaultdict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from PIL import Image

from src.dataset import WikiArtDataset
from src.preprocessing import create_artist_aware_splits

def audit_full_dataset():
    out_dir = Path("experiments/results/accuracy_improvement")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_files = sorted(glob.glob("wikiart-all/data/*.parquet") + glob.glob("wikiart-all/*.parquet"))
    print(f"[Audit] Found {len(parquet_files)} parquet shards.")
    
    # 1. Read all metadata
    cols = ["id", "title", "artistName", "artistId", "genres", "styles", "media", "width", "height"]
    dfs = []
    for f in parquet_files:
        pf = pq.ParquetFile(f)
        table = pf.read(columns=cols)
        df = table.to_pandas()
        dfs.append(df)
    
    raw_df = pd.concat(dfs, ignore_index=True)
    total_raw_rows = len(raw_df)
    print(f"[Audit] Total raw rows: {total_raw_rows:,}")
    
    # Extract style, genre, artist
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

    raw_df["style_raw"] = raw_df["styles"].apply(extract_primary)
    raw_df["genre_raw"] = raw_df["genres"].apply(extract_primary)
    raw_df["artist_raw"] = raw_df["artistName"].fillna("Unknown").astype(str).str.strip()
    raw_df["id_raw"] = raw_df["id"].astype(str).str.strip()

    # Check duplicates and missing
    unique_ids = raw_df["id_raw"].nunique()
    duplicate_ids = total_raw_rows - unique_ids
    missing_styles = (raw_df["style_raw"] == "Unknown").sum()
    missing_artists = (raw_df["artist_raw"] == "Unknown").sum()
    unique_artists_raw = raw_df["artist_raw"].nunique()
    unique_styles_raw = raw_df["style_raw"].nunique()
    
    print(f"[Audit] Unique image IDs: {unique_ids:,} (Duplicates: {duplicate_ids})")
    print(f"[Audit] Missing/Unknown Styles: {missing_styles:,}")
    print(f"[Audit] Missing/Unknown Artists: {missing_artists:,}")
    print(f"[Audit] Total Unique Raw Styles: {unique_styles_raw}")
    print(f"[Audit] Total Unique Raw Artists: {unique_artists_raw}")

    # Inspect all raw style counts
    style_counts_all = raw_df[raw_df["style_raw"] != "Unknown"]["style_raw"].value_counts()
    print("\n[Audit] All Style Classes (Filtered vs Unfiltered count with min_samples=50):")
    valid_styles_50 = style_counts_all[style_counts_all >= 50].index.tolist()
    valid_styles_20 = style_counts_all[style_counts_all >= 20].index.tolist()
    valid_styles_10 = style_counts_all[style_counts_all >= 10].index.tolist()
    print(f"  Total non-Unknown styles: {len(style_counts_all)}")
    print(f"  Styles with >= 50 samples: {len(valid_styles_50)}")
    print(f"  Styles with >= 20 samples: {len(valid_styles_20)}")
    print(f"  Styles with >= 10 samples: {len(valid_styles_10)}")
    print(f"  Filtered rows with min_samples=50: {raw_df['style_raw'].isin(valid_styles_50).sum():,}")

    # 2. Inspect 25% dataset and split
    print("\n[Audit] Building 25% dataset with min_samples=50 and random_seed=42...")
    ds_25 = WikiArtDataset(
        data_dir="wikiart-all",
        target_field="style",
        min_samples_per_class=50,
        data_fraction=0.25,
        random_seed=42
    )
    
    split_cache_dir = "experiments/results/splits/split_25pct_v2"
    train_ds, val_ds, test_ds = create_artist_aware_splits(
        ds_25,
        val_ratio=0.15,
        test_ratio=0.15,
        artist_aware=True,
        random_seed=42,
        split_cache_dir=split_cache_dir
    )
    
    df_25 = ds_25.metadata_df
    train_indices = train_ds.indices
    val_indices = val_ds.indices
    test_indices = test_ds.indices
    
    train_df = df_25.iloc[train_indices]
    val_df = df_25.iloc[val_indices]
    test_df = df_25.iloc[test_indices]
    
    train_artists = set(train_df["artist_label"])
    val_artists = set(val_df["artist_label"])
    test_artists = set(test_df["artist_label"])
    
    leak_tv = len(train_artists.intersection(val_artists))
    leak_tt = len(train_artists.intersection(test_artists))
    leak_vt = len(val_artists.intersection(test_artists))
    
    print(f"[Audit Split] Train samples: {len(train_df):,}, Artists: {len(train_artists):,}")
    print(f"[Audit Split] Val samples:   {len(val_df):,}, Artists: {len(val_artists):,}")
    print(f"[Audit Split] Test samples:  {len(test_df):,}, Artists: {len(test_artists):,}")
    print(f"[Audit Split] Leakage check: Train_inter_Val={leak_tv}, Train_inter_Test={leak_tt}, Val_inter_Test={leak_vt}")
    
    # 3. Analyze class distribution and representation across splits
    all_classes = sorted(ds_25.idx_to_label.values())
    class_dist_rows = []
    
    train_counts = train_df["style_label"].value_counts().to_dict()
    val_counts = val_df["style_label"].value_counts().to_dict()
    test_counts = test_df["style_label"].value_counts().to_dict()
    
    train_artists_per_class = train_df.groupby("style_label")["artist_label"].nunique().to_dict()
    val_artists_per_class = val_df.groupby("style_label")["artist_label"].nunique().to_dict()
    test_artists_per_class = test_df.groupby("style_label")["artist_label"].nunique().to_dict()
    total_artists_per_class = df_25.groupby("style_label")["artist_label"].nunique().to_dict()
    
    zero_test_classes = []
    zero_val_classes = []
    
    for cls_name in all_classes:
        tot_samp = (df_25["style_label"] == cls_name).sum()
        tot_art = total_artists_per_class.get(cls_name, 0)
        
        tr_s = train_counts.get(cls_name, 0)
        va_s = val_counts.get(cls_name, 0)
        te_s = test_counts.get(cls_name, 0)
        
        tr_a = train_artists_per_class.get(cls_name, 0)
        va_a = val_artists_per_class.get(cls_name, 0)
        te_a = test_artists_per_class.get(cls_name, 0)
        
        if te_s == 0:
            zero_test_classes.append(cls_name)
        if va_s == 0:
            zero_val_classes.append(cls_name)
            
        class_dist_rows.append({
            "Style": cls_name,
            "Total samples": tot_samp,
            "Unique artists": tot_art,
            "Train samples": tr_s,
            "Validation samples": va_s,
            "Test samples": te_s,
            "Train artists": tr_a,
            "Validation artists": va_a,
            "Test artists": te_a
        })
        
    class_dist_df = pd.DataFrame(class_dist_rows)
    class_dist_df = class_dist_df.sort_values(by="Total samples", ascending=False).reset_index(drop=True)
    class_dist_csv_path = out_dir / "class_distribution.csv"
    class_dist_df.to_csv(class_dist_csv_path, index=False)
    print(f"[Audit] Saved class distribution to: {class_dist_csv_path}")
    
    print("\n[KEY FINDING: ZERO SUPPORT CLASSES IN TEST SET]")
    print(f"Classes with 0 Test samples ({len(zero_test_classes)}): {zero_test_classes}")
    print(f"Classes with 0 Val samples  ({len(zero_val_classes)}): {zero_val_classes}")
    
    # 4. Generate dataset audit JSON
    audit_summary = {
        "total_raw_parquet_rows": total_raw_rows,
        "parquet_shards": len(parquet_files),
        "unique_image_ids": unique_ids,
        "duplicate_image_ids": duplicate_ids,
        "missing_unknown_styles": int(missing_styles),
        "missing_unknown_artists": int(missing_artists),
        "total_raw_styles": unique_styles_raw,
        "total_raw_artists": unique_artists_raw,
        "min_samples_threshold": 50,
        "active_classes_count": len(all_classes),
        "total_filtered_samples_full": int(raw_df['style_raw'].isin(valid_styles_50).sum()),
        "dataset_25pct_samples": len(df_25),
        "split_samples": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "split_artists": {
            "train": len(train_artists),
            "val": len(val_artists),
            "test": len(test_artists)
        },
        "artist_leakage_check": {
            "train_val_overlap": leak_tv,
            "train_test_overlap": leak_tt,
            "val_test_overlap": leak_vt,
            "passed": (leak_tv == 0 and leak_tt == 0 and leak_vt == 0)
        },
        "zero_support_in_test": zero_test_classes,
        "zero_support_in_val": zero_val_classes,
        "class_imbalance": {
            "min_class_count": int(class_dist_df["Total samples"].min()),
            "max_class_count": int(class_dist_df["Total samples"].max()),
            "imbalance_ratio": float(class_dist_df["Total samples"].max() / max(1, class_dist_df["Total samples"].min()))
        },
        "evaluation_warning_cause_identified": (
            "Because artist-disjoint splitting is used and some rare styles have only 1-3 artists total, "
            "all artists for those rare styles were allocated exclusively to the train set (or val set), "
            "resulting in 0 true test samples for those classes. When the model predicts any of those classes "
            "on the test set, sklearn raises `UserWarning: y_pred contains classes not in y_true`."
        )
    }
    
    audit_json_path = out_dir / "dataset_audit.json"
    with open(audit_json_path, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"[Audit] Saved dataset audit JSON to: {audit_json_path}")
    print("\n" + "=" * 80)
    print(f"AUDIT COMPLETE. Active Classes: {len(all_classes)}, Imbalance Ratio: {audit_summary['class_imbalance']['imbalance_ratio']:.1f}:1")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    audit_full_dataset()
