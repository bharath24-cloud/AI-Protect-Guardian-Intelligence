import copy
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, Dataset
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CLIP normalization parameters
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

def get_transforms(
    image_size: int = 224,
    is_training: bool = True,
    model_name: str = "resnet50",
    augment: bool = True,
    augmentation_level: str = "enhanced"
) -> transforms.Compose:
    """
    Builds data transforms for training, validation, and adversarial inference.
    Supports 'basic' and 'enhanced' artwork-preserving data augmentation.
    """
    model_lower = model_name.lower()
    if "clip" in model_lower:
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    if is_training and augment:
        if augmentation_level == "basic":
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            # Enhanced artwork-preserving augmentation
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.85, 1.18)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.RandomGrayscale(p=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
    else:
        # Deterministic validation and test transforms
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # ~256 for 224
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def print_transforms_summary(train_tf: transforms.Compose, eval_tf: transforms.Compose):
    """Prints exact training and validation transforms."""
    print("\n" + "=" * 60)
    print("DATA TRANSFORMS & AUGMENTATION PIPELINE")
    print("=" * 60)
    print("TRAINING TRANSFORMS:")
    for i, t in enumerate(train_tf.transforms, 1):
        print(f"  {i}. {t.__class__.__name__}: {t}")
    print("\nVALIDATION / TEST TRANSFORMS (DETERMINISTIC):")
    for i, t in enumerate(eval_tf.transforms, 1):
        print(f"  {i}. {t.__class__.__name__}: {t}")
    print("=" * 60 + "\n")


def get_inverse_transform(model_name: str = "resnet50") -> transforms.Compose:
    """
    Un-normalizes a normalized tensor back to [0, 1] RGB for visual display.
    """
    model_lower = model_name.lower()
    if "clip" in model_lower:
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1.0 / s for s in std]

    return transforms.Compose([
        transforms.Normalize(mean=inv_mean, std=inv_std),
    ])


class TransformedSubset(Dataset):
    """
    A high-performance Dataset wrapper around a torch Dataset subset that applies transforms.
    """
    def __init__(self, dataset: Dataset, indices: list, transform: Optional[transforms.Compose] = None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        original_idx = self.indices[idx]
        row = self.dataset.metadata_df.iloc[original_idx]
        file_idx = int(row["_file_idx"])
        rg_idx = int(row["_rg_idx"])
        row_in_rg = int(row["_row_in_rg"])
        label_idx = int(row["target_idx"])

        import io
        from PIL import Image
        try:
            raw_bytes = self.dataset._read_image_bytes(file_idx, rg_idx, row_in_rg)
            pil_img = Image.open(io.BytesIO(raw_bytes))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
        except Exception:
            pil_img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.transform is not None:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = pil_img

        meta = {
            "id": row.get("id", ""),
            "title": row.get("title", ""),
            "artistName": row.get("artistName", "Unknown"),
            "artistId": row.get("artistId", ""),
            "style": row.get("style_label", ""),
            "genre": row.get("genre_label", ""),
            "label_name": self.dataset.idx_to_label[label_idx],
            "width": int(row.get("width", 0)) if pd.notna(row.get("width")) else 0,
            "height": int(row.get("height", 0)) if pd.notna(row.get("height")) else 0,
        }

        return img_tensor, label_idx, meta


def stratified_artist_disjoint_partition(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, set, set, set]:
    """
    Stratified Group Partitioning Algorithm:
    Partitions artists into Train, Val, and Test splits such that:
      1. Train_artists ∩ Val_artists == ∅
      2. Train_artists ∩ Test_artists == ∅
      3. Val_artists ∩ Test_artists == ∅
      4. Maximizes style representation across all three splits while respecting artist-level grouping.
    """
    from collections import defaultdict
    rng = np.random.RandomState(random_seed)

    artist_to_styles = defaultdict(list)
    artist_to_samples = defaultdict(int)
    for _, row in df.iterrows():
        art = row["artist_label"]
        stl = row["style_label"]
        artist_to_styles[art].append(stl)
        artist_to_samples[art] += 1

    style_to_artists = defaultdict(set)
    for art, stls in artist_to_styles.items():
        for stl in set(stls):
            style_to_artists[stl].add(art)

    # Sort styles by rarity (least unique artists first)
    sorted_styles = sorted(style_to_artists.keys(), key=lambda s: len(style_to_artists[s]))

    train_artists = set()
    val_artists = set()
    test_artists = set()
    assigned_artists = set()

    train_target_samples = int(len(df) * (1.0 - val_ratio - test_ratio))
    val_target_samples = int(len(df) * val_ratio)
    test_target_samples = int(len(df) * test_ratio)

    train_current_samples = 0
    val_current_samples = 0
    test_current_samples = 0

    # Phase 1: Stratified allocation of unassigned artists per style (from rarest to common)
    for stl in sorted_styles:
        stl_artists = list(style_to_artists[stl])
        has_train = any(a in train_artists for a in stl_artists)
        has_val = any(a in val_artists for a in stl_artists)
        has_test = any(a in test_artists for a in stl_artists)

        unassigned_stl_artists = [a for a in stl_artists if a not in assigned_artists]
        if not unassigned_stl_artists:
            continue

        unassigned_stl_artists.sort(key=lambda a: artist_to_samples[a])

        # Priority 1: Guarantee Train coverage first
        if not has_train and len(unassigned_stl_artists) > 0:
            a_tr = unassigned_stl_artists.pop(0)
            train_artists.add(a_tr)
            assigned_artists.add(a_tr)
            train_current_samples += artist_to_samples[a_tr]
            has_train = True

        # Priority 2: Guarantee Test coverage
        if not has_test and len(unassigned_stl_artists) > 0:
            a_te = unassigned_stl_artists.pop(0)
            test_artists.add(a_te)
            assigned_artists.add(a_te)
            test_current_samples += artist_to_samples[a_te]
            has_test = True

        # Priority 3: Guarantee Val coverage
        if not has_val and len(unassigned_stl_artists) > 0:
            a_va = unassigned_stl_artists.pop(0)
            val_artists.add(a_va)
            assigned_artists.add(a_va)
            val_current_samples += artist_to_samples[a_va]
            has_val = True

        # For remaining artists in this style, greedily balance sample count ratios
        for a in unassigned_stl_artists:
            r_tr = train_current_samples / max(1, train_target_samples)
            r_va = val_current_samples / max(1, val_target_samples)
            r_te = test_current_samples / max(1, test_target_samples)

            least_filled = min((r_tr, "tr"), (r_va, "va"), (r_te, "te"))[1]
            if least_filled == "tr":
                train_artists.add(a)
                train_current_samples += artist_to_samples[a]
            elif least_filled == "va":
                val_artists.add(a)
                val_current_samples += artist_to_samples[a]
            else:
                test_artists.add(a)
                test_current_samples += artist_to_samples[a]
            assigned_artists.add(a)

    # Phase 2: Any remaining unassigned artists
    all_artists = list(artist_to_styles.keys())
    remaining = [a for a in all_artists if a not in assigned_artists]
    remaining.sort(key=lambda a: artist_to_samples[a], reverse=True)

    for a in remaining:
        r_tr = train_current_samples / max(1, train_target_samples)
        r_va = val_current_samples / max(1, val_target_samples)
        r_te = test_current_samples / max(1, test_target_samples)
        least_filled = min((r_tr, "tr"), (r_va, "va"), (r_te, "te"))[1]
        if least_filled == "tr":
            train_artists.add(a)
            train_current_samples += artist_to_samples[a]
        elif least_filled == "va":
            val_artists.add(a)
            val_current_samples += artist_to_samples[a]
        else:
            test_artists.add(a)
            test_current_samples += artist_to_samples[a]
        assigned_artists.add(a)

    # Map back to row indices
    train_indices = df[df["artist_label"].isin(train_artists)].index.values
    val_indices = df[df["artist_label"].isin(val_artists)].index.values
    test_indices = df[df["artist_label"].isin(test_artists)].index.values

    return train_indices, val_indices, test_indices, train_artists, val_artists, test_artists


def create_artist_aware_splits(
    dataset: Dataset,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    image_size: int = 224,
    model_name: str = "resnet50",
    artist_aware: bool = True,
    random_seed: int = 42,
    augmentation_level: str = "enhanced",
    split_cache_dir: Optional[str] = None
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Creates train, validation, and test datasets with strict artist-aware disjoint partitioning.
    Optionally persists and reloads splits from disk to ensure 100% reproducible experiments.

    Guarantees:
      train_artists ∩ val_artists = 0
      train_artists ∩ test_artists = 0
      val_artists ∩ test_artists = 0
    """
    import json
    import pandas as pd
    df = dataset.metadata_df
    n_samples = len(df)

    split_loaded = False
    if split_cache_dir is not None:
        split_path = Path(split_cache_dir)
        train_idx_file = split_path / "train_indices.json"
        val_idx_file = split_path / "val_indices.json"
        test_idx_file = split_path / "test_indices.json"

        if train_idx_file.exists() and val_idx_file.exists() and test_idx_file.exists():
            with open(train_idx_file, "r") as f:
                train_idx = np.array(json.load(f))
            with open(val_idx_file, "r") as f:
                val_idx = np.array(json.load(f))
            with open(test_idx_file, "r") as f:
                test_idx = np.array(json.load(f))
            
            # Verify indices are within current dataset bounds
            if len(train_idx) + len(val_idx) + len(test_idx) == n_samples and max(max(train_idx), max(val_idx), max(test_idx)) < n_samples:
                print(f"[Splits] Loaded persisted artist-aware split from: {split_path}")
                split_loaded = True
            else:
                existing_total = len(train_idx) + len(val_idx) + len(test_idx)
                print(f"[Splits] WARNING: Cached split in '{split_path}' has {existing_total} indices, "
                      f"which does not match current active dataset ({n_samples} samples). "
                      f"Preserving existing split directory without overwriting.")
                # Prevent silent overwriting of a valid existing split
                split_cache_dir = None

    if not split_loaded:
        if artist_aware and "artist_label" in df.columns and df["artist_label"].nunique() > 10:
            train_idx, val_idx, test_idx, train_artists, val_artists, test_artists = stratified_artist_disjoint_partition(
                df, val_ratio=val_ratio, test_ratio=test_ratio, random_seed=random_seed
            )
        else:
            # Fallback to stratified or standard shuffle split
            indices = np.arange(n_samples)
            targets = df["target_idx"].values
            try:
                sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
                train_val_idx, test_idx = next(sss_test.split(indices, targets))

                val_rel_ratio = val_ratio / (1.0 - test_ratio)
                sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_rel_ratio, random_state=random_seed + 1)
                train_sub_idx, val_sub_idx = next(sss_val.split(train_val_idx, targets[train_val_idx]))

                train_idx = train_val_idx[train_sub_idx]
                val_idx = train_val_idx[val_sub_idx]
            except Exception:
                from sklearn.model_selection import ShuffleSplit
                ss_test = ShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
                train_val_idx, test_idx = next(ss_test.split(indices))

                val_rel_ratio = val_ratio / (1.0 - test_ratio)
                ss_val = ShuffleSplit(n_splits=1, test_size=val_rel_ratio, random_state=random_seed + 1)
                train_sub_idx, val_sub_idx = next(ss_val.split(train_val_idx))

                train_idx = train_val_idx[train_sub_idx]
                val_idx = train_val_idx[val_sub_idx]

        # Persist splits if directory requested
        if split_cache_dir is not None:
            split_path = Path(split_cache_dir)
            split_path.mkdir(parents=True, exist_ok=True)
            with open(split_path / "train_indices.json", "w") as f:
                json.dump(train_idx.tolist(), f)
            with open(split_path / "val_indices.json", "w") as f:
                json.dump(val_idx.tolist(), f)
            with open(split_path / "test_indices.json", "w") as f:
                json.dump(test_idx.tolist(), f)

            # Also save human-readable CSVs with metadata
            df.iloc[train_idx].to_csv(split_path / "train.csv", index=False)
            df.iloc[val_idx].to_csv(split_path / "val.csv", index=False)
            df.iloc[test_idx].to_csv(split_path / "test.csv", index=False)
            print(f"[Splits] Saved artist-aware split files to: {split_path}")

    # Automated Integrity Assertions
    train_artists = set(df.iloc[train_idx]["artist_label"])
    val_artists = set(df.iloc[val_idx]["artist_label"])
    test_artists = set(df.iloc[test_idx]["artist_label"])

    overlap_tv = len(train_artists.intersection(val_artists))
    overlap_tt = len(train_artists.intersection(test_artists))
    overlap_vt = len(val_artists.intersection(test_artists))
    leakage_pass = (overlap_tv == 0 and overlap_tt == 0 and overlap_vt == 0)

    # Image index disjointness check
    idx_overlap_tv = len(set(train_idx).intersection(set(val_idx)))
    idx_overlap_tt = len(set(train_idx).intersection(set(test_idx)))
    idx_overlap_vt = len(set(val_idx).intersection(set(test_idx)))
    idx_disjoint = (idx_overlap_tv == 0 and idx_overlap_tt == 0 and idx_overlap_vt == 0)
    total_idx_count = len(train_idx) + len(val_idx) + len(test_idx)

    # Class coverage analysis
    all_styles = set(df["style_label"]) if "style_label" in df.columns else set()
    tr_styles = set(df.iloc[train_idx]["style_label"]) if "style_label" in df.columns else set()
    va_styles = set(df.iloc[val_idx]["style_label"]) if "style_label" in df.columns else set()
    te_styles = set(df.iloc[test_idx]["style_label"]) if "style_label" in df.columns else set()

    missing_te = all_styles - te_styles
    missing_va = all_styles - va_styles
    missing_tr = all_styles - tr_styles

    print("\n" + "=" * 65)
    print("AUTOMATED DATASET SPLIT INTEGRITY VERIFICATION")
    print("=" * 65)
    print(f"Total dataset samples : {n_samples:,}")
    print(f"Train samples         : {len(train_idx):,} ({len(train_idx)/n_samples*100:.1f}%) | Artists: {len(train_artists):,}")
    print(f"Validation samples    : {len(val_idx):,} ({len(val_idx)/n_samples*100:.1f}%) | Artists: {len(val_artists):,}")
    print(f"Test samples          : {len(test_idx):,} ({len(test_idx)/n_samples*100:.1f}%) | Artists: {len(test_artists):,}")
    print(f"\nArtist Overlap Check  : Train_inter_Val={overlap_tv}, Train_inter_Test={overlap_tt}, Val_inter_Test={overlap_vt} -> {'PASS' if leakage_pass else 'FAIL'}")
    print(f"Image Overlap Check   : Train_inter_Val={idx_overlap_tv}, Train_inter_Test={idx_overlap_tt}, Val_inter_Test={idx_overlap_vt} -> {'PASS' if idx_disjoint else 'FAIL'}")
    print(f"Total Index Coverage  : {total_idx_count:,} / {n_samples:,} -> {'PASS' if total_idx_count == n_samples else 'FAIL'}")
    print(f"\nActive Styles Coverage: Total={len(all_styles)} | Train={len(tr_styles)} | Val={len(va_styles)} | Test={len(te_styles)}")
    if missing_te:
        print(f"  Note: Styles absent in Test ({len(missing_te)}): {sorted(list(missing_te))} (Single-artist classes exclusively in Train)")
    if missing_va:
        print(f"  Note: Styles absent in Val  ({len(missing_va)}): {sorted(list(missing_va))} (<=2 artist classes)")
    if missing_tr:
        print(f"  WARNING: Styles absent in Train ({len(missing_tr)}): {sorted(list(missing_tr))}")
    print("=" * 65 + "\n")

    if not leakage_pass:
        raise RuntimeError("CRITICAL FAILURE: Artist leakage detected between dataset splits!")
    if not idx_disjoint or total_idx_count != n_samples:
        raise RuntimeError("CRITICAL FAILURE: Image index mismatch or duplicate images detected between splits!")
    if len(missing_tr) > 0:
        raise RuntimeError(f"CRITICAL FAILURE: Styles {missing_tr} have 0 training samples!")

    train_tf = get_transforms(image_size=image_size, is_training=True, model_name=model_name, augment=True, augmentation_level=augmentation_level)
    eval_tf = get_transforms(image_size=image_size, is_training=False, model_name=model_name, augment=False)

    print_transforms_summary(train_tf, eval_tf)

    train_dataset = TransformedSubset(dataset, train_idx.tolist(), transform=train_tf)
    val_dataset = TransformedSubset(dataset, val_idx.tolist(), transform=eval_tf)
    test_dataset = TransformedSubset(dataset, test_idx.tolist(), transform=eval_tf)

    # Perform and print Artist Leakage Check
    train_artists = set(df.iloc[train_idx]["artist_label"])
    val_artists = set(df.iloc[val_idx]["artist_label"])
    test_artists = set(df.iloc[test_idx]["artist_label"])

    overlap_tv = len(train_artists.intersection(val_artists))
    overlap_tt = len(train_artists.intersection(test_artists))
    overlap_vt = len(val_artists.intersection(test_artists))

    leakage_pass = (overlap_tv == 0 and overlap_tt == 0 and overlap_vt == 0)

    print("\n" + "=" * 50)
    print("ARTIST LEAKAGE CHECK")
    print("=" * 50)
    print(f"Train samples       : {len(train_dataset)} ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"Validation samples  : {len(val_dataset)} ({len(val_dataset)/len(dataset)*100:.1f}%)")
    print(f"Test samples        : {len(test_dataset)} ({len(test_dataset)/len(dataset)*100:.1f}%)")
    print(f"\nTrain artists       : {len(train_artists)}")
    print(f"Validation artists  : {len(val_artists)}")
    print(f"Test artists        : {len(test_artists)}")
    print(f"\nTrain inter Validation  : {overlap_tv}")
    print(f"Train inter Test        : {overlap_tt}")
    print(f"Validation inter Test   : {overlap_vt}")
    print(f"\nArtist leakage      : {'PASS' if leakage_pass else 'FAIL'}")
    print("=" * 50 + "\n")

    if not leakage_pass:
        raise RuntimeError("CRITICAL: Artist leakage detected between dataset splits!")

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    from src.dataset import WikiArtDataset
    ds = WikiArtDataset(fast_dev_mode=True, fast_dev_samples=200)
    train_ds, val_ds, test_ds = create_artist_aware_splits(ds, artist_aware=True)
    img, lbl, meta = train_ds[0]
    print(f"[Test] Sample train tensor shape: {img.shape}, Label: {lbl}")
