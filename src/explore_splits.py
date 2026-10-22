import json
import os
import sys
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from src.dataset import WikiArtDataset

def stratified_artist_disjoint_split(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Stratified Group Partitioning Algorithm:
    Partitions artists into Train, Val, and Test splits such that:
      1. Train_artists ∩ Val_artists == ∅
      2. Train_artists ∩ Test_artists == ∅
      3. Val_artists ∩ Test_artists == ∅
      4. Maximizes style representation across all three splits.
    """
    rng = np.random.RandomState(random_seed)
    
    # Map artists to their style distributions
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
            
        # Sort unassigned artists by sample count ascending so rare artists fill eval slots first
        unassigned_stl_artists.sort(key=lambda a: artist_to_samples[a])
        
        # Priority 1: Guarantee Train coverage
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

if __name__ == "__main__":
    ds = WikiArtDataset(target_field="style", min_samples_per_class=50, data_fraction=0.25, random_seed=42)
    df = ds.metadata_df
    
    train_idx, val_idx, test_idx, tr_art, va_art, te_art = stratified_artist_disjoint_split(df, random_seed=42)
    
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_idx)} ({len(train_idx)/len(df)*100:.1f}%), Artists: {len(tr_art)}")
    print(f"Val samples:   {len(val_idx)} ({len(val_idx)/len(df)*100:.1f}%), Artists: {len(va_art)}")
    print(f"Test samples:  {len(test_idx)} ({len(test_idx)/len(df)*100:.1f}%), Artists: {len(te_art)}")
    
    # Leakage check
    leak_tv = len(tr_art.intersection(va_art))
    leak_tt = len(tr_art.intersection(te_art))
    leak_vt = len(va_art.intersection(te_art))
    print(f"Artist Leakage: Train_inter_Val={leak_tv}, Train_inter_Test={leak_tt}, Val_inter_Test={leak_vt} -> {'PASS' if (leak_tv == 0 and leak_tt == 0 and leak_vt == 0) else 'FAIL'}")
    
    # Class coverage
    tr_styles = set(df.iloc[train_idx]["style_label"])
    va_styles = set(df.iloc[val_idx]["style_label"])
    te_styles = set(df.iloc[test_idx]["style_label"])
    all_styles = set(df["style_label"])
    
    print(f"\nTotal Active Styles: {len(all_styles)}")
    print(f"Styles in Train: {len(tr_styles)} / {len(all_styles)} (Missing: {len(all_styles - tr_styles)})")
    print(f"Styles in Val:   {len(va_styles)} / {len(all_styles)} (Missing: {len(all_styles - va_styles)})")
    print(f"Styles in Test:  {len(te_styles)} / {len(all_styles)} (Missing: {len(all_styles - te_styles)})")
    
    if len(all_styles - te_styles) > 0:
        print(f"Styles missing in Test: {sorted(list(all_styles - te_styles))}")
    if len(all_styles - va_styles) > 0:
        print(f"Styles missing in Val: {sorted(list(all_styles - va_styles))}")
    if len(all_styles - tr_styles) > 0:
        print(f"Styles missing in Train: {sorted(list(all_styles - tr_styles))}")
