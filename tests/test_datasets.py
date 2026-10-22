import pytest
import numpy as np
from src.datasets.materials_data import (
    MaterialsScienceDataset,
    parse_formula,
    compute_stoichiometric_features
)

def test_formula_parser():
    fracs = parse_formula("TiO2")
    assert "Ti" in fracs
    assert "O" in fracs
    assert round(fracs["Ti"], 3) == round(1.0 / 3.0, 3)
    assert round(fracs["O"], 3) == round(2.0 / 3.0, 3)

def test_stoichiometric_features():
    fracs = parse_formula("Fe2O3")
    feats = compute_stoichiometric_features(fracs)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (24,)
    assert not np.isnan(feats).any()

def test_materials_dataset_loading():
    ds = MaterialsScienceDataset(num_samples=100, random_seed=42)
    feats = ds.get_features()
    targets = ds.get_targets("band_gap")
    meta = ds.get_metadata()
    
    assert len(feats) == 100
    assert feats.shape[1] == 24
    assert len(targets) == 100
    assert len(meta) == 100
    assert "band_gap" in ds.get_target_names()
    assert "formation_energy_per_atom" in ds.get_target_names()
    assert "bulk_modulus" in ds.get_target_names()
