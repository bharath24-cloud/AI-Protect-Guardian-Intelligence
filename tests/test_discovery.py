import pytest
import numpy as np
from src.discovery.discovery_engine import DiscoveryEngine
from src.datasets.materials_data import MaterialsScienceDataset

def test_discovery_engine():
    ds = MaterialsScienceDataset(num_samples=150, random_seed=42)
    feats = ds.get_features()
    targets = ds.get_targets("band_gap")
    feat_names = ds.get_feature_names()
    
    engine = DiscoveryEngine(min_correlation=0.25, min_r2=0.30)
    res = engine.discover(
        features=feats,
        targets=targets,
        feature_names=feat_names,
        target_name="band_gap"
    )
    
    assert "discoveries" in res
    assert "clustering" in res
    assert "anomalies" in res
    assert len(res["discoveries"]) > 0
    assert res["clustering"]["clusters_found"] == 4
