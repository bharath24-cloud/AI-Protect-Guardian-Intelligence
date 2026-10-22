import uuid
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.discovery.clustering import ClusteringEngine
from src.discovery.correlation import CorrelationMiner
from src.discovery.anomaly_detection import AnomalyDetector
from src.discovery.symbolic_mining import SymbolicRelationshipMiner

class DiscoveryEngine:
    """
    Unified Pattern Discovery Engine orchestrating unsupervised clustering,
    multi-scale correlation discovery, anomaly scoring, and symbolic relationship mining.
    """
    def __init__(
        self,
        min_correlation: float = 0.30,
        significance_alpha: float = 0.05,
        min_r2: float = 0.40,
        n_clusters: int = 4
    ):
        self.cluster_engine = ClusteringEngine(n_clusters=n_clusters)
        self.correlation_miner = CorrelationMiner(min_threshold=min_correlation, significance_alpha=significance_alpha)
        self.anomaly_detector = AnomalyDetector()
        self.symbolic_miner = SymbolicRelationshipMiner(min_r2=min_r2)

    def discover(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: List[str],
        target_name: str = "band_gap",
        representations: Optional[np.ndarray] = None,
        material_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Executes full discovery pipeline across feature and latent representation spaces."""
        
        # 1. Cluster Analysis
        rep_space = representations if representations is not None else features
        cluster_res = self.cluster_engine.discover_clusters(rep_space, feature_names)
        
        # 2. Correlation Mining
        correlations = self.correlation_miner.mine_relationships(
            features=features,
            targets=targets,
            feature_names=feature_names,
            target_name=target_name
        )
        
        # 3. Anomaly & Novelty Detection
        anomaly_res = self.anomaly_detector.detect_novel_materials(features, material_ids)
        
        # 4. Symbolic Mining
        equations = self.symbolic_miner.mine_equations(
            features=features,
            targets=targets,
            feature_names=feature_names,
            target_name=target_name
        )
        
        # Collate structured candidate discoveries
        structured_discoveries = []
        
        # From correlations
        for c in correlations:
            d_id = f"disc-{uuid.uuid4().hex[:8]}"
            p_val = float(c.get("p_value", 0.01))
            structured_discoveries.append({
                "discovery_id": d_id,
                "category": "statistical_correlation",
                "description": c["description"],
                "variables": [c["feature"], c["target"]],
                "strength": float(c["strength"]),
                "novelty": float(0.5 + 0.5 * (1.0 - p_val)),
                "support": [{"metric": "p_value", "value": p_val}],
                "possible_confounders": ["atomic_radius", "valence_electron_density"]
            })
            
        # From symbolic equations
        for eq in equations:
            d_id = f"disc-{uuid.uuid4().hex[:8]}"
            feats = eq.get("features", [eq.get("feature", "unknown")])
            structured_discoveries.append({
                "discovery_id": d_id,
                "category": "analytical_relationship",
                "description": eq["description"],
                "formula_str": eq["formula_str"],
                "variables": feats + [eq["target"]],
                "strength": float(eq["r2_score"]),
                "novelty": float(np.clip(eq["r2_score"] * 1.1, 0.0, 1.0)),
                "support": [{"metric": "r2_score", "value": eq["r2_score"]}],
                "possible_confounders": ["chemical_system_class", "stoichiometric_ratio"]
            })
            
        return {
            "total_discoveries": len(structured_discoveries),
            "discoveries": structured_discoveries,
            "clustering": cluster_res,
            "anomalies": anomaly_res
        }
