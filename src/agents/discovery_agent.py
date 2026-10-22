from typing import Dict, Any, List, Optional
import numpy as np
from src.discovery.discovery_engine import DiscoveryEngine

class DiscoveryAgent:
    """Agent responsible for mining representations and detecting candidate physical patterns."""
    def __init__(self, discovery_engine: Optional[DiscoveryEngine] = None):
        self.engine = discovery_engine or DiscoveryEngine()

    def analyze(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: List[str],
        target_name: str = "band_gap",
        representations: Optional[np.ndarray] = None,
        material_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Scans the materials space and returns structured candidate discoveries."""
        res = self.engine.discover(
            features=features,
            targets=targets,
            feature_names=feature_names,
            target_name=target_name,
            representations=representations,
            material_ids=material_ids
        )
        return res["discoveries"]
