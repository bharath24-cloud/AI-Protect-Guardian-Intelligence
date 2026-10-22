from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    """Detects physical outliers and unexplored high-novelty regions in materials parameter space."""
    def __init__(self, contamination: float = 0.05, random_seed: int = 42):
        self.contamination = contamination
        self.random_seed = random_seed
        self.model = IsolationForest(contamination=contamination, random_state=random_seed)

    def detect_novel_materials(
        self,
        features: np.ndarray,
        material_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Identifies outlier materials with high novelty scores."""
        preds = self.model.fit_predict(features) # -1 for anomaly, 1 for normal
        scores = self.model.decision_function(features) # lower score -> more anomalous
        
        # Invert score so higher value = higher novelty [0, 1]
        novelty_scores = 1.0 - (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
        anomaly_indices = np.where(preds == -1)[0]
        
        anomalies = []
        for idx in anomaly_indices:
            mat_id = material_ids[idx] if material_ids is not None and idx < len(material_ids) else f"sample_{idx}"
            anomalies.append({
                "index": int(idx),
                "material_id": mat_id,
                "novelty_score": float(novelty_scores[idx])
            })
            
        anomalies = sorted(anomalies, key=lambda x: x["novelty_score"], reverse=True)
        return {
            "total_anomalies": len(anomaly_indices),
            "anomaly_indices": anomaly_indices.tolist(),
            "novelty_scores": novelty_scores.tolist(),
            "top_novel_materials": anomalies[:10]
        }
