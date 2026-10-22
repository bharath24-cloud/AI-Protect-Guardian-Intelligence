from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

class ClusteringEngine:
    """Discovers unsupervised clusters and density groupings in scientific representations."""
    def __init__(self, n_clusters: int = 4, random_seed: int = 42):
        self.n_clusters = n_clusters
        self.random_seed = random_seed

    def discover_clusters(self, representations: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Applies K-Means clustering and computes cluster statistics."""
        if len(representations) < self.n_clusters:
            return {"clusters_found": 0, "labels": [], "silhouette": 0.0, "profiles": []}
            
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_seed, n_init=10)
        labels = kmeans.fit_predict(representations)
        
        try:
            sil = float(silhouette_score(representations, labels))
        except Exception:
            sil = 0.0
            
        profiles = []
        for c_id in range(self.n_clusters):
            c_mask = (labels == c_id)
            c_size = int(np.sum(c_mask))
            c_center = kmeans.cluster_centers_[c_id].tolist()
            profiles.append({
                "cluster_id": c_id,
                "size": c_size,
                "fraction": float(c_size / len(representations)),
                "center_summary": c_center[:4]
            })
            
        return {
            "clusters_found": self.n_clusters,
            "labels": labels.tolist(),
            "silhouette": sil,
            "profiles": profiles
        }
