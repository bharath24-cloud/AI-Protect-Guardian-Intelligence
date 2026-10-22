from typing import Dict, Any, List, Optional
import numpy as np
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_regression

class CorrelationMiner:
    """Discovers linear (Pearson), rank-order (Spearman), and non-linear (Mutual Info) correlations."""
    def __init__(self, min_threshold: float = 0.30, significance_alpha: float = 0.05):
        self.min_threshold = min_threshold
        self.significance_alpha = significance_alpha

    def mine_relationships(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: List[str],
        target_name: str = "band_gap"
    ) -> List[Dict[str, Any]]:
        """Scans feature-target pairs and extracts statistically significant candidate correlations."""
        discoveries = []
        n_features = features.shape[1]
        
        # Linear Pearson & Spearman
        for i in range(n_features):
            feat_col = features[:, i]
            feat_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            
            # Pearson
            r_pearson, p_pearson = stats.pearsonr(feat_col, targets)
            # Spearman
            r_spearman, p_spearman = stats.spearmanr(feat_col, targets)
            
            if abs(r_pearson) >= self.min_threshold and p_pearson < self.significance_alpha:
                discoveries.append({
                    "type": "linear_correlation",
                    "feature": feat_name,
                    "target": target_name,
                    "coefficient": float(r_pearson),
                    "p_value": float(p_pearson),
                    "strength": float(abs(r_pearson)),
                    "direction": "positive" if r_pearson > 0 else "negative",
                    "description": f"{feat_name} shows strong { 'positive' if r_pearson > 0 else 'negative' } linear correlation (r = {r_pearson:.3f}, p = {p_pearson:.2e}) with {target_name}."
                })
                
            if abs(r_spearman) >= self.min_threshold and p_spearman < self.significance_alpha and abs(r_spearman - r_pearson) > 0.15:
                discoveries.append({
                    "type": "monotonic_non_linear_correlation",
                    "feature": feat_name,
                    "target": target_name,
                    "coefficient": float(r_spearman),
                    "p_value": float(p_spearman),
                    "strength": float(abs(r_spearman)),
                    "direction": "positive" if r_spearman > 0 else "negative",
                    "description": f"{feat_name} demonstrates significant non-linear monotonic relationship (rho = {r_spearman:.3f}, p = {p_spearman:.2e}) with {target_name}."
                })
                
        # Non-linear Mutual Information
        try:
            mi_scores = mutual_info_regression(features, targets, random_state=42)
            for i, mi in enumerate(mi_scores):
                feat_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                if mi > 0.20:
                    discoveries.append({
                        "type": "mutual_information",
                        "feature": feat_name,
                        "target": target_name,
                        "mi_score": float(mi),
                        "strength": float(np.clip(mi, 0.0, 1.0)),
                        "description": f"High mutual information (MI = {mi:.3f} nats) detected between {feat_name} and {target_name}."
                    })
        except Exception:
            pass
            
        # Sort by strength descending
        discoveries = sorted(discoveries, key=lambda x: x["strength"], reverse=True)
        return discoveries
