from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import r2_score

class SymbolicRelationshipMiner:
    """Discovers closed-form analytical equations and power-law scalings between features and targets."""
    def __init__(self, min_r2: float = 0.40):
        self.min_r2 = min_r2

    def mine_equations(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: List[str],
        target_name: str = "band_gap"
    ) -> List[Dict[str, Any]]:
        """Evaluates non-linear candidate scaling hypotheses (power-law, ratios, inverse squares)."""
        discoveries = []
        n_features = features.shape[1]
        
        # 1. Power Law single feature: y ~ a * x^b + c
        for i in range(n_features):
            x = features[:, i]
            feat_name = feature_names[i]
            
            # Avoid division by zero or negative powers on negative numbers
            if np.min(x) <= 0:
                x_pos = x - np.min(x) + 0.1
            else:
                x_pos = x
                
            for power in [0.5, 1.5, 2.0, -1.0, -0.5]:
                try:
                    transformed_x = x_pos ** power
                    poly = np.polyfit(transformed_x, targets, deg=1)
                    y_pred = np.polyval(poly, transformed_x)
                    r2 = float(r2_score(targets, y_pred))
                    
                    if r2 >= self.min_r2:
                        discoveries.append({
                            "type": "symbolic_power_law",
                            "feature": feat_name,
                            "target": target_name,
                            "power": power,
                            "coefficients": poly.tolist(),
                            "r2_score": r2,
                            "strength": r2,
                            "formula_str": f"{target_name} ≈ {poly[0]:.3f} * ({feat_name})^{power} + {poly[1]:.3f}",
                            "description": f"Analytical power-law relationship discovered: {target_name} scales with ({feat_name})^{power} with R² = {r2:.3f}."
                        })
                except Exception:
                    continue

        # 2. Ratio interactions: y ~ a * (x1 / (x2 + eps)) + b
        for i in range(min(6, n_features)):
            for j in range(i + 1, min(6, n_features)):
                x1 = features[:, i]
                x2 = features[:, j]
                f1_name = feature_names[i]
                f2_name = feature_names[j]
                
                ratio = (x1 + 1e-4) / (x2 + 1e-4)
                try:
                    poly = np.polyfit(ratio, targets, deg=1)
                    y_pred = np.polyval(poly, ratio)
                    r2 = float(r2_score(targets, y_pred))
                    if r2 >= self.min_r2 + 0.05:
                        discoveries.append({
                            "type": "symbolic_ratio_interaction",
                            "features": [f1_name, f2_name],
                            "target": target_name,
                            "r2_score": r2,
                            "strength": r2,
                            "formula_str": f"{target_name} ≈ {poly[0]:.3f} * ({f1_name} / {f2_name}) + {poly[1]:.3f}",
                            "description": f"Coupled interaction ratio discovered: {target_name} scales with ({f1_name} / {f2_name}) with R² = {r2:.3f}."
                        })
                except Exception:
                    continue

        discoveries = sorted(discoveries, key=lambda x: x["strength"], reverse=True)
        return discoveries[:10]
