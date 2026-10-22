import uuid
from typing import Dict, Any, List, Optional

class HypothesisAgent:
    """Converts observational patterns and correlations into explicit, falsifiable scientific hypotheses."""
    def __init__(self, default_confidence: float = 0.5):
        self.default_confidence = default_confidence

    def generate_hypotheses(self, discoveries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transforms discovery items into formal hypotheses."""
        hypotheses = []
        
        for d in discoveries:
            hyp_id = f"hyp-{uuid.uuid4().hex[:8]}"
            vars_list = d.get("variables", [])
            strength = float(d.get("strength", 0.5))
            novelty = float(d.get("novelty", 0.5))
            
            if d.get("category") == "analytical_relationship":
                statement = f"The physical property {vars_list[-1]} strictly obeys the analytical relationship: {d.get('formula_str', '')}."
                expected_rel = "analytical_power_law"
            else:
                statement = f"Higher {vars_list[0]} deterministically influences {vars_list[-1]} across stoichiometric compositions."
                expected_rel = "monotonic_correlation"
                
            hypotheses.append({
                "hypothesis_id": hyp_id,
                "statement": statement,
                "variables": vars_list,
                "expected_relationship": expected_rel,
                "evidence": d.get("support", []),
                "confidence": round(float(np_clip(0.3 + 0.4 * strength, 0.1, 0.9)), 3),
                "novelty_score": round(novelty, 3),
                "testability_score": 0.95,
                "status": "proposed"
            })
            
        return hypotheses

def np_clip(val, low, high):
    return max(low, min(high, val))
