from typing import Dict, Any

class ScientificRewardEngine:
    """
    Computes mathematically rigorous, multi-objective scientific reward for RL experiment selection:
    Reward = w1*InfoGain + w2*Novelty + w3*AccuracyDelta - w4*Cost - w5*Redundancy
    """
    def __init__(
        self,
        w_info_gain: float = 1.0,
        w_novelty: float = 0.5,
        w_accuracy_delta: float = 1.0,
        w_cost: float = 0.2,
        w_redundancy: float = 0.5
    ):
        self.w_info_gain = w_info_gain
        self.w_novelty = w_novelty
        self.w_accuracy_delta = w_accuracy_delta
        self.w_cost = w_cost
        self.w_redundancy = w_redundancy

    def calculate_reward(
        self,
        evidence: Dict[str, Any],
        experiment_config: Dict[str, Any],
        hypothesis: Dict[str, Any],
        is_redundant: bool = False
    ) -> Dict[str, float]:
        """Calculates total scalar reward and detailed component breakdown."""
        # 1. Information Gain: measured by cross-validation R² or significance
        r2 = max(0.0, float(evidence.get("metric_value", 0.0)))
        info_gain = r2
        
        # 2. Novelty
        novelty = float(hypothesis.get("novelty_score", 0.5))
        
        # 3. Accuracy Delta / Improvement
        verdict = evidence.get("support_verdict", "requires more experiments")
        if verdict == "supported":
            acc_delta = 1.0
        elif verdict == "weak evidence":
            acc_delta = 0.4
        elif verdict == "rejected":
            acc_delta = -0.2 # small penalty for failed hypotheses
        else:
            acc_delta = 0.0
            
        # 4. Computational Cost Penalty
        cost = float(experiment_config.get("computational_cost", 1.0))
        
        # 5. Redundancy Penalty
        redundancy = 1.0 if is_redundant else 0.0
        
        total_reward = (
            (self.w_info_gain * info_gain) +
            (self.w_novelty * novelty) +
            (self.w_accuracy_delta * acc_delta) -
            (self.w_cost * (cost / 2.0)) -
            (self.w_redundancy * redundancy)
        )
        
        return {
            "total_reward": round(float(total_reward), 4),
            "info_gain_component": round(self.w_info_gain * info_gain, 4),
            "novelty_component": round(self.w_novelty * novelty, 4),
            "accuracy_component": round(self.w_accuracy_delta * acc_delta, 4),
            "cost_penalty": round(self.w_cost * (cost / 2.0), 4),
            "redundancy_penalty": round(self.w_redundancy * redundancy, 4)
        }
