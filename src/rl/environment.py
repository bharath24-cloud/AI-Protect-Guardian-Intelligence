from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class DiscoveryEnv:
    """
    Markov Decision Process Environment for Autonomous Scientific Discovery.
    State representation:
      [mean_conf, mean_uncertainty, mean_novelty, budget_ratio, hist_success_rate, expected_gain, queue_size, avg_cost]
    """
    def __init__(self, initial_budget: float = 50.0):
        self.initial_budget = initial_budget
        self.remaining_budget = initial_budget
        self.history_records: List[Dict[str, Any]] = []

    def get_state(self, candidate_hypotheses: List[Dict[str, Any]], candidate_experiments: List[Dict[str, Any]]) -> np.ndarray:
        """Constructs the continuous 8-dimensional state vector."""
        if not candidate_hypotheses:
            return np.zeros(8, dtype=np.float32)
            
        confs = [h.get("confidence", 0.5) for h in candidate_hypotheses]
        novs = [h.get("novelty_score", 0.5) for h in candidate_hypotheses]
        costs = [e.get("computational_cost", 1.0) for e in candidate_experiments]
        
        # Historical success rate
        if self.history_records:
            success_count = sum(1 for r in self.history_records if r.get("support_verdict") == "supported")
            hist_success = success_count / len(self.history_records)
        else:
            hist_success = 0.5
            
        mean_conf = float(np.mean(confs))
        mean_uncert = float(1.0 - mean_conf)
        mean_nov = float(np.mean(novs))
        budget_ratio = float(max(0.0, self.remaining_budget / self.initial_budget))
        expected_gain = float(mean_uncert * mean_nov)
        queue_size = float(min(1.0, len(candidate_hypotheses) / 20.0))
        avg_cost = float(min(1.0, np.mean(costs) / 5.0))
        
        state = np.array([
            mean_conf,
            mean_uncert,
            mean_nov,
            budget_ratio,
            hist_success,
            expected_gain,
            queue_size,
            avg_cost
        ], dtype=np.float32)
        
        return state

    def step_budget(self, cost: float):
        self.remaining_budget = max(0.0, self.remaining_budget - cost)

    def record_outcome(self, outcome: Dict[str, Any]):
        self.history_records.append(outcome)
