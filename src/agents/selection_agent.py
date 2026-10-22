from typing import Dict, Any, List, Optional
import numpy as np

class ExperimentSelectionAgent:
    """Selects the next optimal experiment from candidate queue using an RL policy or heuristic fallback."""
    def __init__(self, rl_policy: Optional[Any] = None, strategy_name: str = "rl_policy"):
        self.rl_policy = rl_policy
        self.strategy_name = strategy_name

    def select(
        self,
        candidate_experiments: List[Dict[str, Any]],
        state_vector: Optional[np.ndarray] = None
    ) -> int:
        """Returns the index of the selected experiment in the candidate list."""
        if not candidate_experiments:
            return 0
            
        if self.strategy_name == "rl_policy" and self.rl_policy is not None and state_vector is not None:
            return self.rl_policy.select_action(state_vector, len(candidate_experiments))
            
        # Fallback to multi-criteria heuristic scoring
        scores = []
        for exp in candidate_experiments:
            gain = exp.get("expected_info_gain", 0.5)
            nov = exp.get("novelty_score", 0.5)
            cost = exp.get("computational_cost", 1.0)
            score = (0.5 * gain) + (0.4 * nov) - (0.1 * cost / 3.0)
            scores.append(score)
            
        return int(np.argmax(scores))
