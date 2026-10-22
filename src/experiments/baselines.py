from typing import List, Dict, Any, Optional
import numpy as np

class RandomSelectionBaseline:
    """Baseline 1: Randomly samples the next experiment from the candidate pool."""
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def select_experiment(self, candidate_experiments: List[Dict[str, Any]]) -> int:
        if not candidate_experiments:
            return 0
        return int(self.rng.choice(len(candidate_experiments)))


class UncertaintySamplingBaseline:
    """Baseline 2: Selects the experiment associated with the highest model uncertainty."""
    def select_experiment(self, candidate_experiments: List[Dict[str, Any]]) -> int:
        if not candidate_experiments:
            return 0
        uncertainties = [
            exp.get("estimated_uncertainty", 0.5) for exp in candidate_experiments
        ]
        return int(np.argmax(uncertainties))


class GreedyInfoGainBaseline:
    """Baseline 3: Greedily selects the experiment with the highest immediate expected information gain."""
    def select_experiment(self, candidate_experiments: List[Dict[str, Any]]) -> int:
        if not candidate_experiments:
            return 0
        gains = [
            exp.get("expected_info_gain", 0.5) for exp in candidate_experiments
        ]
        return int(np.argmax(gains))


class HeuristicScoringBaseline:
    """Baseline 4: Multi-criteria weighted heuristic formula (Novelty + InfoGain - Cost)."""
    def __init__(self, w_nov: float = 0.4, w_gain: float = 0.5, w_cost: float = 0.2):
        self.w_nov = w_nov
        self.w_gain = w_gain
        self.w_cost = w_cost

    def select_experiment(self, candidate_experiments: List[Dict[str, Any]]) -> int:
        if not candidate_experiments:
            return 0
        scores = []
        for exp in candidate_experiments:
            nov = exp.get("novelty_score", 0.5)
            gain = exp.get("expected_info_gain", 0.5)
            cost = exp.get("computational_cost", 1.0)
            score = (self.w_nov * nov) + (self.w_gain * gain) - (self.w_cost * cost / 5.0)
            scores.append(score)
        return int(np.argmax(scores))
