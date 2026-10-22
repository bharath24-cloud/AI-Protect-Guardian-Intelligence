from typing import Dict, Any, List, Optional
import numpy as np

class LearningAgent:
    """Oversees RL policy optimization and continual representation learning updates."""
    def __init__(self, rl_policy: Optional[Any] = None, continual_learner: Optional[Any] = None):
        self.rl_policy = rl_policy
        self.continual_learner = continual_learner

    def update_policy(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False
    ) -> Optional[float]:
        """Performs a single reinforcement learning optimization step."""
        if self.rl_policy is not None and hasattr(self.rl_policy, "update"):
            return self.rl_policy.update(state, action, reward, next_state, done)
        return None

    def update_continual_representations(
        self,
        new_data: np.ndarray,
        task_id: str
    ) -> Dict[str, Any]:
        """Triggers continual learning update on newly assimilated materials data."""
        if self.continual_learner is not None:
            return self.continual_learner.learn_task(new_data, task_id)
        return {"status": "skipped"}
