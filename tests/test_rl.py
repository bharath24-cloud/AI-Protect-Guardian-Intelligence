import pytest
import numpy as np
from src.rl.environment import DiscoveryEnv
from src.rl.policy import DQNPolicy
from src.rl.rewards import ScientificRewardEngine

def test_rl_subsystem():
    env = DiscoveryEnv(initial_budget=20.0)
    policy = DQNPolicy(state_dim=8, hidden_dim=32, max_actions=5)
    reward_engine = ScientificRewardEngine()
    
    hyps = [{"confidence": 0.6, "novelty_score": 0.8}]
    exps = [{"computational_cost": 1.5}]
    
    state = env.get_state(hyps, exps)
    assert len(state) == 8
    
    action = policy.select_action(state, num_available_actions=3)
    assert 0 <= action < 3
    
    reward_dict = reward_engine.calculate_reward(
        evidence={"metric_value": 0.72, "support_verdict": "supported"},
        experiment_config={"computational_cost": 1.5},
        hypothesis={"novelty_score": 0.8}
    )
    assert "total_reward" in reward_dict
    assert reward_dict["total_reward"] > 0
    
    next_state = env.get_state(hyps, exps)
    # Test update
    policy.update(state, action, reward_dict["total_reward"], next_state, done=False)
