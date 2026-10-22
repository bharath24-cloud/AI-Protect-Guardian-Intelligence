import json
import sys
from pathlib import Path
from typing import Dict, Any, List

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.datasets.materials_data import MaterialsScienceDataset
from src.loop.discovery_loop import AutonomousDiscoveryLoop

def run_ablation_studies(num_cycles: int = 4) -> pd.DataFrame:
    """
    Performs scientific ablation studies by removing one subsystem at a time:
    1. Proposed Full Architecture
    2. No RL (Random action)
    3. No Continual Learning (No replay updates)
    4. No Critic (No adversarial objections)
    5. No Scientific Memory (No deduplication filter)
    6. No Novelty Score in Reward (w_nov = 0.0)
    """
    print("=" * 80)
    print("           AUTONOMOUS SCIENTIFIC DISCOVERY ABLATION STUDY")
    print("=" * 80)
    
    ablations = [
        ("Full Proposed System", {}),
        ("Ablation 1: No RL Policy (Random)", {"no_rl": True}),
        ("Ablation 2: No Continual Learning", {"no_cl": True}),
        ("Ablation 3: No Critic Agent", {"no_critic": True}),
        ("Ablation 4: No Scientific Memory", {"no_memory": True}),
        ("Ablation 5: No Novelty Reward", {"no_novelty": True})
    ]
    
    records = []
    
    for name, flags in ablations:
        print(f"\n[Ablation] Running configuration: {name}...")
        dataset = MaterialsScienceDataset(num_samples=800, random_seed=42)
        journal_file = f"results/logs/ablation_{name.lower().replace(' ', '_').replace(':', '')}.jsonl"
        
        loop = AutonomousDiscoveryLoop(
            dataset=dataset,
            journal_path=journal_file,
            budget_limit=25.0,
            random_seed=42
        )
        
        # Apply ablations
        if flags.get("no_rl"):
            loop.selection_agent.strategy_name = "random"
        if flags.get("no_cl"):
            loop.learning_agent.continual_learner = None
        if flags.get("no_critic"):
            loop.critic_agent.min_sample_size = 0
            loop.critic_agent.min_effect_size = 0.0
        if flags.get("no_memory"):
            loop.memory_agent.filter_redundant_hypotheses = lambda hyps: hyps
        if flags.get("no_novelty"):
            loop.reward_engine.w_novelty = 0.0
            
        cycle_res = loop.run_cycles(num_cycles=num_cycles)
        
        total_reward = sum(r["reward"] for r in cycle_res)
        avg_r2 = np.mean([r["r2_score"] for r in cycle_res])
        supported_count = sum(1 for r in cycle_res if r["verdict"] == "supported")
        
        records.append({
            "Configuration": name,
            "Cycles": len(cycle_res),
            "Cumulative Reward": round(float(total_reward), 3),
            "Mean R² Score": round(float(avg_r2), 3),
            "Supported Hypotheses": supported_count,
            "Success Rate (%)": round((supported_count / max(1, len(cycle_res))) * 100.0, 1)
        })
        
    df = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print("                    ABLATION STUDY RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    out_csv = Path("results/metrics/ablation_study.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved ablation study results to: {out_csv}")
    
    return df

if __name__ == "__main__":
    run_ablation_studies(num_cycles=4)
