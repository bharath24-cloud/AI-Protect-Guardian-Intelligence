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
from src.experiments.baselines import (
    RandomSelectionBaseline,
    UncertaintySamplingBaseline,
    GreedyInfoGainBaseline,
    HeuristicScoringBaseline
)

def run_baseline_benchmarks(num_cycles: int = 5) -> pd.DataFrame:
    """
    Executes and quantitatively compares all 5 scientific experiment selection strategies.
    """
    print("=" * 80)
    print("        SCIENTIFIC EXPERIMENT SELECTION BASELINE COMPARISON")
    print("=" * 80)
    
    strategies = [
        ("Random Selection", "random"),
        ("Uncertainty Sampling", "uncertainty"),
        ("Greedy Information Gain", "greedy"),
        ("Heuristic Multi-Criteria", "heuristic"),
        ("Proposed RL AI Scientist", "rl_policy")
    ]
    
    records = []
    
    for name, strat_key in strategies:
        print(f"\n[Benchmarking] Running strategy: {name} ({num_cycles} cycles)...")
        dataset = MaterialsScienceDataset(num_samples=800, random_seed=42)
        journal_file = f"results/logs/journal_{strat_key}.jsonl"
        
        loop = AutonomousDiscoveryLoop(
            dataset=dataset,
            journal_path=journal_file,
            budget_limit=25.0,
            random_seed=42
        )
        loop.selection_agent.strategy_name = strat_key
        
        cycle_res = loop.run_cycles(num_cycles=num_cycles)
        
        total_reward = sum(r["reward"] for r in cycle_res)
        avg_r2 = np.mean([r["r2_score"] for r in cycle_res])
        supported_count = sum(1 for r in cycle_res if r["verdict"] == "supported")
        total_time = sum(r["cycle_time_seconds"] for r in cycle_res)
        
        records.append({
            "Strategy": name,
            "Total Discovery Cycles": len(cycle_res),
            "Cumulative RL Reward": round(float(total_reward), 3),
            "Mean Experiment R²": round(float(avg_r2), 3),
            "Supported Hypotheses": supported_count,
            "Success Rate (%)": round((supported_count / max(1, len(cycle_res))) * 100.0, 1),
            "Total Runtime (s)": round(float(total_time), 2)
        })
        
    df = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print("                 BENCHMARK COMPARISON RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    out_csv = Path("results/metrics/baseline_comparison.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved benchmark results table to: {out_csv}")
    
    return df

if __name__ == "__main__":
    run_baseline_benchmarks(num_cycles=4)
