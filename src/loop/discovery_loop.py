import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from src.datasets.materials_data import MaterialsScienceDataset
from src.representation.autoencoder import CompositionalAutoencoder
from src.agents.discovery_agent import DiscoveryAgent
from src.agents.hypothesis_agent import HypothesisAgent
from src.agents.planner_agent import ExperimentPlannerAgent
from src.agents.selection_agent import ExperimentSelectionAgent
from src.agents.critic_agent import CriticAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.memory_agent import MemoryAgent
from src.agents.learning_agent import LearningAgent
from src.experiments.experiment_runner import ExperimentRunner
from src.rl.environment import DiscoveryEnv
from src.rl.policy import DQNPolicy
from src.rl.rewards import ScientificRewardEngine
from src.continual.continual_learner import ContinualLearner
from src.memory.scientific_memory import ScientificMemory

class AutonomousDiscoveryLoop:
    """
    Master Closed-Loop Autonomous Scientific Discovery Coordinator.
    Orchestrates the continuous multi-agent cycle of representation learning,
    pattern discovery, hypothesis generation, RL experiment selection,
    adversarial critique, statistical evaluation, and persistent memory updates.
    """
    def __init__(
        self,
        dataset: Optional[MaterialsScienceDataset] = None,
        memory: Optional[ScientificMemory] = None,
        journal_path: str = "results/research_journal.jsonl",
        budget_limit: float = 30.0,
        random_seed: int = 42
    ):
        self.dataset = dataset or MaterialsScienceDataset(num_samples=1000, random_seed=random_seed)
        self.memory = memory or ScientificMemory()
        self.journal_path = journal_path
        Path(self.journal_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Representations & Continual Learning
        self.rep_model = CompositionalAutoencoder(input_dim=len(self.dataset.get_feature_names()))
        self.continual_learner = ContinualLearner(self.rep_model)
        
        # RL Subsystem
        self.env = DiscoveryEnv(initial_budget=budget_limit)
        self.rl_policy = DQNPolicy(state_dim=8, hidden_dim=64, max_actions=10)
        self.reward_engine = ScientificRewardEngine()
        
        # Multi-Agent Pool
        self.discovery_agent = DiscoveryAgent()
        self.hypothesis_agent = HypothesisAgent()
        self.planner_agent = ExperimentPlannerAgent()
        self.selection_agent = ExperimentSelectionAgent(rl_policy=self.rl_policy, strategy_name="rl_policy")
        self.critic_agent = CriticAgent()
        self.evaluation_agent = EvaluationAgent()
        self.memory_agent = MemoryAgent(self.memory)
        self.learning_agent = LearningAgent(rl_policy=self.rl_policy, continual_learner=self.continual_learner)
        
        # Experiment Runner
        self.runner = ExperimentRunner(random_seed=random_seed)
        
        self.current_cycle = 0
        self.journal_records = []
        self.is_running = False

    def run_cycles(self, num_cycles: int = 5) -> List[Dict[str, Any]]:
        """Executes sequential autonomous scientific discovery cycles."""
        self.is_running = True
        cycle_summaries = []
        
        # Pre-train representation model on initial data
        features = self.dataset.get_features()
        feature_names = self.dataset.get_feature_names()
        target_names = self.dataset.get_target_names()
        metadata_df = self.dataset.get_metadata()
        material_ids = metadata_df["material_id"].tolist()
        
        print(f"[AI-Scientist] Initializing Representation Learning on {len(features)} material records...")
        self.rep_model.fit(features, epochs=10)
        
        for c in range(1, num_cycles + 1):
            if self.env.remaining_budget <= 0.0:
                print(f"[AI-Scientist] Budget exhausted at cycle {c}. Stopping loop.")
                break
                
            self.current_cycle = c
            t0 = time.time()
            target_prop = target_names[(c - 1) % len(target_names)]
            targets = self.dataset.get_targets(target_prop)
            
            # 1. Representation & Pattern Discovery
            latents = self.rep_model.encode(features)
            discoveries = self.discovery_agent.analyze(
                features=features,
                targets=targets,
                feature_names=feature_names,
                target_name=target_prop,
                representations=latents,
                material_ids=material_ids
            )
            
            # 2. Hypothesis Generation & Deduplication
            candidate_hypotheses = self.hypothesis_agent.generate_hypotheses(discoveries)
            novel_hypotheses = self.memory_agent.filter_redundant_hypotheses(candidate_hypotheses)
            
            if not novel_hypotheses:
                # If all discovered are duplicates, reuse best candidate
                novel_hypotheses = candidate_hypotheses[:3]
                
            # 3. Experiment Planning
            planned_experiments = self.planner_agent.design_experiments(novel_hypotheses)
            exp_dict_list = [
                {
                    "experiment_id": exp.experiment_id,
                    "hypothesis_id": exp.hypothesis_id,
                    "expected_info_gain": 0.6,
                    "novelty_score": 0.7,
                    "computational_cost": exp.computational_budget_cost
                }
                for exp in planned_experiments
            ]
            
            # 4. RL Experiment Selection
            current_state = self.env.get_state(novel_hypotheses, exp_dict_list)
            selected_idx = self.selection_agent.select(exp_dict_list, state_vector=current_state)
            selected_exp_config = planned_experiments[selected_idx]
            selected_hyp = next(
                h for h in novel_hypotheses if h["hypothesis_id"] == selected_exp_config.hypothesis_id
            )
            
            # 5. Experiment Execution
            exp_result = self.runner.execute(
                config=selected_exp_config,
                features=features,
                targets=targets,
                feature_names=feature_names
            )
            self.env.step_budget(selected_exp_config.computational_budget_cost)
            
            # 6. Critic Challenge
            critic_feedback = self.critic_agent.critique(selected_hyp, exp_result)
            
            # 7. Statistical Evaluation & Verdict
            evidence = self.evaluation_agent.evaluate(selected_hyp, exp_result, critic_feedback)
            
            # 8. Multi-Objective Reward & RL Update
            reward_dict = self.reward_engine.calculate_reward(
                evidence=evidence,
                experiment_config={"computational_cost": selected_exp_config.computational_budget_cost},
                hypothesis=selected_hyp,
                is_redundant=False
            )
            next_state = self.env.get_state(novel_hypotheses, exp_dict_list)
            policy_loss = self.learning_agent.update_policy(
                state=current_state,
                action=selected_idx,
                reward=reward_dict["total_reward"],
                next_state=next_state,
                done=(c == num_cycles)
            )
            
            # 9. Continual Learning Update
            cl_res = self.learning_agent.update_continual_representations(
                new_data=features[:50],
                task_id=f"cycle_{c}_{target_prop}"
            )
            
            # 10. Commit to Persistent Memory & Knowledge Graph
            mat_family = metadata_df.iloc[0].get("task_family", "Transition Metal Oxides")
            self.memory_agent.commit_cycle(
                material_family=mat_family,
                property_name=target_prop,
                hypothesis=selected_hyp,
                experiment={
                    "experiment_id": selected_exp_config.experiment_id,
                    "hypothesis_id": selected_hyp["hypothesis_id"],
                    "experiment_type": selected_exp_config.experiment_type,
                    "parameters": {"cv_folds": selected_exp_config.cv_folds, "features": selected_exp_config.feature_subset},
                    "random_seed": selected_exp_config.random_seed,
                    "computational_cost": selected_exp_config.computational_budget_cost,
                    "status": exp_result.status,
                    "metrics": exp_result.metrics
                },
                evidence=evidence
            )
            
            # 11. Journal Logging
            elapsed = time.time() - t0
            journal_entry = {
                "cycle": c,
                "target_property": target_prop,
                "observation": discoveries[0]["description"] if discoveries else "No significant pattern",
                "hypothesis": selected_hyp["statement"],
                "hypothesis_id": selected_hyp["hypothesis_id"],
                "experiment_id": selected_exp_config.experiment_id,
                "r2_score": exp_result.r2_score,
                "p_value": exp_result.p_value,
                "effect_size": exp_result.effect_size_cohens_d,
                "critic_challenges": critic_feedback["objections"],
                "verdict": evidence["support_verdict"],
                "updated_confidence": evidence["updated_confidence"],
                "reward": reward_dict["total_reward"],
                "remaining_budget": round(self.env.remaining_budget, 2),
                "policy_loss": policy_loss,
                "cycle_time_seconds": round(elapsed, 2)
            }
            
            self._log_journal(journal_entry)
            cycle_summaries.append(journal_entry)
            
            print(f"[Cycle {c}/{num_cycles}] Prop: {target_prop:<20} | Verdict: {evidence['support_verdict']:<16} | R²: {exp_result.r2_score:>6.3f} | Reward: {reward_dict['total_reward']:>6.3f} | Cost: {selected_exp_config.computational_budget_cost}")

        self.is_running = False
        return cycle_summaries

    def _log_journal(self, entry: Dict[str, Any]):
        self.journal_records.append(entry)
        with open(self.journal_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
