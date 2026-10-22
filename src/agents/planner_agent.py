import uuid
from typing import Dict, Any, List, Optional
from src.experiments.experiment import ExperimentConfig

class ExperimentPlannerAgent:
    """Translates high-level scientific hypotheses into executable computational experiment configurations."""
    def __init__(self, default_cv_folds: int = 5):
        self.default_cv_folds = default_cv_folds

    def design_experiments(self, hypotheses: List[Dict[str, Any]]) -> List[ExperimentConfig]:
        """Creates formal ExperimentConfig objects for candidate hypotheses."""
        configs = []
        for h in hypotheses:
            exp_id = f"exp-{uuid.uuid4().hex[:8]}"
            target_prop = h["variables"][-1] if h["variables"] else "band_gap"
            feature_subset = [v for v in h["variables"] if v != target_prop]
            
            # Model selection based on hypothesis complexity
            if h.get("expected_relationship") == "analytical_power_law":
                arch = "gradient_boosting"
                cost = 1.5
            else:
                arch = "random_forest"
                cost = 1.0
                
            config = ExperimentConfig(
                experiment_id=exp_id,
                hypothesis_id=h["hypothesis_id"],
                experiment_type="cross_validation_regression",
                target_property=target_prop,
                feature_subset=feature_subset,
                model_architecture=arch,
                cv_folds=self.default_cv_folds,
                computational_budget_cost=cost
            )
            configs.append(config)
            
        return configs
