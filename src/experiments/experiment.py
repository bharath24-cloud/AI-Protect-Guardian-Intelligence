from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time

@dataclass
class ExperimentConfig:
    experiment_id: str
    hypothesis_id: str
    experiment_type: str # e.g. "cross_validation_regression", "feature_ablation", "subgroup_analysis"
    target_property: str
    feature_subset: List[str]
    model_architecture: str = "gradient_boosting"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    cv_folds: int = 5
    computational_budget_cost: float = 1.0


@dataclass
class ExperimentResult:
    experiment_id: str
    hypothesis_id: str
    status: str # "completed", "failed", "inconclusive"
    r2_score: float
    mae: float
    rmse: float
    p_value: float
    effect_size_cohens_d: float
    confidence_interval: List[float]
    execution_time_seconds: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
