import time
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.experiments.experiment import ExperimentConfig, ExperimentResult
from src.experiments.statistical_tests import (
    perform_statistical_hypothesis_test,
    calculate_bootstrap_ci
)

class ExperimentRunner:
    """
    Controlled Computational Experiment Engine.
    Executes cross-validated models and runs statistical hypothesis tests to validate relationships.
    """
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def execute(
        self,
        config: ExperimentConfig,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: List[str]
    ) -> ExperimentResult:
        """Runs the specified computational experiment using cross-validation."""
        t0 = time.time()
        
        # Select active feature indices
        if config.feature_subset:
            feat_indices = [
                i for i, name in enumerate(feature_names)
                if name in config.feature_subset
            ]
            if not feat_indices:
                feat_indices = list(range(min(4, features.shape[1])))
        else:
            feat_indices = list(range(features.shape[1]))
            
        X = features[:, feat_indices]
        y = targets
        
        # Instantiate model architecture
        if config.model_architecture == "random_forest":
            model = RandomForestRegressor(n_estimators=50, random_state=self.random_seed)
        elif config.model_architecture == "ridge":
            model = Ridge(alpha=1.0)
        else:
            model = GradientBoostingRegressor(n_estimators=40, learning_rate=0.1, random_state=self.random_seed)
            
        # 5-fold cross-validation
        kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=self.random_seed)
        y_preds = np.zeros_like(y)
        baseline_preds = np.full_like(y, np.mean(y)) # Dummy mean predictor baseline
        
        for train_idx, val_idx in kf.split(X):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val = X[val_idx]
            model.fit(X_tr, y_tr)
            y_preds[val_idx] = model.predict(X_val)
            
        # Metrics
        r2 = float(r2_score(y, y_preds))
        mae = float(mean_absolute_error(y, y_preds))
        rmse = float(np.sqrt(mean_squared_error(y, y_preds)))
        
        # Residuals
        model_res = np.abs(y - y_preds)
        base_res = np.abs(y - baseline_preds)
        
        # Statistical tests
        stat_test = perform_statistical_hypothesis_test(model_res, base_res)
        ci = calculate_bootstrap_ci(y, y_preds, metric_fn=r2_score, n_bootstraps=200, random_seed=self.random_seed)
        
        elapsed = time.time() - t0
        
        return ExperimentResult(
            experiment_id=config.experiment_id,
            hypothesis_id=config.hypothesis_id,
            status="completed",
            r2_score=round(r2, 4),
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            p_value=stat_test["p_value"],
            effect_size_cohens_d=stat_test["effect_size_cohens_d"],
            confidence_interval=ci,
            execution_time_seconds=round(elapsed, 3),
            metrics={
                "cv_folds": config.cv_folds,
                "n_samples": len(y),
                "n_features_used": len(feat_indices),
                "is_significant": stat_test["is_statistically_significant"],
                "p_value_wilcoxon": stat_test["p_value_wilcoxon"]
            }
        )
