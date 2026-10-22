from typing import Dict, Any, List, Optional
from src.experiments.experiment import ExperimentResult

class CriticAgent:
    """
    Adversarial Critic Agent.
    Actively attempts to disprove scientific hypotheses, checks confounding factors,
    evaluates statistical weakness, and prevents premature claims of discovery.
    """
    def __init__(self, min_sample_size: int = 30, min_effect_size: float = 0.25):
        self.min_sample_size = min_sample_size
        self.min_effect_size = min_effect_size

    def critique(
        self,
        hypothesis: Dict[str, Any],
        experiment_result: ExperimentResult
    ) -> Dict[str, Any]:
        """Subject the experimental outcome to rigorous adversarial challenge."""
        objections = []
        falsification_risk = 0.0
        
        # 1. Statistical Significance Check
        if experiment_result.p_value > 0.05:
            objections.append(f"Statistically weak evidence (p = {experiment_result.p_value:.3f} > 0.05). Null hypothesis cannot be rejected.")
            falsification_risk += 0.4
            
        # 2. Effect Size & Predictive Power Check
        if experiment_result.r2_score < 0.25:
            objections.append(f"Insufficient explanatory power (Cross-Validation R² = {experiment_result.r2_score:.3f} < 0.25).")
            falsification_risk += 0.35
            
        if experiment_result.effect_size_cohens_d < self.min_effect_size:
            objections.append(f"Negligible effect size (Cohen's d = {experiment_result.effect_size_cohens_d:.3f} < {self.min_effect_size}).")
            falsification_risk += 0.25
            
        # 3. Sample Size Check
        n_samples = experiment_result.metrics.get("n_samples", 100)
        if n_samples < self.min_sample_size:
            objections.append(f"Small sample size (N = {n_samples} < {self.min_sample_size}). High risk of spurious sampling artifact.")
            falsification_risk += 0.2
            
        # 4. Confounder Risk
        vars_count = len(hypothesis.get("variables", []))
        if vars_count == 2:
            objections.append("Univariate analysis: Potential unobserved confounding variables (e.g. ionic radii ratio, atomic packing density).")
            falsification_risk += 0.1
            
        falsification_risk = float(min(1.0, falsification_risk))
        has_severe_objections = len(objections) >= 2 or experiment_result.r2_score < 0.15
        
        return {
            "has_severe_objections": has_severe_objections,
            "falsification_risk_score": round(falsification_risk, 3),
            "objections_count": len(objections),
            "objections": objections,
            "critic_summary": "Passed initial scrutiny with low risk." if not objections else f"Challenged by {len(objections)} statistical objections."
        }
