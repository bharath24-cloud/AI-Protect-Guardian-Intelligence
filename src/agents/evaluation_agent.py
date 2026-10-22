import uuid
from typing import Dict, Any, List, Optional
from src.experiments.experiment import ExperimentResult

class EvaluationAgent:
    """
    Evaluates experimental evidence and critic feedback.
    Assigns rigorous, non-exaggerated scientific statuses.
    """
    def evaluate(
        self,
        hypothesis: Dict[str, Any],
        experiment_result: ExperimentResult,
        critic_feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesizes experiment metrics and critic challenges into a validated scientific verdict."""
        evidence_id = f"evi-{uuid.uuid4().hex[:8]}"
        
        r2 = experiment_result.r2_score
        p_val = experiment_result.p_value
        effect_d = experiment_result.effect_size_cohens_d
        fals_risk = critic_feedback.get("falsification_risk_score", 0.0)
        has_severe = critic_feedback.get("has_severe_objections", False)
        
        # Rigorous status determination
        if r2 >= 0.50 and p_val < 0.01 and effect_d >= 0.5 and not has_severe:
            status = "supported"
            updated_conf = min(0.95, hypothesis.get("confidence", 0.5) + 0.25)
        elif r2 >= 0.30 and p_val < 0.05 and effect_d >= 0.25:
            status = "weak evidence"
            updated_conf = hypothesis.get("confidence", 0.5) + 0.10
        elif r2 < 0.10 and p_val > 0.10:
            status = "rejected"
            updated_conf = max(0.05, hypothesis.get("confidence", 0.5) - 0.40)
        elif has_severe:
            status = "contradicted"
            updated_conf = max(0.10, hypothesis.get("confidence", 0.5) - 0.25)
        else:
            status = "requires more experiments"
            updated_conf = hypothesis.get("confidence", 0.5)
            
        evidence_record = {
            "evidence_id": evidence_id,
            "hypothesis_id": hypothesis["hypothesis_id"],
            "experiment_id": experiment_result.experiment_id,
            "metric_name": "cross_validation_r2",
            "metric_value": r2,
            "p_value": p_val,
            "effect_size": effect_d,
            "confidence_interval": experiment_result.confidence_interval,
            "support_verdict": status,
            "updated_confidence": round(float(updated_conf), 3),
            "critique_notes": "; ".join(critic_feedback.get("objections", []))
        }
        
        return evidence_record
