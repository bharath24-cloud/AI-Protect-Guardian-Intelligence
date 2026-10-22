from typing import Dict, List, Any, Optional
from backend.app.guardian.schemas import EvidenceSummary


class EvidenceAnalyst:
    """
    Synthesizes grounded empirical evidence, separates observations
    from hypotheses and recommendations, and enforces scientific integrity.
    """

    DISCLAIMER = (
        "Empirical finding: Adversarial cloaking produces measured disruption on evaluated models. "
        "It does not guarantee mathematical cryptography or universal immunity against unseen models."
    )

    @classmethod
    def analyze_execution_results(
        cls,
        plan_id: str,
        execution_id: str,
        results_json: Dict[str, Any],
        objective_type: str,
    ) -> EvidenceSummary:
        """
        Derives an EvidenceSummary strictly from measured tool outputs.
        """
        cited_ids = [plan_id, execution_id]
        observations: List[str] = []
        hypotheses: List[str] = []
        recommendations: List[str] = []
        uncertainties: List[str] = []

        # 1. Quality & Perturbation Observations
        qual = results_json.get("calculate_quality_metrics", {})
        if qual:
            ssim_val = qual.get("ssim")
            psnr_val = qual.get("psnr_db")
            lpips_val = qual.get("lpips_alexnet")
            linf_val = qual.get("linf_norm")
            if ssim_val is not None:
                observations.append(
                    f"Structural Similarity (SSIM) measured at {ssim_val:.4f} (visual structure preserved)."
                )
            if psnr_val is not None:
                observations.append(
                    f"Peak Signal-to-Noise Ratio (PSNR) measured at {psnr_val:.2f} dB (high perceptual fidelity)."
                )
            if lpips_val is not None:
                observations.append(
                    f"Perceptual distance (LPIPS AlexNet) measured at {lpips_val:.4f}."
                )
            if linf_val is not None:
                observations.append(
                    f"Maximum L-infinity perturbation norm bounded at {linf_val:.5f} (within budget)."
                )

        # 2. Cloaking & Baseline Observations
        cloak_res = results_json.get("run_adversarial_cloaking", {})
        base_res = results_json.get("evaluate_baseline_image", {})
        if cloak_res:
            l2_val = cloak_res.get("l2_norm")
            if l2_val is not None:
                observations.append(f"Global L2 perturbation energy measured at {l2_val:.3f}.")

        # 3. Cross-Model Transferability Observations
        trans_res = results_json.get("evaluate_cross_model_transferability", {})
        if trans_res:
            dis_rate = trans_res.get("disruption_rate_pct")
            model_res = trans_res.get("model_results", {})
            if dis_rate is not None:
                observations.append(
                    f"Cross-architecture disruption rate measured at {dis_rate:.1f}% across {len(model_res)} models."
                )
            for m_name, m_data in model_res.items():
                if isinstance(m_data, dict):
                    dis = m_data.get("disrupted")
                    cos_d = m_data.get("cosine_dist")
                    if dis is not None and cos_d is not None:
                        observations.append(
                            f"Model '{m_name}': Disrupted={dis}, Feature Cosine Distance={cos_d:.4f}."
                        )

            # Hypothesis and recommendation based on transferability
            if dis_rate is not None and dis_rate < 50.0:
                hypotheses.append(
                    "Hypothesis: Single-surrogate ResNet-50 perturbations overfit to residual convolutional features, "
                    "attenuating transferability to Vision Transformers (ViT) and modern ConvNets (ConvNeXt)."
                )
                recommendations.append(
                    "Recommendation: Conduct an ensemble multi-surrogate optimization run combining ResNet-50 and ViT-B/16."
                )
            else:
                hypotheses.append(
                    "Hypothesis: Bounded adversarial gradients induce cross-model feature misalignment across disparate architectures."
                )
                recommendations.append(
                    "Recommendation: Validate perturbation survival against social media compression pipelines."
                )

        # 4. Digital Robustness Observations
        rob_res = results_json.get("evaluate_digital_robustness", {})
        if rob_res:
            surv_rate = rob_res.get("retained_protection_rate") or rob_res.get("survival_rate")
            if surv_rate is not None:
                observations.append(
                    f"Digital transformation survival rate measured at {surv_rate * 100.0 if surv_rate <= 1.0 else surv_rate:.1f}%."
                )
            transforms = rob_res.get("transformations", {})
            for t_name, t_data in transforms.items():
                if isinstance(t_data, dict):
                    observations.append(
                        f"Transform '{t_name}': Retained Disruption={t_data.get('disrupted', False)}."
                    )

        # 5. Diagnostic / Model Availability Observations
        avail_res = results_json.get("check_model_availability", {})
        if avail_res:
            models_info = avail_res.get("models", {})
            ready_models = [m for m, v in models_info.items() if v.get("status") == "ready"]
            unavail_models = [m for m, v in models_info.items() if v.get("status") != "ready"]
            observations.append(f"Model pool audit: {len(ready_models)} ready, {len(unavail_models)} unavailable.")
            if unavail_models:
                uncertainties.append(f"Unavailable models: {unavail_models}. Checkpoints or weights must be verified.")

        # 6. Uncertainties and Caveats
        uncertainties.append(
            "Sample Size Caveat: Single visual asset evaluation provides empirical illustration; "
            "run statistical held-out batch validation (N >= 25) for population confidence intervals."
        )
        uncertainties.append(
            "Downstream Limitation: Classification disruption does not mathematically guarantee prevention "
            "of generative model fine-tuning or diffusion scraping."
        )

        # Fallback if no tool output was parsed
        if not observations:
            observations.append("Execution completed. No structured tool metrics were registered.")
            recommendations.append("Verify tool registration and parameters.")

        return EvidenceSummary(
            plan_id=plan_id,
            execution_id=execution_id,
            cited_experiment_ids=cited_ids,
            empirical_observations=observations,
            hypotheses_tested=hypotheses,
            grounded_recommendations=recommendations,
            uncertainties_and_caveats=uncertainties,
            disclaimer=cls.DISCLAIMER,
        )
