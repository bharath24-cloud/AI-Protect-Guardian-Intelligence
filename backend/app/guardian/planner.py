import json
import re
import uuid
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from backend.app.guardian.schemas import (
    GoalRequest,
    PlanStep,
    ResourceEstimate,
    SuccessCriteria,
    ExperimentPlanResponse,
)
from backend.app.guardian.policies import PolicyEngine
from backend.app.guardian.tool_registry import default_tool_registry


class AdaptiveExperimentPlanner:
    """
    Translates experimental goals into validated, reproducible,
    and policy-bound experiment plans.
    """

    OBJECTIVE_TYPES = [
        "evaluate_cloaking",
        "compare_experiments",
        "optimize_disruption",
        "cross_transfer",
        "robustness_benchmark",
        "model_investigation",
        "recommend_next",
    ]

    def __init__(self, tool_registry=None):
        self.registry = tool_registry or default_tool_registry

    def parse_objective(self, goal_text: str) -> Tuple[str, float]:
        """
        Classifies the user goal into a canonical objective type with confidence.
        """
        text = goal_text.lower().strip()

        # 1. Compare experiments
        if any(k in text for k in ["compare", "versus", "vs", "diff", "delta between"]):
            return "compare_experiments", 0.95

        # 2. Investigate model
        if any(k in text for k in ["investigate", "failing model", "unavailable model", "why is clip", "diagnose model", "check model"]):
            return "model_investigation", 0.90

        # 3. Optimize disruption under constraints
        if any(k in text for k in ["optimize", "improve disruption", "maximize disruption", "subject to", "constraint", "ssim >="]):
            return "optimize_disruption", 0.90

        # 4. Cross-model transferability
        if any(k in text for k in ["transferability", "cross-model", "black-box", "vit", "convnext", "transfer rate"]):
            return "cross_transfer", 0.92

        # 5. Robustness under transformations
        if any(k in text for k in ["robustness", "transformation", "jpeg", "compression", "resizing", "blur", "crop", "survive"]):
            return "robustness_benchmark", 0.92

        # 6. Recommend next experiment
        if any(k in text for k in ["recommend", "what next", "suggest next", "next step", "next experiment"]):
            return "recommend_next", 0.88

        # 7. Evaluate cloaking configuration
        if any(k in text for k in ["evaluate", "cloak", "protect", "pgd", "perturb", "synthesize", "test image"]):
            return "evaluate_cloaking", 0.85

        # Ambiguous check: very short or unspecified
        if len(text.split()) < 3:
            return "ambiguous", 0.30

        # Default fallback
        return "evaluate_cloaking", 0.50

    def create_plan(
        self,
        request: GoalRequest,
        default_image_path: Optional[str] = None,
    ) -> ExperimentPlanResponse:
        """
        Constructs a complete, validated experiment plan from a goal request.
        """
        objective_type, confidence = self.parse_objective(request.goal)

        # Ambiguous handling
        if objective_type == "ambiguous":
            plan_id = str(uuid.uuid4())
            return ExperimentPlanResponse(
                plan_id=plan_id,
                title="Clarification Required",
                goal=request.goal,
                objective_type="ambiguous",
                status="blocked",
                requires_approval=False,
                approval_status="pending",
                config_hash="0" * 64,
                steps=[],
                resource_estimate=ResourceEstimate(),
                success_criteria=[],
                missing_prerequisites=["Specific experimental objective"],
                clarification_needed=(
                    "Your request is ambiguous. Please specify your target objective: "
                    "e.g., 'Evaluate cloaking on artwork', 'Test cross-model transferability', "
                    "'Benchmark JPEG robustness', or 'Compare experiment A and B'."
                ),
            )

        # Dispatch objective builder
        if objective_type == "evaluate_cloaking":
            plan = self._build_evaluate_cloaking_plan(request, default_image_path)
        elif objective_type == "compare_experiments":
            plan = self._build_compare_experiments_plan(request)
        elif objective_type == "optimize_disruption":
            plan = self._build_optimize_disruption_plan(request, default_image_path)
        elif objective_type == "cross_transfer":
            plan = self._build_cross_transfer_plan(request, default_image_path)
        elif objective_type == "robustness_benchmark":
            plan = self._build_robustness_plan(request, default_image_path)
        elif objective_type == "model_investigation":
            plan = self._build_model_investigation_plan(request)
        elif objective_type == "recommend_next":
            plan = self._build_recommend_next_plan(request)
        else:
            plan = self._build_evaluate_cloaking_plan(request, default_image_path)

        # Compute deterministic config hash
        plan_dict = {
            "objective_type": plan.objective_type,
            "goal": plan.goal,
            "steps": [s.model_dump() for s in plan.steps],
        }
        plan.config_hash = PolicyEngine.compute_config_hash(plan_dict)

        # Evaluate approval requirement
        plan.requires_approval = PolicyEngine.plan_requires_approval(plan.steps)
        if plan.requires_approval:
            plan.status = "awaiting_approval"
            plan.approval_status = "pending"
        else:
            plan.status = "approved"
            plan.approval_status = "approved"

        # Check resource limits
        res_ok, res_err = PolicyEngine.check_resource_limits(plan.resource_estimate)
        if not res_ok:
            plan.status = "blocked"
            plan.missing_prerequisites.append(res_err or "Exceeds hardware resource ceiling.")

        return plan

    # -------------------------------------------------------------------------
    # Plan Builders for the 7 Supported Goals
    # -------------------------------------------------------------------------
    def _build_evaluate_cloaking_plan(
        self,
        request: GoalRequest,
        default_image_path: Optional[str],
    ) -> ExperimentPlanResponse:
        plan_id = str(uuid.uuid4())
        target_img = self._resolve_image_path(request, default_image_path)
        prereqs = []
        if not target_img:
            prereqs.append("Target artwork image path or database image ID")

        eps = request.constraints.get("epsilon", 8.0 / 255.0) if request.constraints else 8.0 / 255.0
        iters = request.constraints.get("iterations", 20) if request.constraints else 20

        steps = [
            PlanStep(
                step_number=1,
                tool_name="validate_image_input",
                description="Validate input artwork image format, dimensions, and channels.",
                parameters={"image_path": target_img, "image_id": request.image_id},
                requires_approval=False,
            ),
            PlanStep(
                step_number=2,
                tool_name="evaluate_baseline_image",
                description="Run baseline inference across vision architectures before perturbation.",
                parameters={"image_path": target_img, "models": ["resnet50", "convnext_tiny", "vit_b_16", "clip_vit_b32"]},
                requires_approval=True,
            ),
            PlanStep(
                step_number=3,
                tool_name="run_adversarial_cloaking",
                description=f"Synthesize bounded PGD adversarial perturbation (eps={eps:.4f}, iters={iters}).",
                parameters={"image_path": target_img, "epsilon": eps, "iterations": iters},
                requires_approval=True,
            ),
            PlanStep(
                step_number=4,
                tool_name="calculate_quality_metrics",
                description="Calculate tensor-level SSIM, PSNR, LPIPS, and perturbation norms.",
                parameters={"original_image_path": target_img},
                requires_approval=True,
            ),
            PlanStep(
                step_number=5,
                tool_name="evaluate_cross_model_transferability",
                description="Evaluate classification disruption and cosine distance on black-box models.",
                parameters={"original_image_path": target_img},
                requires_approval=True,
            ),
        ]

        return ExperimentPlanResponse(
            plan_id=plan_id,
            title="Adversarial Cloaking & Multi-Model Evaluation",
            goal=request.goal,
            objective_type="evaluate_cloaking",
            status="proposed",
            requires_approval=True,
            approval_status="pending",
            config_hash="",
            steps=steps,
            resource_estimate=ResourceEstimate(
                gpu_vram_gb=1.2,
                estimated_runtime_seconds=18.0,
                disk_mb=2.5,
                hardware_device="cuda",
            ),
            success_criteria=[
                SuccessCriteria(
                    target_metric="top1_flip",
                    condition="surrogate_prediction_flipped == True",
                    threshold=1.0,
                    description="Surrogate ResNet-50 Top-1 class disrupted.",
                ),
                SuccessCriteria(
                    target_metric="ssim",
                    condition="ssim >= 0.85",
                    threshold=0.85,
                    description="Perceptual structural similarity preserved.",
                ),
                SuccessCriteria(
                    target_metric="linf_norm",
                    condition=f"linf <= {eps}",
                    threshold=eps,
                    description="Perturbation strictly bounded within budget.",
                ),
            ],
            missing_prerequisites=prereqs,
        )

    def _build_compare_experiments_plan(self, request: GoalRequest) -> ExperimentPlanResponse:
        plan_id = str(uuid.uuid4())
        # Extract experiment IDs from goal or context
        exp_ids = []
        if request.context and "experiment_ids" in request.context:
            exp_ids = request.context["experiment_ids"]
        else:
            # Regex match UUIDs in goal
            uuids = re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", request.goal)
            exp_ids = uuids

        prereqs = []
        if len(exp_ids) < 2:
            prereqs.append("At least 2 valid experiment or job IDs to compare")

        steps = [
            PlanStep(
                step_number=1,
                tool_name="retrieve_and_compare_experiments",
                description="Retrieve experiment configurations, metrics, and compute delta alignment.",
                parameters={"experiment_ids": exp_ids},
                requires_approval=False,
            ),
        ]

        return ExperimentPlanResponse(
            plan_id=plan_id,
            title="Cross-Experiment Differential Benchmark",
            goal=request.goal,
            objective_type="compare_experiments",
            status="proposed",
            requires_approval=False,
            approval_status="approved",
            config_hash="",
            steps=steps,
            resource_estimate=ResourceEstimate(
                gpu_vram_gb=0.0,
                estimated_runtime_seconds=0.5,
                disk_mb=0.1,
                hardware_device="cpu",
            ),
            success_criteria=[
                SuccessCriteria(
                    target_metric="metric_alignment",
                    condition="deltas_computed == True",
                    threshold=1.0,
                    description="Successful side-by-side metric differential computed.",
                ),
            ],
            missing_prerequisites=prereqs,
        )

    def _build_optimize_disruption_plan(
        self,
        request: GoalRequest,
        default_image_path: Optional[str],
    ) -> ExperimentPlanResponse:
        plan_id = str(uuid.uuid4())
        target_img = self._resolve_image_path(request, default_image_path)
        min_ssim = request.constraints.get("min_ssim", 0.88) if request.constraints else 0.88

        steps = [
            PlanStep(
                step_number=1,
                tool_name="validate_image_input",
                description="Inspect input image dimensions and quality baseline.",
                parameters={"image_path": target_img},
                requires_approval=False,
            ),
            PlanStep(
                step_number=2,
                tool_name="run_adversarial_cloaking",
                description=f"Phase 1: Bounded cloaking with adaptive step size to respect SSIM >= {min_ssim}.",
                parameters={"image_path": target_img, "epsilon": 8.0 / 255.0, "iterations": 25, "step_size": 1.5 / 255.0},
                requires_approval=True,
            ),
            PlanStep(
                step_number=3,
                tool_name="calculate_quality_metrics",
                description="Verify SSIM constraint satisfaction on candidate perturbation.",
                parameters={"original_image_path": target_img},
                requires_approval=True,
            ),
            PlanStep(
                step_number=4,
                tool_name="evaluate_cross_model_transferability",
                description="Measure resulting disruption rate across multi-model pool.",
                parameters={"original_image_path": target_img},
                requires_approval=True,
            ),
        ]

        return ExperimentPlanResponse(
            plan_id=plan_id,
            title="Constrained Disruption Optimization",
            goal=request.goal,
            objective_type="optimize_disruption",
            status="proposed",
            requires_approval=True,
            approval_status="pending",
            config_hash="",
            steps=steps,
            resource_estimate=ResourceEstimate(
                gpu_vram_gb=1.3,
                estimated_runtime_seconds=22.0,
                disk_mb=2.5,
                hardware_device="cuda",
            ),
            success_criteria=[
                SuccessCriteria(
                    target_metric="ssim",
                    condition=f"ssim >= {min_ssim}",
                    threshold=min_ssim,
                    description="Perceptual fidelity constraint satisfied.",
                ),
                SuccessCriteria(
                    target_metric="disruption_rate",
                    condition="disruption_rate_pct >= 50.0",
                    threshold=50.0,
                    description="Achieves significant cross-model disruption.",
                ),
            ],
            missing_prerequisites=[] if target_img else ["Target artwork image"],
        )

    def _build_cross_transfer_plan(
        self,
        request: GoalRequest,
        default_image_path: Optional[str],
    ) -> ExperimentPlanResponse:
        plan_id = str(uuid.uuid4())
        target_img = self._resolve_image_path(request, default_image_path)

        steps = [
            PlanStep(
                step_number=1,
                tool_name="check_model_availability",
                description="Audit readiness of ConvNeXt-Tiny, ViT-B/16, and CLIP-ViT-B/32.",
                parameters={},
                requires_approval=False,
            ),
            PlanStep(
                step_number=2,
                tool_name="run_adversarial_cloaking",
                description="Synthesize perturbation using ResNet-50 surrogate.",
                parameters={"image_path": target_img, "epsilon": 8.0 / 255.0, "iterations": 20},
                requires_approval=True,
            ),
            PlanStep(
                step_number=3,
                tool_name="evaluate_cross_model_transferability",
                description="Compute black-box disruption and cosine embedding distance across pool.",
                parameters={"original_image_path": target_img},
                requires_approval=True,
            ),
        ]

        return ExperimentPlanResponse(
            plan_id=plan_id,
            title="Cross-Architecture Transferability Benchmark",
            goal=request.goal,
            objective_type="cross_transfer",
            status="proposed",
            requires_approval=True,
            approval_status="pending",
            config_hash="",
            steps=steps,
            resource_estimate=ResourceEstimate(
                gpu_vram_gb=1.4,
                estimated_runtime_seconds=16.0,
                disk_mb=2.0,
                hardware_device="cuda",
            ),
            success_criteria=[
                SuccessCriteria(
                    target_metric="clip_disruption",
                    condition="clip_cosine_distance > 0.02",
                    threshold=0.02,
                    description="Multimodal embedding drift detected on CLIP.",
                ),
            ],
            missing_prerequisites=[] if target_img else ["Target artwork image"],
        )

    def _build_robustness_plan(
        self,
        request: GoalRequest,
        default_image_path: Optional[str],
    ) -> ExperimentPlanResponse:
        plan_id = str(uuid.uuid4())
        target_img = self._resolve_image_path(request, default_image_path)

        steps = [
            PlanStep(
                step_number=1,
                tool_name="run_adversarial_cloaking",
                description="Generate cloaked asset for transformation endurance benchmarking.",
                parameters={"image_path": target_img, "epsilon": 8.0 / 255.0, "iterations": 20},
                requires_approval=True,
            ),
            PlanStep(
                step_number=2,
                tool_name="evaluate_digital_robustness",
                description="Benchmark survival rate across 7 digital operations (JPEG, resize, crop, blur, photometric).",
                parameters={"original_image_path": target_img},
                requires_approval=True,
            ),
        ]

        return ExperimentPlanResponse(
            plan_id=plan_id,
            title="Digital Transformation Robustness Suite",
            goal=request.goal,
            objective_type="robustness_benchmark",
            status="proposed",
            requires_approval=True,
            approval_status="pending",
            config_hash="",
            steps=steps,
            resource_estimate=ResourceEstimate(
                gpu_vram_gb=1.2,
                estimated_runtime_seconds=15.0,
                disk_mb=2.0,
                hardware_device="cuda",
            ),
            success_criteria=[
                SuccessCriteria(
                    target_metric="overall_survival_rate",
                    condition="survival_rate_pct >= 50.0",
                    threshold=50.0,
                    description="Perturbation survives common online transformations.",
                ),
            ],
            missing_prerequisites=[] if target_img else ["Target artwork image"],
        )

    def _build_model_investigation_plan(self, request: GoalRequest) -> ExperimentPlanResponse:
        plan_id = str(uuid.uuid4())
        steps = [
            PlanStep(
                step_number=1,
                tool_name="check_model_availability",
                description="Inspect all model weights, checkpoints, parameter sizes, and device placement.",
                parameters={},
                requires_approval=False,
            ),
            PlanStep(
                step_number=2,
                tool_name="inspect_pipeline",
                description="Check PyTorch and CUDA runtime compatibility.",
                parameters={},
                requires_approval=False,
            ),
        ]

        return ExperimentPlanResponse(
            plan_id=plan_id,
            title="Model Diagnostics & Integrity Investigation",
            goal=request.goal,
            objective_type="model_investigation",
            status="proposed",
            requires_approval=False,
            approval_status="approved",
            config_hash="",
            steps=steps,
            resource_estimate=ResourceEstimate(
                gpu_vram_gb=0.2,
                estimated_runtime_seconds=2.0,
                disk_mb=0.0,
                hardware_device="cpu",
            ),
            success_criteria=[
                SuccessCriteria(
                    target_metric="diagnostic_completion",
                    condition="all_models_audited == True",
                    threshold=1.0,
                    description="Diagnostic status confirmed for each model architecture.",
                ),
            ],
            missing_prerequisites=[],
        )

    def _build_recommend_next_plan(self, request: GoalRequest) -> ExperimentPlanResponse:
        plan_id = str(uuid.uuid4())
        steps = [
            PlanStep(
                step_number=1,
                tool_name="retrieve_and_compare_experiments",
                description="Retrieve past experiment runs to identify weakest disruption vector.",
                parameters={"experiment_ids": []},
                requires_approval=False,
            ),
        ]

        return ExperimentPlanResponse(
            plan_id=plan_id,
            title="Empirical Next-Experiment Recommendation",
            goal=request.goal,
            objective_type="recommend_next",
            status="proposed",
            requires_approval=False,
            approval_status="approved",
            config_hash="",
            steps=steps,
            resource_estimate=ResourceEstimate(
                gpu_vram_gb=0.0,
                estimated_runtime_seconds=1.0,
                disk_mb=0.1,
                hardware_device="cpu",
            ),
            success_criteria=[
                SuccessCriteria(
                    target_metric="recommendation_generated",
                    condition="grounded_recommendation == True",
                    threshold=1.0,
                    description="Actionable hypothesis and follow-up plan synthesized.",
                ),
            ],
            missing_prerequisites=[],
        )

    def _resolve_image_path(
        self,
        request: GoalRequest,
        default_image_path: Optional[str],
    ) -> Optional[str]:
        if request.context and "image_path" in request.context:
            return request.context["image_path"]
        if default_image_path and Path(default_image_path).exists():
            return default_image_path
        # Look in upload dir or sample
        fallback = Path("tests/fixtures/sample_artwork.jpg")
        if fallback.exists():
            return str(fallback)
        fallback2 = Path("data/uploads/sample_artwork.jpg")
        if fallback2.exists():
            return str(fallback2)
        return None
