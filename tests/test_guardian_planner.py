import pytest
from backend.app.guardian.schemas import GoalRequest
from backend.app.guardian.planner import AdaptiveExperimentPlanner
from backend.app.guardian.tool_registry import ToolRegistry


def test_planner_objective_classification():
    planner = AdaptiveExperimentPlanner()

    cases = [
        ("Evaluate cloaking on sample artwork", "evaluate_cloaking"),
        ("Compare experiment 12345678-1234-1234-1234-123456789abc and 87654321-4321-4321-4321-cba987654321", "compare_experiments"),
        ("Optimize disruption subject to SSIM >= 0.85 constraint", "optimize_disruption"),
        ("Test cross-model transferability on ViT and ConvNeXt", "cross_transfer"),
        ("Evaluate robustness under JPEG compression and resizing", "robustness_benchmark"),
        ("Investigate why CLIP model is unavailable or failing", "model_investigation"),
        ("Recommend the next experiment based on previous results", "recommend_next"),
    ]

    for text, expected_type in cases:
        obj_type, conf = planner.parse_objective(text)
        assert obj_type == expected_type, f"Failed for '{text}': got {obj_type}, expected {expected_type}"
        assert conf >= 0.80


def test_planner_ambiguous_request():
    planner = AdaptiveExperimentPlanner()
    req = GoalRequest(goal="hi")
    plan = planner.create_plan(req)

    assert plan.objective_type == "ambiguous"
    assert plan.clarification_needed is not None
    assert "ambiguous" in plan.clarification_needed.lower()
    assert plan.status == "blocked"


def test_planner_evaluate_cloaking_structure():
    planner = AdaptiveExperimentPlanner()
    req = GoalRequest(
        goal="Synthesize adversarial cloak on artwork and evaluate vision models",
        constraints={"epsilon": 0.03137, "iterations": 20},
    )
    plan = planner.create_plan(req)

    assert plan.objective_type == "evaluate_cloaking"
    assert len(plan.steps) == 5
    assert plan.requires_approval is True
    assert plan.status == "awaiting_approval"
    assert len(plan.config_hash) == 64
    assert plan.resource_estimate.gpu_vram_gb > 0
    assert plan.resource_estimate.estimated_runtime_seconds > 0

    tool_names = [s.tool_name for s in plan.steps]
    assert "validate_image_input" in tool_names
    assert "evaluate_baseline_image" in tool_names
    assert "run_adversarial_cloaking" in tool_names
    assert "calculate_quality_metrics" in tool_names
    assert "evaluate_cross_model_transferability" in tool_names

    crit_metrics = [c.target_metric for c in plan.success_criteria]
    assert "top1_flip" in crit_metrics
    assert "ssim" in crit_metrics


def test_planner_compare_experiments_read_only():
    planner = AdaptiveExperimentPlanner()
    req = GoalRequest(
        goal="Compare experiment 11111111-1111-1111-1111-111111111111 and 22222222-2222-2222-2222-222222222222",
    )
    plan = planner.create_plan(req)

    assert plan.objective_type == "compare_experiments"
    # Compare experiment is read-only, does not require approval
    assert plan.requires_approval is False
    assert plan.status == "approved"
    assert plan.resource_estimate.gpu_vram_gb == 0.0


def test_planner_resource_estimation():
    planner = AdaptiveExperimentPlanner()
    req = GoalRequest(goal="Benchmark robustness under transformations")
    plan = planner.create_plan(req)

    assert plan.resource_estimate.gpu_vram_gb <= 3.6
    assert plan.resource_estimate.estimated_runtime_seconds > 0
    assert plan.resource_estimate.hardware_device in ["cuda", "cpu"]
