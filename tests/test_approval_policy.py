import pytest
from backend.app.guardian.schemas import PlanStep, ResourceEstimate
from backend.app.guardian.policies import PolicyEngine


def test_tool_classification_read_only_vs_compute():
    read_only = ["inspect_pipeline", "check_model_availability", "validate_image_input", "estimate_experiment_resources"]
    compute_heavy = ["run_adversarial_cloaking", "evaluate_baseline_image", "evaluate_cross_model_transferability", "evaluate_digital_robustness"]

    for t in read_only:
        assert PolicyEngine.is_read_only_tool(t) is True
        step = PlanStep(step_number=1, tool_name=t, description="test")
        assert PolicyEngine.step_requires_approval(step) is False

    for t in compute_heavy:
        assert PolicyEngine.is_read_only_tool(t) is False
        step = PlanStep(step_number=1, tool_name=t, description="test")
        assert PolicyEngine.step_requires_approval(step) is True


def test_deterministic_config_hash():
    plan_dict_1 = {
        "objective_type": "evaluate_cloaking",
        "goal": "Evaluate cloaking on artwork",
        "steps": [
            {"step_number": 1, "tool_name": "validate_image_input", "parameters": {"image_path": "test.jpg"}},
            {"step_number": 2, "tool_name": "run_adversarial_cloaking", "parameters": {"epsilon": 0.03137}},
        ],
    }

    plan_dict_2 = {
        "objective_type": "evaluate_cloaking",
        "goal": "Evaluate cloaking on artwork",
        "steps": [
            {"step_number": 1, "tool_name": "validate_image_input", "parameters": {"image_path": "test.jpg"}},
            {"step_number": 2, "tool_name": "run_adversarial_cloaking", "parameters": {"epsilon": 0.03137}},
        ],
    }

    # Different parameters should yield different hash
    plan_dict_tampered = {
        "objective_type": "evaluate_cloaking",
        "goal": "Evaluate cloaking on artwork",
        "steps": [
            {"step_number": 1, "tool_name": "validate_image_input", "parameters": {"image_path": "test.jpg"}},
            {"step_number": 2, "tool_name": "run_adversarial_cloaking", "parameters": {"epsilon": 0.06274}},  # modified!
        ],
    }

    hash_1 = PolicyEngine.compute_config_hash(plan_dict_1)
    hash_2 = PolicyEngine.compute_config_hash(plan_dict_2)
    hash_tampered = PolicyEngine.compute_config_hash(plan_dict_tampered)

    assert len(hash_1) == 64
    assert hash_1 == hash_2
    assert hash_1 != hash_tampered


def test_approval_hash_validation():
    valid_hash = "a" * 64
    tampered_hash = "b" * 64

    # Matches
    ok, msg = PolicyEngine.validate_approval_hash(valid_hash, valid_hash)
    assert ok is True
    assert "verified" in msg.lower()

    # Mismatch
    ok, msg = PolicyEngine.validate_approval_hash(valid_hash, tampered_hash)
    assert ok is False
    assert "tampering detected" in msg.lower()

    # Empty approval hash
    ok, msg = PolicyEngine.validate_approval_hash(valid_hash, "")
    assert ok is False


def test_resource_limits_enforcement():
    # Acceptable VRAM
    res_valid = ResourceEstimate(gpu_vram_gb=1.5, estimated_runtime_seconds=20.0, disk_mb=5.0)
    ok, err = PolicyEngine.check_resource_limits(res_valid)
    assert ok is True
    assert err is None

    # Exceeding VRAM (limit is 3.6 GB)
    res_excess = ResourceEstimate(gpu_vram_gb=4.5, estimated_runtime_seconds=20.0, disk_mb=5.0)
    ok, err = PolicyEngine.check_resource_limits(res_excess)
    # If host has cuda available, it triggers ceiling check
    import torch
    if torch.cuda.is_available():
        assert ok is False
        assert "exceeding safety threshold" in err.lower()


def test_state_transitions():
    valid_cases = [
        ("proposed", "awaiting_approval"),
        ("proposed", "approved"),
        ("awaiting_approval", "approved"),
        ("awaiting_approval", "rejected"),
        ("approved", "queued"),
        ("queued", "running"),
        ("running", "completed"),
        ("running", "failed"),
    ]

    for curr, nxt in valid_cases:
        ok, msg = PolicyEngine.validate_state_transition(curr, nxt)
        assert ok is True, f"Failed valid transition {curr} -> {nxt}: {msg}"

    invalid_cases = [
        ("completed", "running"),
        ("completed", "queued"),
        ("proposed", "completed"),
        ("awaiting_approval", "running"),
        ("rejected", "running"),
    ]

    for curr, nxt in invalid_cases:
        ok, msg = PolicyEngine.validate_state_transition(curr, nxt)
        assert ok is False, f"Allowed invalid transition {curr} -> {nxt}"
