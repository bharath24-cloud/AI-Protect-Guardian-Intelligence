import pytest
from pathlib import Path
from backend.app.guardian.tool_registry import ToolRegistry, default_tool_registry
from backend.app.guardian.schemas import ToolResult


def test_tool_registry_registration():
    reg = ToolRegistry()
    tools = reg.list_tools()
    tool_names = [t["name"] for t in tools]

    expected = [
        "inspect_pipeline",
        "check_model_availability",
        "validate_image_input",
        "estimate_experiment_resources",
        "evaluate_baseline_image",
        "run_adversarial_cloaking",
        "calculate_quality_metrics",
        "evaluate_cross_model_transferability",
        "evaluate_digital_robustness",
        "retrieve_and_compare_experiments",
        "generate_research_report_tool",
    ]

    for exp in expected:
        assert exp in tool_names, f"Tool '{exp}' missing from registry."


def test_inspect_pipeline_execution():
    res = default_tool_registry.execute_tool("inspect_pipeline")
    assert isinstance(res, ToolResult)
    assert res.status == "success"
    assert "pytorch_version" in res.measurements
    assert "cuda_available" in res.measurements


def test_check_model_availability():
    res = default_tool_registry.execute_tool("check_model_availability")
    assert isinstance(res, ToolResult)
    assert res.status == "success"
    models = res.measurements.get("models", {})
    assert "resnet50" in models
    assert "convnext_tiny" in models
    assert "vit_b_16" in models
    assert "clip_vit_b32" in models


def test_validate_image_input_missing():
    res = default_tool_registry.execute_tool("validate_image_input", image_path="nonexistent_file_xyz.png")
    assert isinstance(res, ToolResult)
    assert res.status == "failed"
    assert res.error["type"] == "IMAGE_NOT_FOUND"


def test_estimate_experiment_resources():
    res = default_tool_registry.execute_tool("estimate_experiment_resources", iterations=20, model_count=4)
    assert isinstance(res, ToolResult)
    assert res.status == "success"
    assert res.measurements["gpu_vram_gb"] > 0
    assert res.measurements["estimated_runtime_seconds"] > 0
    assert res.measurements["hardware_device"] in ["cuda", "cpu"]


def test_unregistered_tool_call():
    res = default_tool_registry.execute_tool("nonexistent_arbitrary_tool_call")
    assert isinstance(res, ToolResult)
    assert res.status == "failed"
    assert res.error["type"] == "TOOL_NOT_FOUND"
