import hashlib
import json
import shutil
from typing import Dict, List, Any, Tuple, Optional
import torch

from backend.app.guardian.schemas import ResourceEstimate, PlanStep


class PolicyEngine:
    """
    Enforces authorization gates, compute limits, state transitions,
    and cryptographic configuration-hash bindings for Guardian plans.
    """

    # Maximum permitted VRAM usage for the RTX 3050 Laptop GPU (4.0 GB)
    MAX_PERMITTED_VRAM_GB: float = 3.6
    # Minimum free disk space in megabytes required to initiate cloaking
    MIN_FREE_DISK_MB: float = 200.0

    # Tools considered read-only that require no compute approval
    READ_ONLY_TOOLS = {
        "inspect_pipeline",
        "check_model_availability",
        "validate_image_input",
        "estimate_experiment_resources",
        "retrieve_and_compare_experiments",
        "generate_research_report_tool",
    }

    # Tools that require explicit user approval due to GPU compute or mutation
    COMPUTE_INTENSIVE_TOOLS = {
        "evaluate_baseline_image",
        "run_adversarial_cloaking",
        "calculate_quality_metrics",
        "evaluate_cross_model_transferability",
        "evaluate_digital_robustness",
    }

    # Valid lifecycle status transitions
    VALID_TRANSITIONS = {
        "proposed": {"awaiting_approval", "approved", "rejected", "blocked"},
        "awaiting_approval": {"approved", "rejected", "blocked"},
        "approved": {"queued", "running", "rejected"},
        "queued": {"running", "cancelled", "failed"},
        "running": {"completed", "failed", "cancelled"},
        "completed": set(),  # terminal
        "failed": {"queued"},  # allows retry
        "cancelled": set(),  # terminal
        "rejected": set(),  # terminal
        "blocked": {"proposed"},
    }

    @classmethod
    def is_read_only_tool(cls, tool_name: str) -> bool:
        return tool_name in cls.READ_ONLY_TOOLS

    @classmethod
    def step_requires_approval(cls, step: PlanStep) -> bool:
        """Determines if a step or its parameters require user authorization."""
        if step.tool_name in cls.COMPUTE_INTENSIVE_TOOLS:
            return True
        # Read-only tools never require approval
        return False

    @classmethod
    def plan_requires_approval(cls, steps: List[PlanStep]) -> bool:
        """Returns True if ANY step in the plan requires explicit authorization."""
        return any(cls.step_requires_approval(s) for s in steps)

    @classmethod
    def compute_config_hash(cls, plan_dict: Dict[str, Any]) -> str:
        """
        Computes a deterministic SHA-256 fingerprint from the plan's objective,
        steps, tool names, and parameter dictionaries.
        """
        normalized = {
            "objective_type": plan_dict.get("objective_type", ""),
            "goal": plan_dict.get("goal", "").strip().lower(),
            "steps": [
                {
                    "step_number": s.get("step_number") if isinstance(s, dict) else s.step_number,
                    "tool_name": s.get("tool_name") if isinstance(s, dict) else s.tool_name,
                    "parameters": s.get("parameters", {}) if isinstance(s, dict) else s.parameters,
                }
                for s in plan_dict.get("steps", [])
            ],
        }
        serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @classmethod
    def validate_approval_hash(
        cls,
        current_config_hash: str,
        authorized_config_hash: str,
    ) -> Tuple[bool, str]:
        """
        Verifies that the plan has not been modified since the user authorized it.
        """
        if not authorized_config_hash:
            return False, "Plan has not been authorized by user."
        if current_config_hash != authorized_config_hash:
            return False, (
                f"Configuration tampering detected: Plan hash '{current_config_hash[:10]}...' "
                f"does not match authorized hash '{authorized_config_hash[:10]}...'. Re-approval required."
            )
        return True, "Authorization hash verified."

    @classmethod
    def check_resource_limits(
        cls,
        estimate: ResourceEstimate,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates whether the estimated resource consumption adheres to host limits.
        """
        # 1. GPU VRAM check
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            total_vram_gb = device_props.total_memory / (1024 ** 3)
            if estimate.gpu_vram_gb > cls.MAX_PERMITTED_VRAM_GB:
                return False, (
                    f"Plan requires {estimate.gpu_vram_gb:.2f} GB VRAM, exceeding safety threshold "
                    f"of {cls.MAX_PERMITTED_VRAM_GB:.2f} GB on {device_props.name} ({total_vram_gb:.2f} GB total)."
                )

        # 2. Disk space check
        total, used, free = shutil.disk_usage(".")
        free_mb = free / (1024 * 1024)
        if free_mb < cls.MIN_FREE_DISK_MB:
            return False, f"Insufficient disk space: {free_mb:.1f} MB available, minimum {cls.MIN_FREE_DISK_MB} MB required."

        return True, None

    @classmethod
    def validate_state_transition(cls, current_status: str, target_status: str) -> Tuple[bool, str]:
        """Ensures state machine transitions follow valid paths."""
        valid_targets = cls.VALID_TRANSITIONS.get(current_status, set())
        if target_status not in valid_targets and current_status != target_status:
            return False, f"Invalid transition from '{current_status}' to '{target_status}'."
        return True, f"Valid transition to '{target_status}'."
