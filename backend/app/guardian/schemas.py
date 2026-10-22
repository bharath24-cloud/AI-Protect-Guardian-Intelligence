from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class GoalRequest(BaseModel):
    """User request stating an experimental or diagnostic goal."""
    goal: str = Field(..., description="High-level goal or query for Guardian Intelligence.")
    image_id: Optional[str] = Field(None, description="Optional target image ID in database.")
    constraints: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Constraints (e.g. min SSIM, max epsilon).")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context or parameters.")


class PlanStep(BaseModel):
    """Single discrete step in an experiment plan."""
    step_number: int
    tool_name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False
    status: str = "pending"  # pending, running, completed, failed, skipped


class ResourceEstimate(BaseModel):
    """Estimated resource requirements for an experiment plan."""
    gpu_vram_gb: float = Field(0.0, description="Estimated peak VRAM in GB.")
    estimated_runtime_seconds: float = Field(0.0, description="Estimated execution time.")
    disk_mb: float = Field(0.0, description="Estimated disk storage in MB.")
    hardware_device: str = Field("cuda" if True else "cpu", description="Target execution device.")


class SuccessCriteria(BaseModel):
    """Measurable empirical success condition."""
    target_metric: str
    condition: str
    threshold: float
    description: str


class ExperimentPlanResponse(BaseModel):
    """Structured, reproducible plan produced by the Adaptive Planner."""
    plan_id: str
    version: int = 1
    title: str
    goal: str
    objective_type: str
    status: str = "proposed"  # proposed, awaiting_approval, approved, queued, running, completed, failed, blocked
    requires_approval: bool = True
    approval_status: str = "pending"  # pending, approved, rejected, invalidated
    config_hash: str
    steps: List[PlanStep] = Field(default_factory=list)
    resource_estimate: ResourceEstimate
    success_criteria: List[SuccessCriteria] = Field(default_factory=list)
    missing_prerequisites: List[str] = Field(default_factory=list)
    clarification_needed: Optional[str] = None
    created_at: Optional[str] = None


class PlanApprovalRequest(BaseModel):
    """Explicit user authorization for compute-heavy or impactful experiments."""
    plan_id: str
    config_hash: str = Field(..., description="Hash of the exact plan being authorized.")
    decision: str = Field("approved", description="'approved' or 'rejected'")
    approved_by: str = Field("user", description="Authorizing identity.")
    notes: Optional[str] = None


class PlanExecutionRequest(BaseModel):
    """Request to launch execution of an approved plan."""
    plan_id: str
    config_hash: str


class ToolResult(BaseModel):
    """Standardized response from registered ML tools."""
    status: str = Field(..., description="success, failed, skipped, unavailable")
    tool_name: str
    operation_id: str
    experiment_id: Optional[str] = None
    output_references: Dict[str, Any] = Field(default_factory=dict)
    measurements: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None


class TimelineEvent(BaseModel):
    """Single progress update in an agent timeline."""
    timestamp: str
    step_number: int
    tool_name: str
    status: str
    message: str
    measurements: Optional[Dict[str, Any]] = None


class EvidenceSummary(BaseModel):
    """Grounded empirical evidence, hypotheses, and recommendations."""
    plan_id: str
    execution_id: Optional[str] = None
    cited_experiment_ids: List[str] = Field(default_factory=list)
    empirical_observations: List[str] = Field(default_factory=list)
    hypotheses_tested: List[str] = Field(default_factory=list)
    grounded_recommendations: List[str] = Field(default_factory=list)
    uncertainties_and_caveats: List[str] = Field(default_factory=list)
    disclaimer: str = (
        "Empirical finding: Adversarial cloaking produces measured disruption on evaluated models. "
        "It does not guarantee mathematical cryptography or universal immunity against unseen models."
    )


class ExperimentComparisonRequest(BaseModel):
    """Request to compare two or more experiment runs."""
    experiment_ids: List[str]


class GuardianChatRequest(BaseModel):
    """Interactive chat input to Guardian Intelligence."""
    message: str
    plan_id: Optional[str] = None
    image_id: Optional[str] = None
    session_id: Optional[str] = None


class GuardianChatResponse(BaseModel):
    """Conversational response from Guardian Intelligence."""
    reply: str
    intent: str  # plan_proposed, status_inquiry, general_response, clarification_needed, execution_started
    plan: Optional[ExperimentPlanResponse] = None
    evidence: Optional[EvidenceSummary] = None
    suggested_actions: List[str] = Field(default_factory=list)
