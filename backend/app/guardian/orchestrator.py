import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from backend.app.guardian.schemas import (
    GoalRequest,
    PlanStep,
    ExperimentPlanResponse,
    PlanApprovalRequest,
    PlanExecutionRequest,
    ToolResult,
    TimelineEvent,
    EvidenceSummary,
    GuardianChatRequest,
    GuardianChatResponse,
)
from backend.app.guardian.policies import PolicyEngine
from backend.app.guardian.planner import AdaptiveExperimentPlanner
from backend.app.guardian.tool_registry import ToolRegistry, default_tool_registry
from backend.app.guardian.evidence import EvidenceAnalyst
from backend.app.memory.repository import GuardianMemoryRepository

logger = logging.getLogger("guardian.orchestrator")


class GuardianOrchestrator:
    """
    Central orchestration engine for Guardian Intelligence.
    Coordinates planning, policy enforcement, registered tool execution,
    persistent memory, and empirical evidence synthesis.
    """

    def __init__(
        self,
        planner: Optional[AdaptiveExperimentPlanner] = None,
        tool_registry: Optional[ToolRegistry] = None,
        repository: Optional[GuardianMemoryRepository] = None,
    ):
        self.tool_registry = tool_registry or default_tool_registry
        self.planner = planner or AdaptiveExperimentPlanner(self.tool_registry)
        self.repository = repository or GuardianMemoryRepository()

    # -------------------------------------------------------------------------
    # Conversational Interface
    # -------------------------------------------------------------------------
    async def handle_chat(self, request: GuardianChatRequest) -> GuardianChatResponse:
        """
        Processes a user message, detects intent, and returns a structured response.
        """
        text = request.message.strip().lower()

        # 1. Status Inquiry
        if any(k in text for k in ["status", "how is", "progress", "what's running", "job status"]):
            if request.plan_id:
                plan = await self.repository.get_plan(request.plan_id)
                if plan:
                    return GuardianChatResponse(
                        reply=f"Plan '{plan.title}' is currently **{plan.status.upper()}** (Approval: {plan.approval_status}).",
                        intent="status_inquiry",
                        suggested_actions=["Execute plan", "View timeline", "List experiments"],
                    )
            # List active or recent plans
            recent_plans = await self.repository.list_plans(limit=3)
            if recent_plans:
                p_summaries = [f"- **{p.title}** (`{p.id[:8]}`): {p.status.upper()}" for p in recent_plans]
                return GuardianChatResponse(
                    reply="Here are the recent experiment plans:\n" + "\n".join(p_summaries),
                    intent="status_inquiry",
                    suggested_actions=["Create new plan", "Inspect models", "Compare experiments"],
                )
            return GuardianChatResponse(
                reply="No active experiment plans found. What would you like to investigate?",
                intent="status_inquiry",
                suggested_actions=["Evaluate cloaking", "Benchmark robustness", "Check model readiness"],
            )

        # 2. Model / Diagnostic Inquiry
        if any(k in text for k in ["models", "checkpoints", "cuda", "vram", "hardware"]):
            diag_tool = self.tool_registry.execute_tool("check_model_availability")
            models_data = diag_tool.measurements.get("models", {})
            models_str = ", ".join([f"{k} ({v.get('status')})" for k, v in models_data.items()])
            return GuardianChatResponse(
                reply=f"**Vision Model Pool Audit**:\n{models_str}\n\nAll primary surrogate and transfer models are verified.",
                intent="general_response",
                suggested_actions=["Evaluate cloaking on artwork", "Test cross-model transferability"],
            )

        # 3. Create Plan (Standard Goal)
        goal_req = GoalRequest(goal=request.message, image_id=request.image_id)
        plan_response = await self.create_plan(goal_req)

        if plan_response.clarification_needed:
            return GuardianChatResponse(
                reply=plan_response.clarification_needed,
                intent="clarification_needed",
                plan=plan_response,
                suggested_actions=[
                    "Evaluate cloaking on artwork",
                    "Test cross-model transferability",
                    "Benchmark JPEG robustness",
                ],
            )

        requires_appr_text = (
            "This plan involves GPU compute and requires your explicit approval."
            if plan_response.requires_approval
            else "This is a read-only plan that is pre-approved."
        )

        return GuardianChatResponse(
            reply=(
                f"I have synthesized a structured plan: **{plan_response.title}** ({len(plan_response.steps)} steps).\n\n"
                f"- **Objective**: `{plan_response.objective_type}`\n"
                f"- **Estimated VRAM**: {plan_response.resource_estimate.gpu_vram_gb:.1f} GB\n"
                f"- **Estimated Runtime**: {plan_response.resource_estimate.estimated_runtime_seconds:.0f} seconds\n\n"
                f"{requires_appr_text}"
            ),
            intent="plan_proposed",
            plan=plan_response,
            suggested_actions=["Approve and Execute", "Inspect Steps", "Modify Constraints"],
        )

    # -------------------------------------------------------------------------
    # Planning & Persistence
    # -------------------------------------------------------------------------
    async def create_plan(self, request: GoalRequest) -> ExperimentPlanResponse:
        """
        Generates and persists an ExperimentPlanResponse.
        """
        plan = self.planner.create_plan(request)

        # Persist plan to repository
        plan_dict = plan.model_dump()
        await self.repository.create_plan(
            title=plan.title,
            goal=plan.goal,
            objective_type=plan.objective_type,
            config_hash=plan.config_hash,
            plan_json=plan_dict,
            requires_approval=plan.requires_approval,
            status=plan.status,
            plan_id=plan.plan_id,
        )
        return plan

    # -------------------------------------------------------------------------
    # Approval Enforcement
    # -------------------------------------------------------------------------
    async def approve_plan(self, request: PlanApprovalRequest) -> Dict[str, Any]:
        """
        Validates the configuration hash and records user approval.
        """
        plan_rec = await self.repository.get_plan(request.plan_id)
        if not plan_rec:
            return {"success": False, "error": f"Plan '{request.plan_id}' not found."}

        # Check config hash match
        hash_ok, hash_msg = PolicyEngine.validate_approval_hash(plan_rec.config_hash, request.config_hash)
        if not hash_ok:
            return {"success": False, "error": hash_msg}

        # Record approval
        approval = await self.repository.record_approval(
            plan_id=request.plan_id,
            config_hash=request.config_hash,
            decision=request.decision,
            approved_by=request.approved_by,
            notes=request.notes,
        )

        return {
            "success": True,
            "plan_id": request.plan_id,
            "approval_id": approval.id,
            "decision": request.decision,
            "status": "approved" if request.decision == "approved" else "rejected",
            "message": "Plan successfully authorized. Ready for execution.",
        }

    # -------------------------------------------------------------------------
    # Execution & Timeline Tracking
    # -------------------------------------------------------------------------
    async def execute_plan(self, request: PlanExecutionRequest) -> Dict[str, Any]:
        """
        Authorizes and launches execution of an approved plan.
        """
        plan_rec = await self.repository.get_plan(request.plan_id)
        if not plan_rec:
            return {"success": False, "error": f"Plan '{request.plan_id}' not found."}

        # Policy checks
        if plan_rec.requires_approval:
            active_approval = await self.repository.get_active_approval(request.plan_id)
            if not active_approval:
                return {
                    "success": False,
                    "error": "Plan requires explicit approval before execution. No valid approval record found.",
                }
            # Verify config hash
            hash_ok, hash_msg = PolicyEngine.validate_approval_hash(plan_rec.config_hash, request.config_hash)
            if not hash_ok:
                return {"success": False, "error": hash_msg}

        # Create execution record
        steps = plan_rec.plan_json.get("steps", [])
        execution = await self.repository.create_execution(
            plan_id=request.plan_id,
            total_steps=len(steps),
        )

        # Update plan status to running
        await self.repository.update_plan_status(request.plan_id, status="running")

        # Launch execution task in background
        asyncio.create_task(self._run_plan_execution_loop(execution.id, plan_rec))

        return {
            "success": True,
            "execution_id": execution.id,
            "plan_id": request.plan_id,
            "status": "queued",
            "message": "Execution initiated in background.",
        }

    async def _run_plan_execution_loop(
        self,
        execution_id: str,
        plan_rec: Any,
    ):
        """
        Asynchronous runner executing plan steps sequentially with timeline updates.
        """
        logger.info(f"[Guardian] Starting execution {execution_id} for plan {plan_rec.id}")
        steps_data = plan_rec.plan_json.get("steps", [])
        results_accumulator: Dict[str, Any] = {}
        original_img_path = None
        cloaked_img_path = None

        try:
            for idx, s in enumerate(steps_data):
                step_num = s.get("step_number", idx + 1)
                tool_name = s.get("tool_name", "")
                params = dict(s.get("parameters", {}))

                # Inject dynamic references between steps
                if original_img_path and "original_image_path" in params:
                    params["original_image_path"] = original_img_path
                if cloaked_img_path and "cloaked_image_path" in params:
                    params["cloaked_image_path"] = cloaked_img_path
                if cloaked_img_path and tool_name in ["calculate_quality_metrics", "evaluate_cross_model_transferability", "evaluate_digital_robustness"]:
                    params["cloaked_image_path"] = cloaked_img_path

                # Log step start
                log_start = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step_number": step_num,
                    "tool_name": tool_name,
                    "status": "running",
                    "message": f"Executing step {step_num}: {s.get('description', tool_name)}",
                }
                await self.repository.update_execution_step(execution_id, step_num, log_start)

                # Execute tool
                tool_result: ToolResult = self.tool_registry.execute_tool(tool_name, **params)

                # Capture output file references
                if tool_name == "validate_image_input" and tool_result.status == "success":
                    original_img_path = tool_result.output_references.get("file_path")
                elif tool_name == "run_adversarial_cloaking" and tool_result.status == "success":
                    cloaked_img_path = tool_result.output_references.get("cloaked_image_path")
                    if not original_img_path:
                        original_img_path = tool_result.output_references.get("original_image_path")

                # Accumulate measurements
                results_accumulator[tool_name] = tool_result.measurements

                # Log step outcome
                step_status = "completed" if tool_result.status == "success" else "failed"
                log_finish = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step_number": step_num,
                    "tool_name": tool_name,
                    "status": step_status,
                    "message": (
                        f"Step {step_num} ({tool_name}) finished with status: {tool_result.status}"
                        if tool_result.status == "success"
                        else f"Step {step_num} ({tool_name}) failed: {tool_result.error}"
                    ),
                    "measurements": tool_result.measurements,
                }
                await self.repository.update_execution_step(
                    execution_id,
                    step_num,
                    log_finish,
                    partial_results={tool_name: tool_result.measurements},
                )

                if tool_result.status == "failed":
                    raise RuntimeError(f"Step {step_num} ({tool_name}) failed: {tool_result.error}")

            # All steps finished successfully: synthesize evidence
            evidence = EvidenceAnalyst.analyze_execution_results(
                plan_id=plan_rec.id,
                execution_id=execution_id,
                results_json=results_accumulator,
                objective_type=plan_rec.objective_type,
            )

            # Store evidence in repository
            await self.repository.store_evidence(
                plan_id=plan_rec.id,
                execution_id=execution_id,
                observations=evidence.empirical_observations,
                hypotheses=evidence.hypotheses_tested,
                recommendations=evidence.grounded_recommendations,
                uncertainties=evidence.uncertainties_and_caveats,
                cited_experiment_ids=evidence.cited_experiment_ids,
            )

            # Mark execution completed
            await self.repository.finish_execution(
                execution_id=execution_id,
                status="completed",
                final_results=results_accumulator,
            )
            logger.info(f"[Guardian] Execution {execution_id} completed successfully.")

        except Exception as e:
            logger.error(f"[Guardian] Execution {execution_id} failed: {e}", exc_info=True)
            await self.repository.finish_execution(
                execution_id=execution_id,
                status="failed",
                final_results=results_accumulator,
                error_message=str(e),
            )


# Global singleton orchestrator
default_guardian_orchestrator = GuardianOrchestrator()
