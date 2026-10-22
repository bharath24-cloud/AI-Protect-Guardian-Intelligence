from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database import get_db
from backend.app.memory.repository import GuardianMemoryRepository
from backend.app.guardian.schemas import (
    GoalRequest,
    ExperimentPlanResponse,
    PlanApprovalRequest,
    PlanExecutionRequest,
    ExperimentComparisonRequest,
    GuardianChatRequest,
    GuardianChatResponse,
    EvidenceSummary,
)
from backend.app.guardian.orchestrator import default_guardian_orchestrator
from backend.app.guardian.policies import PolicyEngine

router = APIRouter(prefix="/guardian", tags=["Guardian Intelligence"])


@router.post("/chat", response_model=GuardianChatResponse)
async def chat_with_guardian(request: GuardianChatRequest):
    """
    Interactive dialogue endpoint with Guardian Intelligence.
    Understands intent, formulates experiment plans, or answers diagnostic questions.
    """
    try:
        response = await default_guardian_orchestrator.handle_chat(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Guardian chat processing error: {str(e)}",
        )


@router.post("/plan", response_model=ExperimentPlanResponse)
async def create_experiment_plan(request: GoalRequest):
    """
    Translates an experimental goal into a validated, reproducible,
    and policy-checked experiment plan.
    """
    try:
        plan = await default_guardian_orchestrator.create_plan(request)
        return plan
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Plan creation failed: {str(e)}",
        )


@router.get("/plans/{plan_id}")
async def get_plan(plan_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieves an existing experiment plan by its ID."""
    repo = GuardianMemoryRepository(db)
    plan_rec = await repo.get_plan(plan_id)
    if not plan_rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Plan '{plan_id}' not found.")
    return {
        "plan_id": plan_rec.id,
        "title": plan_rec.title,
        "goal": plan_rec.goal,
        "objective_type": plan_rec.objective_type,
        "status": plan_rec.status,
        "requires_approval": plan_rec.requires_approval,
        "approval_status": plan_rec.approval_status,
        "config_hash": plan_rec.config_hash,
        "plan_json": plan_rec.plan_json,
        "created_at": plan_rec.created_at.isoformat() if plan_rec.created_at else None,
        "updated_at": plan_rec.updated_at.isoformat() if plan_rec.updated_at else None,
    }


@router.get("/plans")
async def list_plans(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Lists saved experiment plans ordered by creation time."""
    repo = GuardianMemoryRepository(db)
    plans = await repo.list_plans(limit=limit, offset=offset)
    return [
        {
            "plan_id": p.id,
            "title": p.title,
            "goal": p.goal,
            "objective_type": p.objective_type,
            "status": p.status,
            "requires_approval": p.requires_approval,
            "approval_status": p.approval_status,
            "config_hash": p.config_hash,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in plans
    ]


@router.post("/approve")
async def approve_plan(request: PlanApprovalRequest):
    """
    Records explicit user authorization for an experiment plan,
    strictly binding the approval to the configuration hash.
    """
    result = await default_guardian_orchestrator.approve_plan(request)
    if not result.get("success"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get("error"))
    return result


@router.post("/execute")
async def execute_plan(request: PlanExecutionRequest):
    """
    Launches execution of an approved experiment plan in the background.
    """
    result = await default_guardian_orchestrator.execute_plan(request)
    if not result.get("success"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get("error"))
    return result


@router.get("/jobs/{execution_id}")
async def get_execution_status(execution_id: str, db: AsyncSession = Depends(get_db)):
    """
    Polls the live execution state, progress, timeline logs, and measurements.
    """
    repo = GuardianMemoryRepository(db)
    exec_rec = await repo.get_execution(execution_id)
    if not exec_rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution '{execution_id}' not found.",
        )

    evidence_summary = None
    if exec_rec.evidence:
        evidence_summary = {
            "cited_experiment_ids": exec_rec.evidence.cited_experiment_ids,
            "empirical_observations": exec_rec.evidence.empirical_observations,
            "hypotheses_tested": exec_rec.evidence.hypotheses_tested,
            "grounded_recommendations": exec_rec.evidence.grounded_recommendations,
            "uncertainties_and_caveats": exec_rec.evidence.uncertainties_and_caveats,
        }

    return {
        "execution_id": exec_rec.id,
        "plan_id": exec_rec.plan_id,
        "plan_title": exec_rec.plan.title if exec_rec.plan else None,
        "status": exec_rec.status,
        "current_step": exec_rec.current_step,
        "total_steps": exec_rec.total_steps,
        "progress": round((exec_rec.current_step / max(1, exec_rec.total_steps)) * 100.0, 1),
        "logs": exec_rec.logs,
        "results": exec_rec.results_json,
        "evidence": evidence_summary,
        "error_message": exec_rec.error_message,
        "started_at": exec_rec.started_at.isoformat() if exec_rec.started_at else None,
        "completed_at": exec_rec.completed_at.isoformat() if exec_rec.completed_at else None,
    }


@router.get("/evidence/{plan_id}")
async def get_plan_evidence(plan_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieves empirical evidence and recommendations synthesized for a plan."""
    repo = GuardianMemoryRepository(db)
    evi = await repo.get_evidence_by_plan(plan_id)
    if not evi:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No evidence record registered for plan '{plan_id}'.",
        )
    return {
        "plan_id": evi.plan_id,
        "execution_id": evi.execution_id,
        "cited_experiment_ids": evi.cited_experiment_ids,
        "empirical_observations": evi.empirical_observations,
        "hypotheses_tested": evi.hypotheses_tested,
        "grounded_recommendations": evi.grounded_recommendations,
        "uncertainties_and_caveats": evi.uncertainties_and_caveats,
        "created_at": evi.created_at.isoformat() if evi.created_at else None,
    }


@router.get("/experiments")
async def list_experiments(limit: int = Query(50, ge=1, le=100), db: AsyncSession = Depends(get_db)):
    """Lists historical protection jobs and guardian executions for comparison."""
    repo = GuardianMemoryRepository(db)
    return await repo.list_experiments(limit=limit)


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieves comprehensive measurements for a single experiment."""
    repo = GuardianMemoryRepository(db)
    exp = await repo.get_experiment_by_id(experiment_id)
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found.",
        )
    return exp


@router.post("/compare")
async def compare_experiments(request: ExperimentComparisonRequest, db: AsyncSession = Depends(get_db)):
    """Compares two or more experiments side-by-side with metric delta analysis."""
    repo = GuardianMemoryRepository(db)
    result = await repo.compare_experiments(request.experiment_ids)
    if "error" in result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["error"])
    return result


@router.get("/models")
async def get_model_status():
    """Returns operational status and checkpoint availability of vision models."""
    res = default_guardian_orchestrator.tool_registry.execute_tool("check_model_availability")
    return res.measurements


@router.get("/tools")
async def list_registered_tools():
    """Lists registered tools, capabilities, and approval policy classifications."""
    tools = default_guardian_orchestrator.tool_registry.list_tools()
    annotated = []
    for t in tools:
        name = t["name"]
        annotated.append({
            "name": name,
            "description": t["description"],
            "is_read_only": PolicyEngine.is_read_only_tool(name),
            "requires_approval": name in PolicyEngine.COMPUTE_INTENSIVE_TOOLS,
        })
    return annotated
