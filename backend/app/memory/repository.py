import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from sqlalchemy import select, desc, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.database import AsyncSessionLocal
from backend.app.memory.models import (
    GuardianPlanRecord,
    GuardianApprovalRecord,
    GuardianExecutionRecord,
    GuardianEvidenceRecord,
    generate_uuid,
    utc_now,
)
from backend.app.db_models.models import (
    ProtectionJob,
    ProtectionConfig,
    EvaluationResult,
    RobustnessResult,
    ImageRecord,
)


class GuardianMemoryRepository:
    """
    Repository for persisting and retrieving Guardian Intelligence plans,
    approvals, execution timelines, empirical evidence, and experiment records.
    """

    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session

    async def _get_session(self) -> AsyncSession:
        if self._session is not None:
            return self._session
        return AsyncSessionLocal()

    # -------------------------------------------------------------------------
    # Plan Persistence
    # -------------------------------------------------------------------------
    async def create_plan(
        self,
        title: str,
        goal: str,
        objective_type: str,
        config_hash: str,
        plan_json: Dict[str, Any],
        requires_approval: bool = True,
        status: str = "proposed",
        plan_id: Optional[str] = None,
    ) -> GuardianPlanRecord:
        plan = GuardianPlanRecord(
            id=plan_id or generate_uuid(),
            version=1,
            title=title,
            goal=goal,
            objective_type=objective_type,
            status=status,
            requires_approval=requires_approval,
            approval_status="approved" if not requires_approval else "pending",
            config_hash=config_hash,
            plan_json=plan_json,
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        if self._session:
            self._session.add(plan)
            await self._session.commit()
            await self._session.refresh(plan)
            return plan

        async with AsyncSessionLocal() as session:
            session.add(plan)
            await session.commit()
            await session.refresh(plan)
            return plan

    async def get_plan(self, plan_id: str) -> Optional[GuardianPlanRecord]:
        async def _query(session: AsyncSession):
            stmt = (
                select(GuardianPlanRecord)
                .where(GuardianPlanRecord.id == plan_id)
                .options(
                    selectinload(GuardianPlanRecord.approvals),
                    selectinload(GuardianPlanRecord.executions),
                    selectinload(GuardianPlanRecord.evidence),
                )
            )
            res = await session.execute(stmt)
            return res.scalar_one_or_none()

        if self._session:
            return await _query(self._session)
        async with AsyncSessionLocal() as session:
            return await _query(session)

    async def update_plan_status(
        self,
        plan_id: str,
        status: str,
        approval_status: Optional[str] = None,
    ) -> Optional[GuardianPlanRecord]:
        async def _update(session: AsyncSession):
            stmt = select(GuardianPlanRecord).where(GuardianPlanRecord.id == plan_id)
            res = await session.execute(stmt)
            plan = res.scalar_one_or_none()
            if not plan:
                return None
            plan.status = status
            if approval_status is not None:
                plan.approval_status = approval_status
            plan.updated_at = utc_now()
            await session.commit()
            await session.refresh(plan)
            return plan

        if self._session:
            return await _update(self._session)
        async with AsyncSessionLocal() as session:
            return await _update(session)

    async def list_plans(self, limit: int = 50, offset: int = 0) -> List[GuardianPlanRecord]:
        async def _list(session: AsyncSession):
            stmt = (
                select(GuardianPlanRecord)
                .order_by(desc(GuardianPlanRecord.created_at))
                .limit(limit)
                .offset(offset)
            )
            res = await session.execute(stmt)
            return list(res.scalars().all())

        if self._session:
            return await _list(self._session)
        async with AsyncSessionLocal() as session:
            return await _list(session)

    # -------------------------------------------------------------------------
    # Approval Persistence & Hash Binding
    # -------------------------------------------------------------------------
    async def record_approval(
        self,
        plan_id: str,
        config_hash: str,
        decision: str = "approved",
        approved_by: str = "user",
        notes: Optional[str] = None,
    ) -> GuardianApprovalRecord:
        approval = GuardianApprovalRecord(
            id=generate_uuid(),
            plan_id=plan_id,
            config_hash=config_hash,
            approved_by=approved_by,
            decision=decision,
            notes=notes,
            is_valid=(decision == "approved"),
            created_at=utc_now(),
        )

        async def _record(session: AsyncSession):
            # Check if plan config hash matches
            plan_res = await session.execute(
                select(GuardianPlanRecord).where(GuardianPlanRecord.id == plan_id)
            )
            plan = plan_res.scalar_one_or_none()
            if not plan:
                raise ValueError(f"Plan '{plan_id}' does not exist.")
            if plan.config_hash != config_hash:
                approval.is_valid = False
                approval.notes = (approval.notes or "") + f" [HASH MISMATCH: plan={plan.config_hash[:8]}, approved={config_hash[:8]}]"
            else:
                if decision == "approved":
                    plan.status = "approved"
                    plan.approval_status = "approved"
                else:
                    plan.status = "rejected"
                    plan.approval_status = "rejected"
                plan.updated_at = utc_now()

            session.add(approval)
            await session.commit()
            await session.refresh(approval)
            return approval

        if self._session:
            return await _record(self._session)
        async with AsyncSessionLocal() as session:
            return await _record(session)

    async def get_active_approval(self, plan_id: str) -> Optional[GuardianApprovalRecord]:
        async def _query(session: AsyncSession):
            stmt = (
                select(GuardianApprovalRecord)
                .where(GuardianApprovalRecord.plan_id == plan_id)
                .where(GuardianApprovalRecord.is_valid.is_(True))
                .where(GuardianApprovalRecord.decision == "approved")
                .order_by(desc(GuardianApprovalRecord.created_at))
            )
            res = await session.execute(stmt)
            return res.scalars().first()

        if self._session:
            return await _query(self._session)
        async with AsyncSessionLocal() as session:
            return await _query(session)

    # -------------------------------------------------------------------------
    # Execution Tracking & Step Timelines
    # -------------------------------------------------------------------------
    async def create_execution(
        self,
        plan_id: str,
        total_steps: int,
        execution_id: Optional[str] = None,
    ) -> GuardianExecutionRecord:
        execution = GuardianExecutionRecord(
            id=execution_id or generate_uuid(),
            plan_id=plan_id,
            status="queued",
            current_step=0,
            total_steps=total_steps,
            logs=[],
            results_json={},
            error_message=None,
            started_at=utc_now(),
        )

        async def _create(session: AsyncSession):
            session.add(execution)
            await session.commit()
            await session.refresh(execution)
            return execution

        if self._session:
            return await _create(self._session)
        async with AsyncSessionLocal() as session:
            return await _create(session)

    async def get_execution(self, execution_id: str) -> Optional[GuardianExecutionRecord]:
        async def _get(session: AsyncSession):
            stmt = (
                select(GuardianExecutionRecord)
                .where(GuardianExecutionRecord.id == execution_id)
                .options(
                    selectinload(GuardianExecutionRecord.plan),
                    selectinload(GuardianExecutionRecord.evidence),
                )
            )
            res = await session.execute(stmt)
            return res.scalar_one_or_none()

        if self._session:
            return await _get(self._session)
        async with AsyncSessionLocal() as session:
            return await _get(session)

    async def update_execution_step(
        self,
        execution_id: str,
        step_index: int,
        log_entry: Dict[str, Any],
        partial_results: Optional[Dict[str, Any]] = None,
    ) -> Optional[GuardianExecutionRecord]:
        async def _update(session: AsyncSession):
            stmt = select(GuardianExecutionRecord).where(GuardianExecutionRecord.id == execution_id)
            res = await session.execute(stmt)
            rec = res.scalar_one_or_none()
            if not rec:
                return None
            rec.status = "running"
            rec.current_step = step_index
            current_logs = list(rec.logs) if rec.logs else []
            current_logs.append(log_entry)
            rec.logs = current_logs

            if partial_results:
                curr_res = dict(rec.results_json) if rec.results_json else {}
                curr_res.update(partial_results)
                rec.results_json = curr_res

            await session.commit()
            await session.refresh(rec)
            return rec

        if self._session:
            return await _update(self._session)
        async with AsyncSessionLocal() as session:
            return await _update(session)

    async def finish_execution(
        self,
        execution_id: str,
        status: str,
        final_results: Dict[str, Any],
        error_message: Optional[str] = None,
    ) -> Optional[GuardianExecutionRecord]:
        async def _finish(session: AsyncSession):
            stmt = select(GuardianExecutionRecord).where(GuardianExecutionRecord.id == execution_id)
            res = await session.execute(stmt)
            rec = res.scalar_one_or_none()
            if not rec:
                return None
            rec.status = status
            rec.completed_at = utc_now()
            rec.error_message = error_message

            curr_res = dict(rec.results_json) if rec.results_json else {}
            curr_res.update(final_results)
            rec.results_json = curr_res

            # Update parent plan status as well
            plan_res = await session.execute(
                select(GuardianPlanRecord).where(GuardianPlanRecord.id == rec.plan_id)
            )
            plan = plan_res.scalar_one_or_none()
            if plan:
                plan.status = "completed" if status == "completed" else "failed"
                plan.updated_at = utc_now()

            await session.commit()
            await session.refresh(rec)
            return rec

        if self._session:
            return await _finish(self._session)
        async with AsyncSessionLocal() as session:
            return await _finish(session)

    # -------------------------------------------------------------------------
    # Evidence Persistence
    # -------------------------------------------------------------------------
    async def store_evidence(
        self,
        plan_id: str,
        execution_id: Optional[str],
        observations: List[str],
        hypotheses: List[str],
        recommendations: List[str],
        uncertainties: List[str],
        cited_experiment_ids: List[str],
    ) -> GuardianEvidenceRecord:
        record = GuardianEvidenceRecord(
            id=generate_uuid(),
            plan_id=plan_id,
            execution_id=execution_id,
            cited_experiment_ids=cited_experiment_ids,
            empirical_observations=observations,
            hypotheses_tested=hypotheses,
            grounded_recommendations=recommendations,
            uncertainties_and_caveats=uncertainties,
            created_at=utc_now(),
        )

        async def _store(session: AsyncSession):
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

        if self._session:
            return await _store(self._session)
        async with AsyncSessionLocal() as session:
            return await _store(session)

    async def get_evidence_by_plan(self, plan_id: str) -> Optional[GuardianEvidenceRecord]:
        async def _get(session: AsyncSession):
            stmt = (
                select(GuardianEvidenceRecord)
                .where(GuardianEvidenceRecord.plan_id == plan_id)
                .order_by(desc(GuardianEvidenceRecord.created_at))
            )
            res = await session.execute(stmt)
            return res.scalars().first()

        if self._session:
            return await _get(self._session)
        async with AsyncSessionLocal() as session:
            return await _get(session)

    # -------------------------------------------------------------------------
    # Cross-Experiment Retrieval & Comparison
    # -------------------------------------------------------------------------
    async def list_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists historical experiments from both ProtectionJob and GuardianExecutionRecord."""
        async def _list(session: AsyncSession):
            # Query ProtectionJob
            stmt = (
                select(ProtectionJob)
                .order_by(desc(ProtectionJob.created_at))
                .limit(limit)
                .options(
                    selectinload(ProtectionJob.config),
                    selectinload(ProtectionJob.evaluations),
                    selectinload(ProtectionJob.robustness_results),
                )
            )
            res = await session.execute(stmt)
            jobs = res.scalars().all()

            results = []
            for j in jobs:
                eval_map = {}
                for ev in j.evaluations:
                    eval_map[ev.model_name] = {
                        "status": ev.status,
                        "disrupted": ev.classification_disrupted,
                        "confidence_drop": ev.confidence_drop,
                        "cosine_dist": ev.embedding_cosine_distance,
                    }
                results.append({
                    "experiment_id": j.id,
                    "type": "protection_job",
                    "status": j.status,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "config": {
                        "epsilon": j.config.epsilon if j.config else None,
                        "iterations": j.config.iterations if j.config else None,
                        "surrogate": j.config.surrogate_model if j.config else None,
                    } if j.config else {},
                    "scores": {
                        "overall": j.overall_protection_score,
                        "disruption": j.ai_disruption_score,
                        "transferability": j.transferability_score,
                        "robustness": j.robustness_score,
                        "perceptual_quality": j.perceptual_quality_score,
                    },
                    "evaluations": eval_map,
                    "robustness_count": len(j.robustness_results),
                })
            return results

        if self._session:
            return await _list(self._session)
        async with AsyncSessionLocal() as session:
            return await _list(session)

    async def get_experiment_by_id(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        async def _get(session: AsyncSession):
            stmt = (
                select(ProtectionJob)
                .where(ProtectionJob.id == experiment_id)
                .options(
                    selectinload(ProtectionJob.config),
                    selectinload(ProtectionJob.evaluations),
                    selectinload(ProtectionJob.robustness_results),
                )
            )
            res = await session.execute(stmt)
            job = res.scalar_one_or_none()
            if not job:
                # Check guardian execution
                g_stmt = (
                    select(GuardianExecutionRecord)
                    .where(GuardianExecutionRecord.id == experiment_id)
                    .options(selectinload(GuardianExecutionRecord.plan))
                )
                g_res = await session.execute(g_stmt)
                g_exec = g_res.scalar_one_or_none()
                if not g_exec:
                    return None
                return {
                    "experiment_id": g_exec.id,
                    "type": "guardian_execution",
                    "plan_id": g_exec.plan_id,
                    "plan_title": g_exec.plan.title if g_exec.plan else None,
                    "status": g_exec.status,
                    "started_at": g_exec.started_at.isoformat() if g_exec.started_at else None,
                    "completed_at": g_exec.completed_at.isoformat() if g_exec.completed_at else None,
                    "results": g_exec.results_json,
                }

            return {
                "experiment_id": job.id,
                "type": "protection_job",
                "status": job.status,
                "image_id": job.image_id,
                "protected_image_id": job.protected_image_id,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "config": {
                    "epsilon": job.config.epsilon if job.config else None,
                    "iterations": job.config.iterations if job.config else None,
                    "step_size": job.config.step_size if job.config else None,
                    "surrogate_model": job.config.surrogate_model if job.config else None,
                    "evaluation_models": job.config.evaluation_models if job.config else [],
                } if job.config else {},
                "scores": {
                    "overall": job.overall_protection_score,
                    "disruption": job.ai_disruption_score,
                    "transferability": job.transferability_score,
                    "robustness": job.robustness_score,
                    "perceptual_quality": job.perceptual_quality_score,
                },
                "evaluations": [
                    {
                        "model_name": ev.model_name,
                        "status": ev.status,
                        "is_surrogate": ev.is_surrogate,
                        "disrupted": ev.classification_disrupted,
                        "confidence_drop": ev.confidence_drop,
                        "cosine_dist": ev.embedding_cosine_distance,
                        "orig_top1": ev.original_top1_class,
                        "adv_top1": ev.protected_top1_class,
                    }
                    for ev in job.evaluations
                ],
                "robustness": [
                    {
                        "transformation": r.transformation_name,
                        "retained_rate": r.retained_protection_rate,
                        "transformed_class": r.top1_class_after_transform,
                        "transformed_conf": r.top1_confidence_after_transform,
                    }
                    for r in job.robustness_results
                ],
            }

        if self._session:
            return await _get(self._session)
        async with AsyncSessionLocal() as session:
            return await _get(session)

    async def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compares two or more experiments side-by-side with delta analysis."""
        records = []
        for exp_id in experiment_ids:
            rec = await self.get_experiment_by_id(exp_id)
            if rec:
                records.append(rec)

        if not records:
            return {"error": "None of the specified experiments were found."}

        # Align metrics
        comparison: Dict[str, Any] = {
            "experiment_count": len(records),
            "experiments": records,
            "metric_alignment": {},
            "deltas": {},
        }

        if len(records) >= 2:
            base = records[0]
            comp = records[1]
            base_scores = base.get("scores", {}) or {}
            comp_scores = comp.get("scores", {}) or {}

            deltas = {}
            for metric in ["overall", "disruption", "transferability", "robustness", "perceptual_quality"]:
                v_base = base_scores.get(metric)
                v_comp = comp_scores.get(metric)
                if v_base is not None and v_comp is not None:
                    deltas[metric] = {
                        "baseline": v_base,
                        "comparison": v_comp,
                        "delta": round(v_comp - v_base, 4),
                        "direction": "improved" if (v_comp - v_base) > 0 else "degraded" if (v_comp - v_base) < 0 else "neutral",
                    }
            comparison["deltas"] = deltas

        return comparison
