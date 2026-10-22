import pytest
import uuid
from backend.app.memory.repository import GuardianMemoryRepository
from backend.app.memory.models import GuardianPlanRecord


@pytest.mark.asyncio
async def test_experiment_memory_plan_crud():
    repo = GuardianMemoryRepository()
    plan_id = str(uuid.uuid4())
    config_hash = "f" * 64

    # 1. Create plan
    plan = await repo.create_plan(
        plan_id=plan_id,
        title="Test Memory Plan",
        goal="Test persistent memory storage for Guardian",
        objective_type="evaluate_cloaking",
        config_hash=config_hash,
        plan_json={"steps": [{"step_number": 1, "tool_name": "inspect_pipeline"}]},
        requires_approval=True,
    )
    assert plan.id == plan_id
    assert plan.status == "proposed"
    assert plan.approval_status == "pending"

    # 2. Retrieve plan
    fetched = await repo.get_plan(plan_id)
    assert fetched is not None
    assert fetched.title == "Test Memory Plan"
    assert fetched.config_hash == config_hash

    # 3. Update plan status
    updated = await repo.update_plan_status(plan_id, status="awaiting_approval")
    assert updated.status == "awaiting_approval"


@pytest.mark.asyncio
async def test_experiment_memory_approval_hash_binding():
    repo = GuardianMemoryRepository()
    plan_id = str(uuid.uuid4())
    config_hash = "e" * 64

    await repo.create_plan(
        plan_id=plan_id,
        title="Approval Test Plan",
        goal="Verify hash binding on authorization",
        objective_type="robustness_benchmark",
        config_hash=config_hash,
        plan_json={"steps": []},
        requires_approval=True,
    )

    # Valid approval
    appr_valid = await repo.record_approval(
        plan_id=plan_id,
        config_hash=config_hash,
        decision="approved",
        approved_by="test_user",
        notes="Authorized for execution",
    )
    assert appr_valid.is_valid is True

    active = await repo.get_active_approval(plan_id)
    assert active is not None
    assert active.config_hash == config_hash

    # Mismatched approval (tampered hash)
    plan_id_2 = str(uuid.uuid4())
    await repo.create_plan(
        plan_id=plan_id_2,
        title="Tampered Plan",
        goal="Tamper test",
        objective_type="robustness_benchmark",
        config_hash="1" * 64,
        plan_json={"steps": []},
        requires_approval=True,
    )
    appr_mismatch = await repo.record_approval(
        plan_id=plan_id_2,
        config_hash="2" * 64,  # mismatched
        decision="approved",
        approved_by="attacker",
    )
    assert appr_mismatch.is_valid is False
    assert "HASH MISMATCH" in appr_mismatch.notes


@pytest.mark.asyncio
async def test_experiment_memory_execution_lifecycle():
    repo = GuardianMemoryRepository()
    plan_id = str(uuid.uuid4())
    await repo.create_plan(
        plan_id=plan_id,
        title="Execution Lifecycle Plan",
        goal="Track timeline steps",
        objective_type="evaluate_cloaking",
        config_hash="3" * 64,
        plan_json={"steps": []},
        requires_approval=False,
    )

    # Create execution
    exec_rec = await repo.create_execution(plan_id=plan_id, total_steps=3)
    assert exec_rec.status == "queued"
    assert exec_rec.current_step == 0

    # Step 1 update
    step_log = {"step_number": 1, "tool_name": "validate_image_input", "status": "completed"}
    await repo.update_execution_step(exec_rec.id, step_index=1, log_entry=step_log, partial_results={"step1": True})

    fetched_exec = await repo.get_execution(exec_rec.id)
    assert fetched_exec.status == "running"
    assert fetched_exec.current_step == 1
    assert len(fetched_exec.logs) == 1

    # Finish execution
    final_res = {"quality": {"ssim": 0.895, "psnr": 34.2}}
    finished = await repo.finish_execution(exec_rec.id, status="completed", final_results=final_res)
    assert finished.status == "completed"
    assert finished.completed_at is not None
    assert "quality" in finished.results_json


@pytest.mark.asyncio
async def test_experiment_memory_evidence_storage():
    repo = GuardianMemoryRepository()
    plan_id = str(uuid.uuid4())
    exec_id = str(uuid.uuid4())

    await repo.create_plan(
        plan_id=plan_id,
        title="Evidence Plan",
        goal="Store and retrieve evidence",
        objective_type="cross_transfer",
        config_hash="4" * 64,
        plan_json={"steps": []},
        requires_approval=False,
    )

    evidence = await repo.store_evidence(
        plan_id=plan_id,
        execution_id=exec_id,
        observations=["Obs 1: High perceptual fidelity", "Obs 2: Disruption on ResNet50"],
        hypotheses=["Hypothesis: Transfer rate attenuates on ViT"],
        recommendations=["Recommendation: Run ensemble multi-surrogate"],
        uncertainties=["Sample size N=1 limitation"],
        cited_experiment_ids=[plan_id, exec_id],
    )
    assert evidence.id is not None
    assert len(evidence.empirical_observations) == 2

    # Fetch evidence
    fetched_evi = await repo.get_evidence_by_plan(plan_id)
    assert fetched_evi is not None
    assert fetched_evi.plan_id == plan_id
    assert "ensemble" in fetched_evi.grounded_recommendations[0].lower()
