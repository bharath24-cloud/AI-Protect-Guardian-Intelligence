import asyncio
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image
from sqlalchemy import select

from backend.app.database import AsyncSessionLocal, init_db
from backend.app.db_models.models import ImageRecord, ProtectionConfig, ProtectionJob, EvaluationResult, RobustnessResult
from backend.app.api.jobs import execute_protection_job

async def test_end_to_end_protection_job():
    print("=" * 80)
    print("      AI-PROTECT: END-TO-END PIPELINE & DATABASE INTEGRATION TEST")
    print("=" * 80)

    await init_db()

    # 1. Create a test image file
    uploads_dir = Path("backend/data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    test_img_path = uploads_dir / "e2e_test_artwork.png"

    img = Image.new("RGB", (256, 256), (150, 100, 200))
    img.save(test_img_path)

    async with AsyncSessionLocal() as session:
        # 2. Register Image Record
        img_rec = ImageRecord(
            filename="e2e_test_artwork.png",
            original_filename="e2e_test_artwork.png",
            file_path=str(test_img_path),
            file_size_bytes=os.path.getsize(test_img_path),
            mime_type="image/png",
            width=256,
            height=256,
            sha256_hash="e2e_dummy_hash_001",
            is_protected_output=False
        )
        session.add(img_rec)
        await session.flush()

        # 3. Register Protection Config
        cfg_rec = ProtectionConfig(
            protection_mode="artwork",
            surrogate_model="resnet50",
            evaluation_models=["resnet50", "convnext_tiny", "vit_b_16", "clip_vit_b32"],
            epsilon=8.0 / 255.0,
            iterations=10,
            step_size=2.0 / 255.0,
            loss_objective="classification_disruption",
            random_init=True,
            random_seed=42
        )
        session.add(cfg_rec)
        await session.flush()

        # 4. Create Protection Job
        job_rec = ProtectionJob(
            image_id=img_rec.id,
            config_id=cfg_rec.id,
            status="pending",
            progress=0.0
        )
        session.add(job_rec)
        await session.commit()
        job_id = job_rec.id

    print(f"\n[1/3] Created Job ID: {job_id} in SQLite database.")

    # 5. Execute Job Worker
    print(f"[2/3] Executing protection worker (PGD cloaking + 4-model evaluation + robustness)...")
    await execute_protection_job(job_id)

    # 6. Verify Results in Database
    print(f"[3/3] Inspecting database records for Job {job_id}...")
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(ProtectionJob).where(ProtectionJob.id == job_id))
        job = res.scalar_one()

        assert job.status == "completed", f"Job failed with status: {job.status}, error: {job.error_message}"
        print(f"      Job Status       : {job.status}")
        print(f"      Overall Score    : {job.overall_protection_score}%")
        print(f"      AI Disruption    : {job.ai_disruption_score}%")
        print(f"      Transferability  : {job.transferability_score}%")

        # Check Evaluations
        ev_res = await session.execute(select(EvaluationResult).where(EvaluationResult.job_id == job_id))
        evals = ev_res.scalars().all()
        print(f"\n      Evaluated Models ({len(evals)}):")
        for ev in evals:
            print(f"        - {ev.model_name:<15}: status={ev.status:<10} | type={ev.evaluation_type} | "
                  f"orig={ev.original_top1_class} -> prot={ev.protected_top1_class} | "
                  f"disrupted={ev.classification_disrupted}")
            assert ev.status == "evaluated", f"Model {ev.model_name} had non-evaluated status: {ev.status}"

        # Check Robustness
        rb_res = await session.execute(select(RobustnessResult).where(RobustnessResult.job_id == job_id))
        robust = rb_res.scalars().all()
        print(f"\n      Robustness Benchmarks ({len(robust)}):")
        for r in robust:
            print(f"        - {r.transformation_name:<30}: style={r.top1_class_after_transform} | retention={r.retained_protection_rate}%")

    print("\n" + "=" * 80)
    print("      END-TO-END PIPELINE & DATABASE TEST: ALL CHECKS PASSED!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_end_to_end_protection_job())
