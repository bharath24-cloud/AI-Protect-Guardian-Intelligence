import io
import pytest
from httpx import ASGITransport, AsyncClient
from backend.app.main import app


@pytest.mark.asyncio
async def test_guardian_api_models():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/guardian/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "resnet50" in data["models"]


@pytest.mark.asyncio
async def test_guardian_api_tools():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/guardian/tools")
        assert resp.status_code == 200
        tools = resp.json()
        assert len(tools) >= 10
        names = [t["name"] for t in tools]
        assert "inspect_pipeline" in names
        assert "run_adversarial_cloaking" in names


@pytest.mark.asyncio
async def test_guardian_api_chat():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/guardian/chat",
            json={"message": "What vision models are ready in the pipeline?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "reply" in data
        assert "resnet50" in data["reply"].lower() or "vision model" in data["reply"].lower()


@pytest.mark.asyncio
async def test_guardian_api_plan_approval_execution_lifecycle():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Create plan
        plan_resp = await client.post(
            "/api/guardian/plan",
            json={"goal": "Evaluate cloaking configuration on artwork", "constraints": {"epsilon": 0.03137}},
        )
        assert plan_resp.status_code == 200
        plan_data = plan_resp.json()
        plan_id = plan_data["plan_id"]
        config_hash = plan_data["config_hash"]
        assert plan_data["objective_type"] == "evaluate_cloaking"
        assert plan_data["requires_approval"] is True
        assert len(config_hash) == 64

        # 2. Get plan by ID
        get_plan_resp = await client.get(f"/api/guardian/plans/{plan_id}")
        assert get_plan_resp.status_code == 200
        assert get_plan_resp.json()["plan_id"] == plan_id

        # 3. Reject approval if config hash is tampered
        bad_appr = await client.post(
            "/api/guardian/approve",
            json={
                "plan_id": plan_id,
                "config_hash": "0" * 64,  # mismatched!
                "decision": "approved",
            },
        )
        assert bad_appr.status_code == 400
        assert "tampering detected" in bad_appr.json()["detail"].lower()

        # 4. Valid approval
        good_appr = await client.post(
            "/api/guardian/approve",
            json={
                "plan_id": plan_id,
                "config_hash": config_hash,
                "decision": "approved",
                "approved_by": "lead_ml_engineer",
            },
        )
        assert good_appr.status_code == 200
        assert good_appr.json()["status"] == "approved"

        # 5. Execute plan
        exec_resp = await client.post(
            "/api/guardian/execute",
            json={"plan_id": plan_id, "config_hash": config_hash},
        )
        assert exec_resp.status_code == 200
        exec_data = exec_resp.json()
        assert exec_data["success"] is True
        assert "execution_id" in exec_data

        # 6. Poll execution status
        exec_id = exec_data["execution_id"]
        poll_resp = await client.get(f"/api/guardian/jobs/{exec_id}")
        assert poll_resp.status_code == 200
        poll_data = poll_resp.json()
        assert poll_data["execution_id"] == exec_id
        assert poll_data["status"] in ["queued", "running", "completed"]


@pytest.mark.asyncio
async def test_guardian_api_experiments_listing():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/guardian/experiments")
        assert resp.status_code == 200
        exps = resp.json()
        assert isinstance(exps, list)
