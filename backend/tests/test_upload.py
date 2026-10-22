import io
import pytest
from PIL import Image
from httpx import ASGITransport, AsyncClient

from backend.app.main import app

def create_test_image(size=(128, 128), color=(255, 0, 0), format="PNG") -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["project"] == "AI-Protect"
        assert "available_surrogates" in data

@pytest.mark.asyncio
async def test_upload_valid_image():
    transport = ASGITransport(app=app)
    img_bytes = create_test_image(size=(100, 100), color=(10, 200, 50), format="PNG")
    
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test_art.png", img_bytes, "image/png")}
        resp = await client.post("/api/upload", files=files)
        assert resp.status_code == 201
        data = resp.json()
        assert "image_id" in data
        assert data["width"] == 100
        assert data["height"] == 100
        assert data["mime_type"] == "image/png"
        assert len(data["sha256_hash"]) == 64
        assert "/api/files/uploads/" in data["preview_url"]

@pytest.mark.asyncio
async def test_upload_invalid_extension():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("malicious.txt", b"not an image", "text/plain")}
        resp = await client.post("/api/upload", files=files)
        assert resp.status_code == 415

@pytest.mark.asyncio
async def test_upload_corrupted_image():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("corrupt.png", b"\x89PNG\r\n\x1a\ncorruptedcontent12345", "image/png")}
        resp = await client.post("/api/upload", files=files)
        assert resp.status_code == 400

@pytest.mark.asyncio
async def test_create_protection_job():
    transport = ASGITransport(app=app)
    img_bytes = create_test_image(size=(64, 64), color=(50, 50, 200), format="JPEG")
    
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # First upload image
        files = {"file": ("source.jpg", img_bytes, "image/jpeg")}
        upload_resp = await client.post("/api/upload", files=files)
        assert upload_resp.status_code == 201
        image_id = upload_resp.json()["image_id"]

        # Register protection job
        config_payload = {
            "image_id": image_id,
            "protection_mode": "artwork",
            "surrogate_model": "resnet50",
            "evaluation_models": ["resnet50", "vit_b_16", "convnext_tiny", "clip_vit_b32"],
            "epsilon": 0.03137,
            "iterations": 20,
            "step_size": 0.00784,
            "loss_objective": "classification_disruption",
            "random_init": True,
            "random_seed": 42
        }
        job_resp = await client.post("/api/protect", json=config_payload)
        assert job_resp.status_code == 202
        job_data = job_resp.json()
        assert job_data["status"] == "pending"
        assert job_data["image_id"] == image_id

        # Query job status
        job_id = job_data["job_id"]
        status_resp = await client.get(f"/api/jobs/{job_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["job_id"] == job_id
