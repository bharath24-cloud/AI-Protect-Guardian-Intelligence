import torch
from fastapi import APIRouter
from backend.app.schemas.schemas import HealthCheckResponse
from backend.app.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        project=settings.PROJECT_NAME,
        version="0.1.0",
        database_connected=True,
        torch_available=torch.__version__ is not None,
        cuda_available=torch.cuda.is_available(),
        available_surrogates=["resnet50", "vit_b_16", "convnext_tiny", "clip_vit_b32"],
        available_eval_models=["resnet50", "vit_b_16", "convnext_tiny", "clip_vit_b32"]
    )
