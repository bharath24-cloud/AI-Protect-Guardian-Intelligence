from fastapi import APIRouter
from backend.app.api.health import router as health_router
from backend.app.api.upload import router as upload_router
from backend.app.api.jobs import router as jobs_router
from backend.app.api.guardian import router as guardian_router

api_router = APIRouter()

api_router.include_router(health_router, tags=["System Health"])
api_router.include_router(upload_router, tags=["Image Upload & Validation"])
api_router.include_router(jobs_router, tags=["Protection Pipeline & Jobs"])
api_router.include_router(guardian_router)
