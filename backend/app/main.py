from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.router import api_router
from backend.app.config import settings
from backend.app.database import init_db
import backend.app.db_models.models  # noqa: F401
import backend.app.memory.models     # noqa: F401

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB schemas
    await init_db()
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "AI-Protect: Adaptive Adversarial Cloaking Engine for Digital Content Protection.\n\n"
        "**Scientific Disclaimer**: Adversarial cloaking provides empirical, non-cryptographic protection "
        "against targeted vision and embedding models. Effectiveness depends on model architecture, "
        "attack hyper-parameters, and subsequent image transformations."
    ),
    version="0.1.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount static storage directories
app.mount("/api/files/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/api/files/processed", StaticFiles(directory=settings.PROCESSED_DIR), name="processed")

@app.get("/")
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "subtitle": settings.PROJECT_SUBTITLE,
        "docs_url": "/docs",
        "api_prefix": settings.API_V1_STR,
        "status": "operational"
    }
