from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI-Protect"
    PROJECT_SUBTITLE: str = "Adaptive adversarial protection for digital images"
    API_V1_STR: str = "/api"
    
    # Base directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    
    # Storage settings
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    
    # Privacy & Limits
    MAX_UPLOAD_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
    ALLOWED_MIME_TYPES: list[str] = ["image/jpeg", "image/png", "image/webp"]
    ALLOWED_EXTENSIONS: list[str] = [".jpg", ".jpeg", ".png", ".webp"]
    AUTO_CLEANUP_HOURS: int = 24  # Image retention period before auto-cleanup
    
    # Database
    DATABASE_URL: str = f"sqlite+aiosqlite:///{DATA_DIR / 'aiprotect.db'}"
    SYNC_DATABASE_URL: str = f"sqlite:///{DATA_DIR / 'aiprotect.db'}"

    # Default Attack Hyperparameters
    DEFAULT_EPSILON: float = 8.0 / 255.0
    DEFAULT_ITERATIONS: int = 20
    DEFAULT_STEP_SIZE: float = 2.0 / 255.0
    DEFAULT_SURROGATE: str = "resnet50"
    DEFAULT_EVAL_MODELS: list[str] = ["resnet50", "vit_b_16", "convnext_tiny", "clip_vit_b32"]

    model_config = {
        "env_prefix": "AIPROTECT_",
        "case_sensitive": False
    }

settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
