import os
import time
import uuid
from pathlib import Path
from PIL import Image

from backend.app.config import settings

class StorageManager:
    @staticmethod
    def save_upload_bytes(raw_bytes: bytes, original_filename: str) -> tuple[str, Path]:
        """Saves raw upload bytes with a unique UUID filename and returns (filename, absolute_path)"""
        ext = Path(original_filename).suffix.lower()
        if not ext:
            ext = ".png"
        unique_name = f"{uuid.uuid4()}{ext}"
        target_path = settings.UPLOAD_DIR / unique_name
        
        with open(target_path, "wb") as f:
            f.write(raw_bytes)
            
        return unique_name, target_path

    @staticmethod
    def save_pil_image(image: Image.Image, prefix: str = "protected", format: str = "PNG") -> tuple[str, Path]:
        """Saves a PIL image into the processed directory"""
        unique_name = f"{prefix}_{uuid.uuid4()}.{format.lower()}"
        target_path = settings.PROCESSED_DIR / unique_name
        image.save(target_path, format=format)
        return unique_name, target_path

    @staticmethod
    def cleanup_old_files(max_age_hours: int = settings.AUTO_CLEANUP_HOURS) -> int:
        """Removes files in upload and processed directories older than max_age_hours"""
        now = time.time()
        max_age_seconds = max_age_hours * 3600
        removed_count = 0

        for folder in [settings.UPLOAD_DIR, settings.PROCESSED_DIR]:
            if not folder.exists():
                continue
            for item in folder.iterdir():
                if item.is_file():
                    file_age = now - item.stat().st_mtime
                    if file_age > max_age_seconds:
                        try:
                            item.unlink()
                            removed_count += 1
                        except OSError:
                            pass
        return removed_count

    @staticmethod
    def get_upload_path(filename: str) -> Path | None:
        path = settings.UPLOAD_DIR / filename
        return path if path.exists() else None

    @staticmethod
    def get_processed_path(filename: str) -> Path | None:
        path = settings.PROCESSED_DIR / filename
        return path if path.exists() else None

storage_manager = StorageManager()
