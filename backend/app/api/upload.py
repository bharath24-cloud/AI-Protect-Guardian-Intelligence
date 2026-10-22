from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database import get_db
from backend.app.db_models.models import ImageRecord
from backend.app.preprocessing.validator import ImageValidator
from backend.app.schemas.schemas import ImageUploadResponse
from backend.app.storage.manager import storage_manager

router = APIRouter()

@router.post("/upload", response_model=ImageUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Validate uploaded image, generate SHA256 digest, store securely in temp uploads directory,
    and register metadata in SQLite database.
    """
    # Validate image
    validation_res = await ImageValidator.validate_upload(file)
    
    # Save file
    saved_filename, saved_path = storage_manager.save_upload_bytes(
        validation_res.raw_bytes,
        file.filename or "uploaded_image.png"
    )

    # Persist in DB
    image_record = ImageRecord(
        filename=saved_filename,
        original_filename=file.filename or "image.png",
        file_path=str(saved_path),
        file_size_bytes=len(validation_res.raw_bytes),
        mime_type=validation_res.mime_type,
        width=validation_res.width,
        height=validation_res.height,
        sha256_hash=validation_res.sha256_hash,
        is_protected_output=False
    )
    db.add(image_record)
    await db.commit()
    await db.refresh(image_record)

    preview_url = f"/api/files/uploads/{saved_filename}"

    return ImageUploadResponse(
        image_id=image_record.id,
        filename=image_record.filename,
        original_filename=image_record.original_filename,
        width=image_record.width,
        height=image_record.height,
        file_size_bytes=image_record.file_size_bytes,
        mime_type=image_record.mime_type,
        sha256_hash=image_record.sha256_hash,
        preview_url=preview_url,
        created_at=image_record.created_at
    )
