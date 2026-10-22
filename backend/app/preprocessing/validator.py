import hashlib
import io
from pathlib import Path
from PIL import Image, ImageOps
from fastapi import HTTPException, UploadFile, status

from backend.app.config import settings

class ImageValidationResult:
    def __init__(
        self,
        image: Image.Image,
        raw_bytes: bytes,
        sha256_hash: str,
        width: int,
        height: int,
        mime_type: str,
        detected_format: str
    ):
        self.image = image
        self.raw_bytes = raw_bytes
        self.sha256_hash = sha256_hash
        self.width = width
        self.height = height
        self.mime_type = mime_type
        self.detected_format = detected_format

class ImageValidator:
    @staticmethod
    async def validate_upload(file: UploadFile) -> ImageValidationResult:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must have a valid filename."
            )

        # Check extension
        ext = Path(file.filename).suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file extension '{ext}'. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )

        # Read contents
        contents = await file.read()
        
        # Check size
        if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB."
            )
        
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )

        # Compute SHA-256
        sha256 = hashlib.sha256(contents).hexdigest()

        # PIL image decode
        try:
            pil_image = Image.open(io.BytesIO(contents))
            pil_image.verify()  # Verify structural integrity
            # Reopen because verify() consumes image state
            pil_image = Image.open(io.BytesIO(contents))
            # Handle EXIF orientation if present
            pil_image = ImageOps.exif_transpose(pil_image)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid or corrupted image file: {str(e)}"
            )

        # Detect format & mime
        detected_format = pil_image.format or ext.replace(".", "").upper()
        mime_map = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp"
        }
        mime_type = mime_map.get(detected_format.upper(), file.content_type or "application/octet-stream")

        # Convert to RGB if needed (strip alpha or palette)
        if pil_image.mode not in ("RGB", "RGBA"):
            pil_image = pil_image.convert("RGB")

        width, height = pil_image.size
        if width < 32 or height < 32:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image resolution {width}x{height} is too small. Minimum resolution is 32x32."
            )

        return ImageValidationResult(
            image=pil_image,
            raw_bytes=contents,
            sha256_hash=sha256,
            width=width,
            height=height,
            mime_type=mime_type,
            detected_format=detected_format
        )
