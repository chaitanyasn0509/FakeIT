"""Image upload route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.core.deps import get_current_user
from backend.app.db.models import Job, User
from backend.app.db.session import get_db
from backend.app.schemas import UploadResponse
from backend.app.services.storage import create_storage

router = APIRouter(prefix="/upload", tags=["upload"])

_MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
_ALLOWED_EXTENSIONS = {".tif", ".tiff", ".geotiff"}


@router.post("", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    """Upload a satellite image and create an inference job."""
    filename = file.filename or "image.tif"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Accepted: {sorted(_ALLOWED_EXTENSIONS)}",
        )
    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )
    await file.seek(0)
    storage = create_storage(settings)
    uri = storage.save_upload(file.file, filename)
    job = Job(input_uri=uri, status="uploaded", user_id=current_user.id)
    db.add(job)
    db.commit()
    db.refresh(job)
    return UploadResponse(job_id=job.id, status=job.status)
