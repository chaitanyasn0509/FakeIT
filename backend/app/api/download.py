"""Download routes for generated assets."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.core.deps import get_current_user
from backend.app.db.models import Job, User
from backend.app.db.session import get_db
from backend.app.services.storage import create_storage

router = APIRouter(prefix="/download", tags=["download"])


@router.get("/{job_id}")
def download_asset(
    job_id: str,
    asset: str = Query("output", pattern="^(input|mask|output)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Download an input, mask, or reconstructed GeoTIFF for a job."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this job")
    uri = {"input": job.input_uri, "mask": job.mask_uri, "output": job.output_uri}[asset]
    if not uri:
        raise HTTPException(status_code=404, detail=f"{asset} asset is not available")
    storage = create_storage(settings)
    path = storage.resolve(uri)
    return FileResponse(path, media_type="image/tiff", filename=path.name)
