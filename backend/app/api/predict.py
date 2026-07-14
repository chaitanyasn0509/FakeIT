"""Cloud detection and reconstruction route."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.config import Settings, get_settings
from backend.app.core.deps import get_current_user
from backend.app.db.models import Job, User
from backend.app.db.session import get_db
from backend.app.schemas import PredictRequest, PredictResponse
from backend.app.services.inference import CloudRemovalService, metrics_to_json
from backend.app.services.storage import create_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> PredictResponse:
    """Run cloud detection and reconstruction for an uploaded job."""
    job = db.get(Job, payload.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this job")
    storage = create_storage(settings)
    input_path = storage.resolve(job.input_uri)
    work_dir = Path(settings.local_storage_root) / "jobs" / job.id
    service = CloudRemovalService(settings.config_path)
    try:
        job.status = "processing"
        db.commit()
        result = service.predict(input_path, work_dir)
        job.mask_uri = storage.save_file(result["mask_path"], f"jobs/{job.id}/masks")
        job.output_uri = storage.save_file(result["output_path"], f"jobs/{job.id}/outputs")
        job.metrics_json = metrics_to_json(result["metrics"])
        job.confidence_score = result["confidence_score"]
        job.status = "completed"
    except Exception:
        logger.exception("Inference failed for job %s", job.id)
        job.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail="Inference failed. Please try again later.")
    db.commit()
    return PredictResponse(
        job_id=job.id,
        status=job.status,
        cloud_mask_url=f"/download/{job.id}?asset=mask",
        download_url=f"/download/{job.id}?asset=output",
        metrics=json.loads(job.metrics_json),
        confidence_score=job.confidence_score,
    )
