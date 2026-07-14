"""Inference history routes."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from backend.app.core.deps import get_current_user
from backend.app.db.models import Job, User
from backend.app.db.session import get_db
from backend.app.schemas import JobResponse, PaginatedJobsResponse

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=PaginatedJobsResponse)
def list_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
) -> PaginatedJobsResponse:
    """Return paginated inference jobs for the authenticated user."""
    base_query = select(Job).where(Job.user_id == current_user.id)
    total = db.scalar(select(func.count()).select_from(base_query.subquery())) or 0
    jobs = db.scalars(
        base_query.order_by(desc(Job.created_at)).offset(offset).limit(limit)
    ).all()
    items = [
        JobResponse(
            id=job.id,
            status=job.status,
            input_uri=job.input_uri,
            mask_uri=job.mask_uri,
            output_uri=job.output_uri,
            metrics=json.loads(job.metrics_json or "{}"),
            confidence_score=job.confidence_score,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )
        for job in jobs
    ]
    return PaginatedJobsResponse(items=items, total=total, offset=offset, limit=limit)
