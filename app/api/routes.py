"""
app/api/routes.py
=================
API route definitions for the Lead Processing Pipeline.

Endpoints:
  - POST /upload               - Upload a CSV of leads for async processing
  - GET  /jobs                 - List every upload
  - GET  /jobs/{job_id}        - Get one upload's ingestion progress
  - GET  /leads                - Browse processed leads (filter + paginate)
  - GET  /failed-leads/{job_id}- Dead-letter queue entries for an upload
"""

import io
import logging
from typing import Optional
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Query
from pydantic import BaseModel

from app.config import get_settings
from app.utils import utcnow_iso
from app.db.repository import LeadRepository, JobRepository, FailedLeadRepository
from app.services.processing_service import processing_service

logger = logging.getLogger(__name__)

router = APIRouter()

lead_repo = LeadRepository()
job_repo = JobRepository()
failed_repo = FailedLeadRepository()

REQUIRED_COLUMNS = {'name', 'email', 'message'}


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class JobSummary(BaseModel):
    job_id: str
    status: str
    filename: Optional[str] = None
    total_leads: int
    processed: int
    failed: int
    duplicates: int
    invalid: int
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/upload", tags=["Processing"])
async def upload_leads(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    """
    Upload a CSV of leads (max 50,000 rows) for asynchronous processing.

    Returns immediately with a job_id; validation, deduplication and
    storage happen in the background. Poll GET /jobs/{job_id} for progress.
    Required columns: name, email, message. Optional: city, company, phone.
    """
    settings = get_settings()

    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required column(s): {', '.join(sorted(missing))}"
        )

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV has no rows")

    if len(df) > settings.MAX_LEADS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.MAX_LEADS_PER_REQUEST} leads per upload"
        )

    rows = df.where(pd.notnull(df), None).to_dict('records')

    job_id = str(uuid4())
    await job_repo.create({
        'job_id': job_id,
        'status': 'ingesting',
        'filename': file.filename,
        'total_leads': len(rows),
        'created_at': utcnow_iso(),
    })
    logger.info(f"✓ Job created: {job_id} ({len(rows)} rows from {file.filename})")

    background_tasks.add_task(processing_service.process_upload, job_id, rows)

    return {
        "job_id": job_id,
        "status": "ingesting",
        "total_leads": len(rows),
        "message": f"Leads queued for processing. Check status with GET /jobs/{job_id}"
    }


@router.get("/jobs", tags=["Status"])
async def list_jobs(
    page_size: int = Query(50, ge=1, le=200),
    cursor: Optional[str] = Query(None, description="Opaque pagination cursor from a previous response"),
) -> dict:
    """List every upload, most recent first."""
    jobs, next_cursor = await job_repo.list_all(limit=page_size, cursor=cursor)
    return {
        "jobs": [{k: v for k, v in job.items() if k != 'list_key'} for job in jobs],
        "next_cursor": next_cursor,
    }


@router.get("/jobs/{job_id}", tags=["Status"], response_model=JobSummary)
async def get_job_status(job_id: str) -> JobSummary:
    """Get one upload's ingestion progress (processed / failed / duplicates / invalid)."""
    job = await job_repo.get_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job.pop('list_key', None)
    return JobSummary(**job)


@router.get("/leads", tags=["Results"])
async def list_leads(
    city: Optional[str] = Query(None, description="Filter by city"),
    intent: Optional[str] = Query(None, description="Filter by intent (hot/warm/cold)"),
    date_from: Optional[str] = Query(None, description="ISO date/datetime lower bound (created_at)"),
    date_to: Optional[str] = Query(None, description="ISO date/datetime upper bound (created_at)"),
    page_size: int = Query(100, ge=1, le=1000),
    cursor: Optional[str] = Query(None, description="Opaque pagination cursor from a previous response"),
) -> dict:
    """
    Browse processed leads, filterable by city, intent and a created_at
    date range. Pagination is cursor-based (DynamoDB doesn't support
    numbered pages over large result sets) - pass back `next_cursor`
    to fetch the following page.
    """
    leads, next_cursor = await lead_repo.list_leads(
        city=city, intent=intent, date_from=date_from, date_to=date_to,
        limit=page_size, cursor=cursor,
    )
    return {
        "leads": leads,
        "next_cursor": next_cursor,
        "filters_applied": {
            k: v for k, v in {
                "city": city, "intent": intent, "date_from": date_from, "date_to": date_to
            }.items() if v is not None
        },
    }


@router.get("/failed-leads/{job_id}", tags=["Debug"])
async def get_failed_leads(job_id: str) -> dict:
    """Dead-letter queue entries for an upload - useful for debugging and retry analysis."""
    failed_leads = await failed_repo.get_by_job_id(job_id)
    return {
        "job_id": job_id,
        "failed_count": len(failed_leads),
        "failed_leads": failed_leads,
    }
