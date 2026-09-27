"""
app/services/processing_service.py
==================================
Ingestion pipeline (Part 1/2/5): validates, deduplicates and stores
uploaded leads with bounded concurrency, retrying transient write
failures with backoff and dead-lettering permanent ones.

Classification is NOT done here - see app/workers/classification_worker.py.
"""

import asyncio
import logging
from typing import List, Dict, Any
from uuid import uuid4

from app.config import get_settings
from app.db.repository import LeadRepository, JobRepository, FailedLeadRepository
from app.services.lead_service import validate_lead, build_lead_item

logger = logging.getLogger(__name__)


class ProcessingService:

    def __init__(self):
        self.settings = get_settings()
        self.lead_repo = LeadRepository()
        self.job_repo = JobRepository()
        self.failed_repo = FailedLeadRepository()

    async def process_upload(self, job_id: str, rows: List[Dict[str, Any]]):
        """Validate, deduplicate and store every row from an uploaded CSV."""

        logger.info(f"Starting ingestion for job {job_id}: {len(rows)} rows")

        semaphore = asyncio.Semaphore(self.settings.CONCURRENT_WORKERS)
        lock = asyncio.Lock()
        totals = {'processed': 0, 'failed': 0, 'duplicates': 0, 'invalid': 0}
        pending_flush = {'processed': 0, 'failed': 0, 'duplicates': 0, 'invalid': 0}

        async def record(outcome: str):
            async with lock:
                totals[outcome] += 1
                pending_flush[outcome] += 1
                if sum(pending_flush.values()) >= self.settings.BATCH_SIZE:
                    await self._flush(job_id, pending_flush)

        async def worker(row: Dict[str, Any]):
            async with semaphore:
                try:
                    outcome = await self._process_one(row, job_id)
                except Exception as e:
                    # A single malformed row must never crash the whole
                    # 50,000-row batch via gather's default fail-fast
                    # behavior - every other row still deserves a chance.
                    logger.error(f"Unexpected error processing row {row.get('email')}: {e}")
                    await self.failed_repo.create({
                        'lead_id': f"unprocessable:{uuid4()}",
                        'job_id': job_id,
                        'email': row.get('email'),
                        'stage': 'ingestion',
                        'error': str(e),
                        'retry_count': 0,
                    })
                    outcome = 'failed'
                await record(outcome)

        try:
            await asyncio.gather(*(worker(row) for row in rows))

            async with lock:
                if any(pending_flush.values()):
                    await self._flush(job_id, pending_flush)

            await self.job_repo.mark_completed(job_id)

            logger.info(
                f"✓ Ingestion complete for job {job_id}: "
                f"processed={totals['processed']} duplicates={totals['duplicates']} "
                f"invalid={totals['invalid']} failed={totals['failed']}"
            )

        except Exception as e:
            logger.error(f"✗ Ingestion failed for job {job_id}: {e}")
            await self.job_repo.mark_failed(job_id, str(e))

    async def _flush(self, job_id: str, pending_flush: Dict[str, int]):
        """Flush accumulated counters to tbl_jobs in one write, then reset them."""
        await self.job_repo.increment_metrics(
            job_id,
            processed=pending_flush['processed'],
            failed=pending_flush['failed'],
            duplicates=pending_flush['duplicates'],
            invalid=pending_flush['invalid'],
        )
        for key in pending_flush:
            pending_flush[key] = 0

    async def _process_one(self, row: Dict[str, Any], job_id: str) -> str:
        """
        Process a single row: validate, then attempt an idempotent
        create with retry/backoff, dead-lettering on permanent failure.

        Returns one of: 'processed', 'duplicates', 'invalid', 'failed'.
        """
        is_valid, error = validate_lead(row)
        if not is_valid:
            logger.warning(f"Invalid lead ({row.get('email')}): {error}")
            return 'invalid'

        item = build_lead_item(row, job_id)

        for attempt in range(self.settings.MAX_RETRIES):
            try:
                created = await self.lead_repo.create_if_absent(item)
                if created:
                    return 'processed'
                logger.info(f"Duplicate lead: {item['email']}")
                return 'duplicates'
            except Exception as e:
                is_last_attempt = attempt == self.settings.MAX_RETRIES - 1
                if not is_last_attempt:
                    backoff = min(
                        self.settings.RETRY_BACKOFF_BASE * (2 ** attempt),
                        self.settings.RETRY_BACKOFF_MAX
                    )
                    logger.warning(
                        f"Write failed for {item['email']} (attempt {attempt + 1}/"
                        f"{self.settings.MAX_RETRIES}): {e}. Retrying in {backoff}s..."
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"All retries exhausted for {item['email']}: {e}")
                    await self.failed_repo.create({
                        'lead_id': item['lead_id'],
                        'job_id': job_id,
                        'email': item['email'],
                        'stage': 'ingestion',
                        'error': str(e),
                        'retry_count': self.settings.MAX_RETRIES,
                    })
                    return 'failed'

        return 'failed'  # unreachable, keeps type checkers happy


# Singleton instance
processing_service = ProcessingService()
