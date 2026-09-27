"""
app/workers/classification_worker.py
=====================================
Part 5: a scheduled, Lambda-style handler that picks up unprocessed
(classification_status='pending') leads and classifies them.

`run_once()` is the actual handler - it takes no state beyond what it
reads from DynamoDB, so it can be invoked exactly as written by a real
AWS Lambda on an EventBridge schedule. For local dev, `run_forever()`
just calls it in a loop from an asyncio background task started by the
app's lifespan (see app/main.py) - a stand-in scheduler, not a
requirement that it run in-process.
"""

import asyncio
import logging
from typing import Dict, Any

from app.config import get_settings
from app.db.repository import LeadRepository, FailedLeadRepository
from app.services.classification_service import classification_service

logger = logging.getLogger(__name__)


class ClassificationWorker:

    def __init__(self):
        self.settings = get_settings()
        self.lead_repo = LeadRepository()
        self.failed_repo = FailedLeadRepository()
        self._stop = False

    async def run_forever(self):
        logger.info(
            f"Classification worker started (poll every "
            f"{self.settings.WORKER_POLL_INTERVAL_SECONDS}s)"
        )
        while not self._stop:
            try:
                await self.run_once()
            except Exception as e:
                logger.error(f"Classification worker tick failed: {e}")
            await asyncio.sleep(self.settings.WORKER_POLL_INTERVAL_SECONDS)

    def stop(self):
        self._stop = True

    async def run_once(self) -> Dict[str, int]:
        """One handler invocation: classify up to WORKER_BATCH_SIZE pending leads."""

        if classification_service is None or not classification_service.is_model_loaded():
            logger.warning("Classification model not loaded - skipping this tick")
            return {'classified': 0, 'failed': 0, 'skipped': 0}

        leads = await self.lead_repo.query_pending(limit=self.settings.WORKER_BATCH_SIZE)
        if not leads:
            return {'classified': 0, 'failed': 0, 'skipped': 0}

        logger.info(f"Classification worker: {len(leads)} pending lead(s)")

        semaphore = asyncio.Semaphore(self.settings.WORKER_CONCURRENCY)
        lock = asyncio.Lock()
        counts = {'classified': 0, 'failed': 0, 'skipped': 0}

        async def record(outcome: str):
            async with lock:
                counts[outcome] += 1

        async def worker(lead: Dict[str, Any]):
            async with semaphore:
                try:
                    outcome = await self._process_lead(lead)
                except Exception as e:
                    # Never let one lead's unexpected exception cancel every
                    # other in-flight lead in this tick via gather's default
                    # fail-fast behavior - the lead just stays 'pending' and
                    # gets picked up again next tick.
                    logger.error(f"Unexpected error processing lead {lead.get('lead_id')}: {e}")
                    outcome = 'skipped'
                await record(outcome)

        await asyncio.gather(*(worker(lead) for lead in leads))

        logger.info(
            f"✓ Classification tick done: classified={counts['classified']} "
            f"failed={counts['failed']} skipped={counts['skipped']}"
        )
        return counts

    async def _process_lead(self, lead: Dict[str, Any]) -> str:
        lead_id = lead['lead_id']

        # Conditional claim: guarantees this lead is never classified twice,
        # even if two worker ticks (or two worker processes) overlap. A
        # failed claim attempt (transient DB error, or simulated failure)
        # isn't retried here - the lead is still safely 'pending' and will
        # simply be picked up again on the next poll tick.
        try:
            claimed = await self.lead_repo.claim_for_classification(lead_id)
        except Exception as e:
            logger.warning(f"Could not claim lead {lead_id} this tick: {e}")
            return 'skipped'

        if not claimed:
            return 'skipped'

        for attempt in range(self.settings.MAX_RETRIES):
            try:
                intent, confidence = await asyncio.to_thread(
                    classification_service.classify, lead['message']
                )
                await self.lead_repo.mark_classified(lead_id, intent, confidence)
                return 'classified'
            except Exception as e:
                is_last_attempt = attempt == self.settings.MAX_RETRIES - 1
                if not is_last_attempt:
                    backoff = min(
                        self.settings.RETRY_BACKOFF_BASE * (2 ** attempt),
                        self.settings.RETRY_BACKOFF_MAX
                    )
                    logger.warning(
                        f"Classification write failed for {lead_id} "
                        f"(attempt {attempt + 1}/{self.settings.MAX_RETRIES}): {e}. "
                        f"Retrying in {backoff}s..."
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"Classification permanently failed for {lead_id}: {e}")
                    await self.lead_repo.mark_classification_failed(lead_id)
                    await self.failed_repo.create({
                        'lead_id': lead_id,
                        'job_id': lead.get('job_id'),
                        'email': lead.get('email'),
                        'stage': 'classification',
                        'error': str(e),
                        'retry_count': self.settings.MAX_RETRIES,
                    })
                    return 'failed'

        return 'failed'  # unreachable


# Singleton instance - started/stopped from app.main's lifespan
classification_worker = ClassificationWorker()
