"""
app/db/repository.py
====================
Async data access layer (aioboto3) using the repository pattern.

Every method that writes calls `simulate_write_failure()` immediately
before the real DynamoDB call, on every invocation (so callers that
retry re-roll the fault-injection chance on each attempt). Retry/backoff
and dead-letter handling live in the service layer, not here - these
repositories only ever make a single attempt per call.
"""

import base64
import json
import logging
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from botocore.exceptions import ClientError

from app.db.dynamodb import get_db_connection, simulate_write_failure
from app.utils import utcnow_iso

logger = logging.getLogger(__name__)


def encode_cursor(last_evaluated_key: Optional[dict]) -> Optional[str]:
    if not last_evaluated_key:
        return None
    return base64.urlsafe_b64encode(json.dumps(last_evaluated_key).encode()).decode()


def decode_cursor(cursor: Optional[str]) -> Optional[dict]:
    if not cursor:
        return None
    return json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())


class LeadRepository:
    """Repository for lead data operations"""

    TABLE_NAME = 'tbl_leads'

    def __init__(self):
        self.db = get_db_connection()

    async def _table(self):
        return await self.db.get_table(self.TABLE_NAME)

    async def create_if_absent(self, lead: Dict[str, Any]) -> bool:
        """
        Insert a lead if (and only if) its lead_id doesn't already exist.

        This is the idempotency + duplicate-detection mechanism: lead_id
        is a deterministic hash of (email, message), so re-ingesting the
        same content always maps to the same key.

        Returns:
            True if this call created the row, False if it already
            existed (i.e. this was a duplicate).

        Raises:
            Exception: on any non-conditional failure (real or
            simulated) - the caller is responsible for retrying.
        """
        simulate_write_failure()
        table = await self._table()
        try:
            await table.put_item(
                Item=lead,
                ConditionExpression='attribute_not_exists(lead_id)'
            )
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                return False
            raise

    async def get_by_id(self, lead_id: str) -> Optional[Dict]:
        table = await self._table()
        response = await table.get_item(Key={'lead_id': lead_id})
        return response.get('Item')

    async def claim_for_classification(self, lead_id: str) -> bool:
        """
        Atomically flip a lead from 'pending' to 'in_progress'.

        Guarantees the classification worker never processes the same
        lead twice, even if two worker ticks overlap: only the caller
        that wins the conditional update gets to classify it.
        """
        simulate_write_failure()
        table = await self._table()
        try:
            await table.update_item(
                Key={'lead_id': lead_id},
                UpdateExpression='SET #status = :in_progress',
                ConditionExpression='#status = :pending',
                ExpressionAttributeNames={'#status': 'classification_status'},
                ExpressionAttributeValues={
                    ':in_progress': 'in_progress',
                    ':pending': 'pending',
                }
            )
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                return False
            raise

    async def release_claim(self, lead_id: str):
        """Put a claimed lead back to 'pending' (e.g. classification attempt failed)."""
        simulate_write_failure()
        table = await self._table()
        await table.update_item(
            Key={'lead_id': lead_id},
            UpdateExpression='SET #status = :pending',
            ExpressionAttributeNames={'#status': 'classification_status'},
            ExpressionAttributeValues={':pending': 'pending'}
        )

    async def mark_classification_failed(self, lead_id: str):
        """
        Terminal state after exhausting classification retries. Deliberately
        NOT released back to 'pending' - otherwise a permanently-broken
        lead would be retried forever, every worker tick. Recovery from
        here is a deliberate action (inspect tbl_failed_leads, requeue).
        """
        simulate_write_failure()
        table = await self._table()
        await table.update_item(
            Key={'lead_id': lead_id},
            UpdateExpression='SET #status = :failed',
            ExpressionAttributeNames={'#status': 'classification_status'},
            ExpressionAttributeValues={':failed': 'failed'}
        )

    async def mark_classified(self, lead_id: str, intent: str, confidence: float):
        simulate_write_failure()
        table = await self._table()
        await table.update_item(
            Key={'lead_id': lead_id},
            UpdateExpression='''
                SET intent = :intent,
                    intent_confidence = :confidence,
                    #status = :done,
                    processed_at = :processed_at
            ''',
            ExpressionAttributeNames={'#status': 'classification_status'},
            ExpressionAttributeValues={
                ':intent': intent,
                # DynamoDB's resource-level API rejects native floats outright
                # ("Use Decimal types instead") - str() first avoids binary
                # float-to-Decimal artifacts (e.g. 0.1 -> 0.1000000000000000055...).
                ':confidence': Decimal(str(confidence)),
                ':done': 'done',
                ':processed_at': utcnow_iso(),
            }
        )

    async def query_pending(self, limit: int = 100) -> List[Dict]:
        """Leads awaiting classification, oldest first (used by the scheduled worker)."""
        table = await self._table()
        response = await table.query(
            IndexName='gsi_status_created',
            KeyConditionExpression='classification_status = :status',
            ExpressionAttributeValues={':status': 'pending'},
            Limit=limit,
            ScanIndexForward=True,
        )
        return response.get('Items', [])

    async def list_leads(
        self,
        city: Optional[str] = None,
        intent: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        List leads filtered by city / intent / a created_at date range,
        with real (cursor-based) pagination.

        DynamoDB can only efficiently query one attribute at a time via
        a GSI. When both city and intent are given, this queries
        `intent` (assumed the more common/selective business filter -
        "show me hot leads") and applies `city` as a FilterExpression;
        when only one is given, that one drives the Query directly.
        When neither is given, this falls back to a paginated Scan -
        a known, documented limitation for the unfiltered case.
        """
        table = await self._table()
        exclusive_start_key = decode_cursor(cursor)

        key_condition = None
        index_name = None
        expr_values: Dict[str, Any] = {}
        filter_parts = []

        if intent:
            index_name = 'gsi_intent_date'
            key_condition = 'intent = :intent'
            expr_values[':intent'] = intent
            if city:
                filter_parts.append('city = :city')
                expr_values[':city'] = city
        elif city:
            index_name = 'gsi_city_date'
            key_condition = 'city = :city'
            expr_values[':city'] = city

        if date_from and date_to:
            key_condition = f'{key_condition} AND created_at BETWEEN :date_from AND :date_to'
            expr_values[':date_from'] = date_from
            expr_values[':date_to'] = date_to
        elif date_from:
            key_condition = f'{key_condition} AND created_at >= :date_from'
            expr_values[':date_from'] = date_from
        elif date_to:
            key_condition = f'{key_condition} AND created_at <= :date_to'
            expr_values[':date_to'] = date_to

        kwargs: Dict[str, Any] = {'Limit': limit}
        if exclusive_start_key:
            kwargs['ExclusiveStartKey'] = exclusive_start_key

        if index_name:
            kwargs['IndexName'] = index_name
            kwargs['KeyConditionExpression'] = key_condition
            kwargs['ExpressionAttributeValues'] = expr_values
            if filter_parts:
                kwargs['FilterExpression'] = ' AND '.join(filter_parts)
            response = await table.query(**kwargs)
        else:
            # No indexed filter given - documented fallback, not the
            # common/expected path for this endpoint.
            response = await table.scan(**kwargs)

        items = response.get('Items', [])
        next_cursor = encode_cursor(response.get('LastEvaluatedKey'))
        return items, next_cursor


class JobRepository:
    """Repository for job (upload) data operations"""

    TABLE_NAME = 'tbl_jobs'
    LIST_KEY = 'JOB'

    def __init__(self):
        self.db = get_db_connection()

    async def _table(self):
        return await self.db.get_table(self.TABLE_NAME)

    async def create(self, job: Dict[str, Any]) -> str:
        table = await self._table()
        job_id = job['job_id']
        item = {
            'list_key': self.LIST_KEY,
            'job_id': job_id,
            'status': job.get('status', 'ingesting'),
            'filename': job.get('filename'),
            'total_leads': job.get('total_leads', 0),
            'processed': job.get('processed', 0),
            'failed': job.get('failed', 0),
            'duplicates': job.get('duplicates', 0),
            'invalid': job.get('invalid', 0),
            'created_at': job.get('created_at', utcnow_iso()),
            'completed_at': job.get('completed_at'),
            'error': job.get('error'),
            'updated_at': utcnow_iso(),
        }
        await table.put_item(Item=item)
        return job_id

    async def get_by_id(self, job_id: str) -> Optional[Dict]:
        table = await self._table()
        response = await table.get_item(Key={'list_key': self.LIST_KEY, 'job_id': job_id})
        return response.get('Item')

    async def list_all(self, limit: int = 100, cursor: Optional[str] = None) -> Tuple[List[Dict], Optional[str]]:
        """
        Every upload, most recent first - a Query against one partition,
        never a Scan.

        The sort key is job_id (a random UUID, needed so GET /jobs/{id}
        stays an instant GetItem), not created_at - so DynamoDB's own key
        order doesn't correspond to time at all. This page's items are
        re-sorted by created_at here instead of trusting ScanIndexForward.
        """
        table = await self._table()
        kwargs: Dict[str, Any] = {
            'KeyConditionExpression': 'list_key = :lk',
            'ExpressionAttributeValues': {':lk': self.LIST_KEY},
            'Limit': limit,
        }
        exclusive_start_key = decode_cursor(cursor)
        if exclusive_start_key:
            kwargs['ExclusiveStartKey'] = exclusive_start_key
        response = await table.query(**kwargs)
        items = sorted(response.get('Items', []), key=lambda j: j['created_at'], reverse=True)
        return items, encode_cursor(response.get('LastEvaluatedKey'))

    async def update_status(self, job_id: str, status: str):
        table = await self._table()
        await table.update_item(
            Key={'list_key': self.LIST_KEY, 'job_id': job_id},
            UpdateExpression='SET #status = :status, updated_at = :updated_at',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':status': status,
                ':updated_at': utcnow_iso()
            }
        )

    async def increment_metrics(self, job_id: str, processed: int = 0, failed: int = 0,
                                 duplicates: int = 0, invalid: int = 0):
        """Batched counter flush - called once per BATCH_SIZE leads, not once per lead."""
        table = await self._table()
        await table.update_item(
            Key={'list_key': self.LIST_KEY, 'job_id': job_id},
            UpdateExpression='''
                SET #processed = #processed + :processed,
                    #failed = #failed + :failed,
                    #duplicates = #duplicates + :duplicates,
                    #invalid = #invalid + :invalid,
                    updated_at = :updated_at
            ''',
            ExpressionAttributeNames={
                '#processed': 'processed',
                '#failed': 'failed',
                '#duplicates': 'duplicates',
                '#invalid': 'invalid',
            },
            ExpressionAttributeValues={
                ':processed': processed,
                ':failed': failed,
                ':duplicates': duplicates,
                ':invalid': invalid,
                ':updated_at': utcnow_iso()
            }
        )

    async def mark_completed(self, job_id: str):
        table = await self._table()
        await table.update_item(
            Key={'list_key': self.LIST_KEY, 'job_id': job_id},
            UpdateExpression='SET #status = :status, completed_at = :completed_at, updated_at = :updated_at',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':status': 'ingested',
                ':completed_at': utcnow_iso(),
                ':updated_at': utcnow_iso()
            }
        )

    async def mark_failed(self, job_id: str, error: str):
        table = await self._table()
        await table.update_item(
            Key={'list_key': self.LIST_KEY, 'job_id': job_id},
            UpdateExpression='SET #status = :status, #error = :error, completed_at = :completed_at, updated_at = :updated_at',
            ExpressionAttributeNames={'#status': 'status', '#error': 'error'},
            ExpressionAttributeValues={
                ':status': 'failed',
                ':error': error,
                ':completed_at': utcnow_iso(),
                ':updated_at': utcnow_iso()
            }
        )


class FailedLeadRepository:
    """Repository for the dead-letter queue"""

    TABLE_NAME = 'tbl_failed_leads'

    def __init__(self):
        self.db = get_db_connection()

    async def _table(self):
        return await self.db.get_table(self.TABLE_NAME)

    async def create(self, failed_lead: Dict[str, Any]):
        """Log a permanent failure. lead_id + failed_at keeps a history per lead."""
        table = await self._table()
        item = {
            'lead_id': failed_lead['lead_id'],
            'failed_at': utcnow_iso(),
            'job_id': failed_lead.get('job_id'),
            'email': failed_lead.get('email'),
            'stage': failed_lead.get('stage'),  # 'ingestion' | 'classification'
            'error': failed_lead.get('error'),
            'retry_count': failed_lead.get('retry_count', 0),
        }
        await table.put_item(Item=item)
        logger.error(f"✗ Lead added to DLQ: {item['lead_id']} ({item['stage']}) - {item['error']}")

    async def get_by_job_id(self, job_id: str) -> List[Dict]:
        table = await self._table()
        response = await table.query(
            IndexName='gsi_job_failed',
            KeyConditionExpression='job_id = :job_id',
            ExpressionAttributeValues={':job_id': job_id}
        )
        return response.get('Items', [])
