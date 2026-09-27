"""
scripts/create_tables.py
========================
DynamoDB table creation script with proper schema design.

Creates:
  - tbl_leads: Leads, keyed by a deterministic hash of (email, message)
      so that re-ingesting the same content is naturally idempotent.
  - tbl_jobs: Tracks each upload's progress, keyed so that "list every
      upload" is a Query on a single partition, never a table Scan.
  - tbl_failed_leads: Dead-letter queue for leads that permanently
      failed (ingestion or classification) after exhausting retries.

Run: python scripts/create_tables.py

See README.md "Storage design" section for the access-pattern
justification behind each key and index below.
"""

import os
import sys
import boto3
import logging

# Allow running as `python scripts/create_tables.py` directly (not just
# `python -m scripts.create_tables`) by putting the project root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles often default to a non-UTF-8 codepage that can't print
# the checkmarks below via a plain print() - force UTF-8 so this script
# doesn't crash on its own success message.
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _get_client(settings):
    if settings.USE_LOCAL_DYNAMODB:
        logger.info(f"Connecting to DynamoDB Local: {settings.DYNAMODB_ENDPOINT}")
        return boto3.client(
            'dynamodb',
            endpoint_url=settings.DYNAMODB_ENDPOINT,
            region_name=settings.DYNAMODB_REGION,
            aws_access_key_id='testing',
            aws_secret_access_key='testing'
        )
    logger.info(f"Connecting to AWS DynamoDB: {settings.DYNAMODB_REGION}")
    return boto3.client('dynamodb', region_name=settings.DYNAMODB_REGION)


def _create_table(dynamodb, **kwargs):
    table_name = kwargs['TableName']
    try:
        dynamodb.create_table(**kwargs)
        logger.info(f"✓ Table {table_name} created successfully")
    except dynamodb.exceptions.ResourceInUseException:
        logger.info(f"✓ Table {table_name} already exists")
    except Exception as e:
        logger.error(f"Error creating {table_name}: {e}")
        raise


def create_tables():
    """Create all required DynamoDB tables"""

    settings = get_settings()
    dynamodb = _get_client(settings)

    # ========================================================================
    # TABLE 1: tbl_leads
    #
    # PK: lead_id = sha256(email.lower().strip() + "#" + message.lower().strip())
    #   Deterministic (not a random UUID) so a conditional PutItem
    #   (attribute_not_exists(lead_id)) makes re-ingesting the exact same
    #   (email, message) pair a no-op instead of a duplicate row - this is
    #   the idempotency guarantee Part 2 asks for. A different message from
    #   the same email is a *different* lead_id, so a returning contact's
    #   new inquiry is never silently dropped.
    #
    # GSIs, one per query the API actually needs:
    #   - gsi_status_created: the scheduled classification worker's only
    #     query is "give me pending leads, oldest first". Self-shrinking:
    #     once a lead's classification_status flips to 'done', it falls
    #     out of the 'pending' partition automatically.
    #   - gsi_intent_date / gsi_city_date: GET /leads filters by intent
    #     and/or city plus a date range. Each is queried on whichever
    #     filter is present/more selective; the other filter (if both are
    #     given) is applied as a FilterExpression / in-memory pass. Both
    #     are sparse indexes - leads without a city (optional field) or
    #     not yet classified (no intent yet) simply don't appear in them.
    # ========================================================================

    logger.info("Creating table: tbl_leads...")
    _create_table(
        dynamodb,
        TableName='tbl_leads',
        KeySchema=[
            {'AttributeName': 'lead_id', 'KeyType': 'HASH'}
        ],
        AttributeDefinitions=[
            {'AttributeName': 'lead_id', 'AttributeType': 'S'},
            {'AttributeName': 'classification_status', 'AttributeType': 'S'},
            {'AttributeName': 'intent', 'AttributeType': 'S'},
            {'AttributeName': 'city', 'AttributeType': 'S'},
            {'AttributeName': 'created_at', 'AttributeType': 'S'},
        ],
        GlobalSecondaryIndexes=[
            {
                'IndexName': 'gsi_status_created',
                'KeySchema': [
                    {'AttributeName': 'classification_status', 'KeyType': 'HASH'},
                    {'AttributeName': 'created_at', 'KeyType': 'RANGE'},
                ],
                'Projection': {'ProjectionType': 'ALL'},
            },
            {
                'IndexName': 'gsi_intent_date',
                'KeySchema': [
                    {'AttributeName': 'intent', 'KeyType': 'HASH'},
                    {'AttributeName': 'created_at', 'KeyType': 'RANGE'},
                ],
                'Projection': {'ProjectionType': 'ALL'},
            },
            {
                'IndexName': 'gsi_city_date',
                'KeySchema': [
                    {'AttributeName': 'city', 'KeyType': 'HASH'},
                    {'AttributeName': 'created_at', 'KeyType': 'RANGE'},
                ],
                'Projection': {'ProjectionType': 'ALL'},
            },
        ],
        BillingMode='PAY_PER_REQUEST',
    )

    # ========================================================================
    # TABLE 2: tbl_jobs
    #
    # PK: list_key (constant value "JOB" on every item)
    # SK: job_id
    #   One row per upload (not per lead). The constant partition key
    #   means GET /jobs/{job_id} is still an instant GetItem (job_id is
    #   the sort key), while GET /jobs (list every upload) is a Query on
    #   a single partition rather than a table Scan. This trades some
    #   write throughput (every job and every job-counter update shares
    #   one partition) for a much simpler listing pattern - acceptable
    #   because jobs are created per-upload, not per-lead, so volume is
    #   orders of magnitude lower than tbl_leads.
    # ========================================================================

    logger.info("Creating table: tbl_jobs...")
    _create_table(
        dynamodb,
        TableName='tbl_jobs',
        KeySchema=[
            {'AttributeName': 'list_key', 'KeyType': 'HASH'},
            {'AttributeName': 'job_id', 'KeyType': 'RANGE'},
        ],
        AttributeDefinitions=[
            {'AttributeName': 'list_key', 'AttributeType': 'S'},
            {'AttributeName': 'job_id', 'AttributeType': 'S'},
        ],
        BillingMode='PAY_PER_REQUEST',
    )

    # ========================================================================
    # TABLE 3: tbl_failed_leads (dead-letter queue)
    #
    # PK: lead_id (same deterministic id as tbl_leads - ties a failure
    #     record back to the exact lead it happened to)
    # SK: failed_at
    #   Composite key so a lead that fails more than once (e.g. fails
    #   ingestion, later also fails classification) accumulates a
    #   *history* of failures instead of one entry overwriting the last.
    #
    # GSI gsi_job_failed: "show me every failure for this upload" -
    #   powers GET /failed-leads/{job_id}.
    # ========================================================================

    logger.info("Creating table: tbl_failed_leads...")
    _create_table(
        dynamodb,
        TableName='tbl_failed_leads',
        KeySchema=[
            {'AttributeName': 'lead_id', 'KeyType': 'HASH'},
            {'AttributeName': 'failed_at', 'KeyType': 'RANGE'},
        ],
        AttributeDefinitions=[
            {'AttributeName': 'lead_id', 'AttributeType': 'S'},
            {'AttributeName': 'failed_at', 'AttributeType': 'S'},
            {'AttributeName': 'job_id', 'AttributeType': 'S'},
        ],
        GlobalSecondaryIndexes=[
            {
                'IndexName': 'gsi_job_failed',
                'KeySchema': [
                    {'AttributeName': 'job_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'failed_at', 'KeyType': 'RANGE'},
                ],
                'Projection': {'ProjectionType': 'ALL'},
            },
        ],
        BillingMode='PAY_PER_REQUEST',
    )

    logger.info("\n" + "=" * 80)
    logger.info("All tables created successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        create_tables()
        print("\n✓ Table creation complete! You can now use the application.")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        exit(1)
