"""
app/db/dynamodb.py
==================
Async DynamoDB connection management (aioboto3) and fault injection.

Handles:
  - A single long-lived aioboto3 resource/client, opened once at app
    startup and closed at shutdown (see app/main.py's lifespan handler).
  - Simulated write failures, used to exercise retry/backoff/DLQ logic
    on demand (Part 5 of the assessment).
"""

import random
import logging
from contextlib import AsyncExitStack
from typing import Optional

import aioboto3

from app.config import get_settings

logger = logging.getLogger(__name__)


class DynamoDBConnection:
    """Owns the app's long-lived aioboto3 DynamoDB resource + client."""

    def __init__(self):
        self.settings = get_settings()
        self.session = aioboto3.Session()
        self._stack: Optional[AsyncExitStack] = None
        self.dynamodb = None  # aioboto3 DynamoDB resource (for Table())
        self.client = None    # aioboto3 DynamoDB client (for list_tables/describe_table)

    def _connection_kwargs(self) -> dict:
        if self.settings.USE_LOCAL_DYNAMODB:
            return dict(
                endpoint_url=self.settings.DYNAMODB_ENDPOINT,
                region_name=self.settings.DYNAMODB_REGION,
                aws_access_key_id='testing',
                aws_secret_access_key='testing',
            )
        return dict(
            region_name=self.settings.DYNAMODB_REGION,
            aws_access_key_id=self.settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.settings.AWS_SECRET_ACCESS_KEY,
        )

    async def connect(self):
        """Open the long-lived resource/client. Call once, from app startup."""
        kwargs = self._connection_kwargs()
        target = kwargs.get('endpoint_url', self.settings.DYNAMODB_REGION)
        logger.info(f"Connecting to DynamoDB: {target}")

        self._stack = AsyncExitStack()
        self.dynamodb = await self._stack.enter_async_context(
            self.session.resource('dynamodb', **kwargs)
        )
        self.client = await self._stack.enter_async_context(
            self.session.client('dynamodb', **kwargs)
        )
        logger.info("✓ Connected to DynamoDB")

    async def close(self):
        """Close the resource/client. Call once, from app shutdown."""
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self.dynamodb = None
            self.client = None

    async def get_table(self, table_name: str):
        """Get an aioboto3 DynamoDB table resource. Cheap - does not hit the network."""
        if self.dynamodb is None:
            raise RuntimeError("DynamoDB not connected. Call connect() first.")
        return await self.dynamodb.Table(table_name)

    async def list_tables(self) -> list:
        try:
            response = await self.client.list_tables()
            return response.get('TableNames', [])
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return []

    async def table_exists(self, table_name: str) -> bool:
        try:
            await self.client.describe_table(TableName=table_name)
            return True
        except self.client.exceptions.ResourceNotFoundException:
            return False
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False


# ============================================================================
# Singleton accessor - opened/closed once via the app's lifespan handler
# ============================================================================

_db_connection: Optional[DynamoDBConnection] = None


def get_db_connection() -> DynamoDBConnection:
    """Get the DynamoDB connection singleton (does not connect it)."""
    global _db_connection
    if _db_connection is None:
        _db_connection = DynamoDBConnection()
    return _db_connection


async def init_db() -> DynamoDBConnection:
    """Connect the singleton DynamoDB connection. Call from app startup."""
    db = get_db_connection()
    await db.connect()
    return db


async def close_db():
    """
    Close the singleton DynamoDB connection's resources. Call from app
    shutdown.

    Deliberately does NOT drop the singleton reference itself: repositories
    are constructed once (at module import time) and cache a reference to
    this object via get_db_connection(), so the same DynamoDBConnection
    instance must survive to be reconnected later (e.g. a second app
    lifespan cycle in tests) - only its internal resource/client are torn
    down and rebuilt.
    """
    if _db_connection is not None:
        await _db_connection.close()


# ============================================================================
# Fault injection (Part 5: "make 10% of database writes fail randomly")
#
# Disabled by default (SIMULATED_FAILURE_RATE=0.0). Every write path in
# repository.py calls simulate_write_failure() immediately before the real
# aioboto3 call, on every attempt (including retries) - so enabling it
# demonstrates the retry/backoff/DLQ logic without needing a real DynamoDB
# outage.
# ============================================================================

class SimulatedWriteFailure(Exception):
    """Raised in place of a real DynamoDB error, to exercise retry/backoff/DLQ paths."""


def simulate_write_failure():
    settings = get_settings()
    if settings.SIMULATED_FAILURE_RATE > 0 and random.random() < settings.SIMULATED_FAILURE_RATE:
        raise SimulatedWriteFailure("Simulated DynamoDB write failure (fault injection)")
