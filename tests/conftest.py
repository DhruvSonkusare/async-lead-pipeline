# tests/conftest.py
#
# Test isolation strategy:
#   - moto's @mock_aws decorator does NOT work with aioboto3/aiobotocore
#     (it patches the sync HTTP stack only) - so we run moto's real
#     ThreadedMotoServer instead, and point the app at it via
#     DYNAMODB_ENDPOINT, exactly like pointing at DynamoDB Local.
#   - Env vars must be set, and the moto server must be running, BEFORE
#     `app.main` (and anything it imports) is ever imported, since the
#     settings/connection singletons are created at first use.

import atexit
import os

os.environ['USE_LOCAL_DYNAMODB'] = 'True'
os.environ['DYNAMODB_REGION'] = 'us-east-1'
os.environ['SIMULATED_FAILURE_RATE'] = '0.0'  # deterministic by default; tests override per-case

from moto.server import ThreadedMotoServer  # noqa: E402

_moto_server = ThreadedMotoServer(port=0, verbose=False)
_moto_server.start()
_moto_port = _moto_server._server.socket.getsockname()[1]
os.environ['DYNAMODB_ENDPOINT'] = f'http://127.0.0.1:{_moto_port}'
atexit.register(_moto_server.stop)

import boto3  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from scripts.create_tables import create_tables  # noqa: E402

_ALL_TABLES = ['tbl_leads', 'tbl_jobs', 'tbl_failed_leads']

# Keep the test suite fast - the actual backoff *behavior* is what's under
# test, not real wall-clock delays.
_settings = get_settings()
_settings.RETRY_BACKOFF_BASE = 0.01
_settings.RETRY_BACKOFF_MAX = 0.05

from app.main import app  # noqa: E402
from app.db.repository import LeadRepository, JobRepository, FailedLeadRepository  # noqa: E402


@pytest.fixture(autouse=True)
def reset_tables():
    """
    Fresh tables before every test.

    create_tables() alone is NOT enough here: it treats "table already
    exists" as success and leaves existing items untouched, so without
    an explicit delete first, every test after the first one would
    silently build on the previous test's leftover data.
    """
    client = boto3.client(
        'dynamodb', region_name='us-east-1',
        endpoint_url=os.environ['DYNAMODB_ENDPOINT'],
        aws_access_key_id='testing', aws_secret_access_key='testing',
    )
    for table_name in _ALL_TABLES:
        try:
            client.delete_table(TableName=table_name)
        except client.exceptions.ResourceNotFoundException:
            pass
    create_tables()
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def lead_repo():
    return LeadRepository()


@pytest.fixture
def job_repo():
    return JobRepository()


@pytest.fixture
def failed_repo():
    return FailedLeadRepository()


@pytest.fixture
def make_csv():
    """Build CSV bytes from a list of row dicts, for POST /upload."""
    import io
    import pandas as pd

    def _make(rows):
        df = pd.DataFrame(rows)
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return buf.read()

    return _make


@pytest.fixture
def sample_lead():
    return {
        "name": "John Doe",
        "email": "john@example.com",
        "message": "Ready to sign!",
        "city": "Mumbai",
        "company": "ACME Corp",
    }


class FailNTimes:
    """Deterministic stand-in for simulate_write_failure: raises for the
    first `n` calls, then is a no-op. Used to test retry/backoff/DLQ
    behavior without relying on real randomness."""

    def __init__(self, n: int):
        self.n = n
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.n:
            raise Exception("Forced failure for test")


@pytest.fixture
def fail_n_times(monkeypatch):
    def _apply(n: int):
        fake = FailNTimes(n)
        monkeypatch.setattr("app.db.repository.simulate_write_failure", fake)
        return fake

    return _apply
