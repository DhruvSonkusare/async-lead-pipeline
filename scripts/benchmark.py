"""
scripts/benchmark.py
=====================
Part 6: process the 50,000-lead dataset end-to-end and report how long
it takes sequentially (CONCURRENT_WORKERS=1) vs. with bounded
concurrency, plus a fault-injection run showing the retry/DLQ path
under a 10% simulated write-failure rate.

This spins up its own local DynamoDB stand-in (moto's ThreadedMotoServer)
so the benchmark is self-contained and reproducible - point
DYNAMODB_ENDPOINT at real DynamoDB Local (docker compose up dynamodb)
instead if you want numbers against the genuine article.

Run: python scripts/benchmark.py
"""

import asyncio
import os
import platform
import sys
import time

# Allow running as `python scripts/benchmark.py` directly (not just
# `python -m scripts.benchmark`) by putting the project root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows consoles often default to a non-UTF-8 codepage that can't print
# the checkmarks in create_tables' log messages.
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ.setdefault('USE_LOCAL_DYNAMODB', 'True')
os.environ.setdefault('DYNAMODB_REGION', 'us-east-1')

import boto3
import pandas as pd
import psutil
from moto.server import ThreadedMotoServer

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'docs', 'leads_50k.csv')
# Full 50,000-row run against moto's single-process dev server takes several
# minutes per pass (see README "What didn't work"); default to a smaller,
# still-representative sample and scale BENCHMARK_ROWS=50000 up yourself
# against real DynamoDB Local (docker compose up dynamodb) for the actual
# submission numbers.
TARGET_ROWS = int(os.environ.get('BENCHMARK_ROWS', '5000'))
ALL_TABLES = ['tbl_leads', 'tbl_jobs', 'tbl_failed_leads']


def start_local_dynamodb():
    """
    If DYNAMODB_ENDPOINT is already set (e.g. pointing at a real
    `docker compose up dynamodb`), use that and don't start anything -
    real numbers beat moto's single-process dev server. Otherwise fall
    back to a self-contained moto server so this script still runs with
    zero setup.
    """
    if os.environ.get('DYNAMODB_ENDPOINT'):
        print(f"Using existing DYNAMODB_ENDPOINT={os.environ['DYNAMODB_ENDPOINT']}\n")
        return None

    server = ThreadedMotoServer(port=0, verbose=False)
    server.start()
    port = server._server.socket.getsockname()[1]
    os.environ['DYNAMODB_ENDPOINT'] = f'http://127.0.0.1:{port}'
    print(f"No DYNAMODB_ENDPOINT set - started a self-contained moto stand-in on port {port}\n")
    return server


def reset_tables():
    from app.config import get_settings
    from scripts.create_tables import create_tables

    settings = get_settings()
    client = boto3.client(
        'dynamodb', region_name=settings.DYNAMODB_REGION,
        endpoint_url=settings.DYNAMODB_ENDPOINT,
        aws_access_key_id='testing', aws_secret_access_key='testing',
    )
    for table_name in ALL_TABLES:
        try:
            client.delete_table(TableName=table_name)
        except client.exceptions.ResourceNotFoundException:
            pass
    create_tables()


def load_rows(n=TARGET_ROWS):
    df = pd.read_csv(DATASET_PATH).head(n)
    df = df.where(pd.notnull(df), None)
    return df.to_dict('records')


async def run_ingestion(rows, concurrency, job_label):
    from app.config import get_settings
    from app.db.repository import JobRepository
    from app.services.processing_service import processing_service
    from app.utils import utcnow_iso
    from uuid import uuid4

    get_settings().CONCURRENT_WORKERS = concurrency
    job_repo = JobRepository()
    job_id = str(uuid4())
    await job_repo.create({
        'job_id': job_id, 'status': 'ingesting', 'filename': job_label,
        'total_leads': len(rows), 'created_at': utcnow_iso(),
    })

    start = time.perf_counter()
    await processing_service.process_upload(job_id, rows)
    elapsed = time.perf_counter() - start

    job = await job_repo.get_by_id(job_id)
    return elapsed, job


async def main():
    print(f"Machine: {platform.system()} {platform.release()}, "
          f"{platform.processor() or platform.machine()}, "
          f"{psutil.cpu_count(logical=True)} logical CPUs, "
          f"{round(psutil.virtual_memory().total / (1024**3), 1)} GB RAM")
    print(f"Python: {platform.python_version()}")

    rows = load_rows()
    print(f"Loaded {len(rows)} rows from {DATASET_PATH}\n")

    from app.config import get_settings
    from app.db.dynamodb import init_db
    settings = get_settings()
    await init_db()

    # ---- Run 1: sequential (CONCURRENT_WORKERS=1) ----
    reset_tables()
    settings.SIMULATED_FAILURE_RATE = 0.0
    elapsed_seq, job_seq = await run_ingestion(rows, concurrency=1, job_label="benchmark-sequential")
    print(f"[Sequential, concurrency=1]   {elapsed_seq:.2f}s  "
          f"({len(rows)/elapsed_seq:.1f} leads/sec)  {job_seq}\n")

    # ---- Run 2: bounded concurrency ----
    reset_tables()
    concurrency = 20
    elapsed_conc, job_conc = await run_ingestion(rows, concurrency=concurrency, job_label="benchmark-concurrent")
    print(f"[Concurrent, concurrency={concurrency}]  {elapsed_conc:.2f}s  "
          f"({len(rows)/elapsed_conc:.1f} leads/sec)  {job_conc}\n")

    speedup = elapsed_seq / elapsed_conc if elapsed_conc else float('inf')
    print(f"Speedup from concurrency: {speedup:.2f}x\n")

    # ---- Run 3: fault injection (10% simulated write failures) ----
    # Backoff is shortened for this demo run only (production defaults are
    # 1s/30s) so a benchmark doesn't spend most of its time sleeping.
    reset_tables()
    settings.SIMULATED_FAILURE_RATE = 0.10
    settings.RETRY_BACKOFF_BASE = 0.05
    settings.RETRY_BACKOFF_MAX = 0.2
    elapsed_fault, job_fault = await run_ingestion(rows, concurrency=concurrency, job_label="benchmark-fault-injection")
    settings.SIMULATED_FAILURE_RATE = 0.0
    print(f"[Concurrent + 10% simulated write failures]  {elapsed_fault:.2f}s  {job_fault}")
    print("(compare 'failed' above to total_leads: with MAX_RETRIES=3 this should be "
          "roughly 0.1%% of ingested rows, not 10%%, since almost all failures succeed on retry)")


if __name__ == "__main__":
    server = start_local_dynamodb()
    try:
        asyncio.run(main())
    finally:
        if server is not None:
            server.stop()
