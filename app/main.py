"""
app/main.py
===========
FastAPI application initialization and setup.

Handles:
  - FastAPI app creation
  - Database connection lifecycle (aioboto3, opened/closed via lifespan)
  - Scheduled classification worker lifecycle
  - Route registration
  - Error handling
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Windows consoles often default to a non-UTF-8 codepage that can't print
# the checkmarks in the log messages below.
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from app.config import get_settings
from app.db.dynamodb import init_db, close_db
from app.api import routes
from app.utils import utcnow_iso

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect DynamoDB and start the classification worker on startup; reverse on shutdown."""
    settings = get_settings()
    logger.info("=" * 80)
    logger.info(f"Starting {settings.APP_NAME}")
    logger.info("=" * 80)

    db = await init_db()

    required_tables = ['tbl_leads', 'tbl_jobs', 'tbl_failed_leads']
    existing = await db.list_tables()
    for table in required_tables:
        if table in existing:
            logger.info(f"  ✓ {table}")
        else:
            logger.warning(f"  ✗ {table} - missing! Run: python scripts/create_tables.py")

    from app.workers.classification_worker import classification_worker
    worker_task = asyncio.create_task(classification_worker.run_forever())

    logger.info("✓ Application startup complete")
    logger.info("=" * 80)

    yield

    logger.info("Shutting down application...")
    classification_worker.stop()
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    await close_db()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description="Async Lead Processing Pipeline with ML Classification",
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    app.include_router(routes.router)

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": utcnow_iso(),
        "app": "Async Lead Processing Pipeline"
    }


@app.get("/")
async def root():
    settings = get_settings()
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()

    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
