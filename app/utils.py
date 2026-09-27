"""
app/utils.py
============
Small shared helpers.
"""

from datetime import datetime, timezone


def utcnow_iso() -> str:
    """Timezone-aware UTC timestamp, ISO-8601 (datetime.utcnow() is deprecated)."""
    return datetime.now(timezone.utc).isoformat()
