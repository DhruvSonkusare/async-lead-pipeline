"""
app/services/lead_service.py
============================
Lead validation and the deterministic identity/dedup key.

Classification is deliberately NOT done here - a lead is stored with
classification_status='pending' and picked up later by the scheduled
worker (app/workers/classification_worker.py). This keeps ingestion
and classification fully decoupled, per Part 5 of the assessment.
"""

import hashlib
import logging
from typing import Dict, Any, Tuple, Optional

from app.utils import utcnow_iso

logger = logging.getLogger(__name__)


def compute_lead_id(email: str, message: str) -> str:
    """
    Deterministic identity for a lead: sha256(email#message), both
    lower-cased and trimmed first.

    This is the whole idempotency + duplicate-detection mechanism:
      - same email + identical message (e.g. the same file re-uploaded)
        -> same lead_id -> a conditional create is rejected as a duplicate
      - same email + a genuinely different message -> a different
        lead_id -> stored as its own lead, so new signal isn't lost
    """
    normalized = f"{email.strip().lower()}#{message.strip().lower()}"
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def validate_lead(lead: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a raw lead row. Returns (is_valid, error_message)."""

    name = lead.get('name')
    if not name or not isinstance(name, str) or not name.strip():
        return False, "Invalid name"
    if len(name) > 200:
        return False, "Name too long (max 200 characters)"

    email = lead.get('email')
    if not email or not isinstance(email, str) or '@' not in email:
        return False, "Invalid email"

    message = lead.get('message')
    if not message or not isinstance(message, str) or not message.strip():
        return False, "Invalid message"
    if len(message) > 5000:
        return False, "Message too long (max 5000 characters)"

    return True, None


def build_lead_item(lead: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """
    Build the tbl_leads item for a validated lead.

    city is omitted entirely (not set to None) when absent, so it never
    appears in gsi_city_date - a GSI key attribute can't hold a NULL,
    and leaving it out is what keeps that index correctly sparse.
    intent / intent_confidence / processed_at are likewise omitted -
    they don't exist until the classification worker sets them.
    """
    email = lead['email'].strip().lower()
    message = lead['message'].strip()

    item = {
        'lead_id': compute_lead_id(email, message),
        'email': email,
        'name': lead['name'].strip(),
        'message': message,
        'classification_status': 'pending',
        'job_id': job_id,
        'created_at': utcnow_iso(),
        'retry_count': 0,
    }

    if lead.get('phone'):
        item['phone'] = lead['phone']
    if lead.get('company'):
        item['company'] = lead['company']
    if lead.get('city'):
        item['city'] = lead['city']

    return item
