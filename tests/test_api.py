# tests/test_api.py
#
# `client.portal` (from starlette's TestClient) runs coroutines on the
# SAME event loop the app's lifespan/aioboto3 connection was opened on.
# A separate asyncio.run() call would spin up a different event loop,
# which aiohttp's connection objects can't be reused across (Windows
# raises "attached to a different loop") - so async helpers below are
# invoked as `client.portal.call(func, *args)`, never asyncio.run().

from app.services.lead_service import compute_lead_id
from app.workers.classification_worker import classification_worker
from app.db.repository import LeadRepository


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_upload_valid_csv_is_processed(client, make_csv, sample_lead):
    csv_bytes = make_csv([sample_lead])
    response = client.post("/upload", files={"file": ("leads.csv", csv_bytes, "text/csv")})
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    status = client.get(f"/jobs/{job_id}").json()
    assert status["total_leads"] == 1
    assert status["processed"] == 1
    assert status["duplicates"] == 0
    assert status["invalid"] == 0
    assert status["failed"] == 0
    assert status["status"] == "ingested"


def test_upload_rejects_missing_required_columns(client, make_csv):
    csv_bytes = make_csv([{"name": "John", "email": "john@example.com"}])  # no "message"
    response = client.post("/upload", files={"file": ("leads.csv", csv_bytes, "text/csv")})
    assert response.status_code == 400


def test_upload_rejects_empty_file(client):
    response = client.post("/upload", files={"file": ("leads.csv", b"", "text/csv")})
    assert response.status_code == 400


def test_upload_counts_invalid_rows(client, make_csv, sample_lead):
    invalid_row = dict(sample_lead)
    invalid_row["email"] = ""  # missing/invalid email
    csv_bytes = make_csv([sample_lead, invalid_row])

    response = client.post("/upload", files={"file": ("leads.csv", csv_bytes, "text/csv")})
    job_id = response.json()["job_id"]

    status = client.get(f"/jobs/{job_id}").json()
    assert status["processed"] == 1
    assert status["invalid"] == 1


def test_upload_deduplicates_identical_email_and_message(client, make_csv, sample_lead):
    csv_bytes = make_csv([sample_lead, sample_lead])  # identical row twice
    response = client.post("/upload", files={"file": ("leads.csv", csv_bytes, "text/csv")})
    job_id = response.json()["job_id"]

    status = client.get(f"/jobs/{job_id}").json()
    assert status["processed"] == 1
    assert status["duplicates"] == 1


def test_upload_keeps_two_different_messages_from_same_email(client, make_csv, sample_lead):
    second = dict(sample_lead)
    second["message"] = "Actually, not interested anymore."
    csv_bytes = make_csv([sample_lead, second])

    response = client.post("/upload", files={"file": ("leads.csv", csv_bytes, "text/csv")})
    job_id = response.json()["job_id"]

    status = client.get(f"/jobs/{job_id}").json()
    assert status["processed"] == 2
    assert status["duplicates"] == 0


def test_get_job_not_found_returns_404(client):
    response = client.get("/jobs/does-not-exist")
    assert response.status_code == 404


def test_list_jobs_orders_by_created_at_not_random_job_id(client, make_csv, sample_lead):
    # Regression test: list_all() used to sort by job_id (a random UUID,
    # the table's sort key) descending, which has nothing to do with
    # actual upload time. Uploading twice in sequence must always show
    # the second upload before the first, regardless of what their
    # random job_ids happen to sort to alphabetically.
    first = client.post("/upload", files={"file": ("a.csv", make_csv([sample_lead]), "text/csv")}).json()
    second_lead = dict(sample_lead)
    second_lead["email"] = "second-upload@example.com"
    second = client.post("/upload", files={"file": ("b.csv", make_csv([second_lead]), "text/csv")}).json()

    jobs = client.get("/jobs").json()["jobs"]
    job_ids_in_order = [j["job_id"] for j in jobs]
    assert job_ids_in_order.index(second["job_id"]) < job_ids_in_order.index(first["job_id"])


def test_list_jobs_includes_new_upload(client, make_csv, sample_lead):
    csv_bytes = make_csv([sample_lead])
    job_id = client.post("/upload", files={"file": ("leads.csv", csv_bytes, "text/csv")}).json()["job_id"]

    jobs = client.get("/jobs").json()["jobs"]
    assert any(j["job_id"] == job_id for j in jobs)


def test_retries_recover_from_transient_write_failures(client, make_csv, sample_lead, fail_n_times):
    fail_n_times(2)  # first 2 write attempts fail, 3rd (last, within MAX_RETRIES=3) succeeds
    csv_bytes = make_csv([sample_lead])
    response = client.post("/upload", files={"file": ("leads.csv", csv_bytes, "text/csv")})
    job_id = response.json()["job_id"]

    status = client.get(f"/jobs/{job_id}").json()
    assert status["processed"] == 1
    assert status["failed"] == 0


def test_permanent_write_failure_goes_to_dead_letter_queue(client, make_csv, sample_lead, fail_n_times, failed_repo):
    fail_n_times(10)  # exceeds MAX_RETRIES - every attempt fails
    csv_bytes = make_csv([sample_lead])
    response = client.post("/upload", files={"file": ("leads.csv", csv_bytes, "text/csv")})
    job_id = response.json()["job_id"]

    status = client.get(f"/jobs/{job_id}").json()
    assert status["processed"] == 0
    assert status["failed"] == 1

    failures = client.portal.call(failed_repo.get_by_job_id, job_id)
    assert len(failures) == 1
    assert failures[0]["stage"] == "ingestion"


def test_classification_worker_classifies_pending_leads(client, make_csv, sample_lead, lead_repo):
    csv_bytes = make_csv([sample_lead])
    client.post("/upload", files={"file": ("leads.csv", csv_bytes, "text/csv")})

    lead_id = compute_lead_id(sample_lead["email"], sample_lead["message"])
    lead = client.portal.call(lead_repo.get_by_id, lead_id)
    assert lead["classification_status"] == "pending"

    client.portal.call(classification_worker.run_once)

    lead = client.portal.call(lead_repo.get_by_id, lead_id)
    assert lead["classification_status"] == "done"
    assert lead["intent"] in ("hot", "warm", "cold")
    assert 0.0 <= lead["intent_confidence"] <= 1.0


def test_worker_tick_survives_claim_failures(client, make_csv, sample_lead, lead_repo, fail_n_times):
    # Regression test: claim_for_classification() used to be unguarded -
    # a single simulated/transient failure on the claim step crashed the
    # *entire* worker tick (via asyncio.gather's fail-fast default),
    # abandoning every other pending lead in that batch too.
    csv_bytes = make_csv([sample_lead])
    client.post("/upload", files={"file": ("leads.csv", csv_bytes, "text/csv")})

    fail_n_times(999)  # every write from here on fails, including the claim
    result = client.portal.call(classification_worker.run_once)  # must not raise
    assert result["classified"] == 0

    lead_id = compute_lead_id(sample_lead["email"], sample_lead["message"])
    lead = client.portal.call(lead_repo.get_by_id, lead_id)
    assert lead["classification_status"] == "pending"  # untouched, safe to retry next tick


def test_list_leads_filters_by_classified_intent(client, make_csv, sample_lead):
    csv_bytes = make_csv([sample_lead])
    client.post("/upload", files={"file": ("leads.csv", csv_bytes, "text/csv")})
    client.portal.call(classification_worker.run_once)

    lead_id = compute_lead_id(sample_lead["email"], sample_lead["message"])
    lead = client.portal.call(LeadRepository().get_by_id, lead_id)
    intent = lead["intent"]

    results = client.get(f"/leads?intent={intent}").json()["leads"]
    assert any(item["lead_id"] == lead_id for item in results)
