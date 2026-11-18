"""Regression checks for cloud API shapes and independent, cancellable file work."""
import json
import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from qdrant_client import models

from app import database, document_jobs as jobs, routes, vectors
from app.database import Database, Identity, authenticate
from app.main import app
from app.processing import Chunk


class QueuedExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        self.calls.append((function, args))

    def run(self):
        function, args = self.calls.pop(0)
        function(*args)


class FakeDatabase:
    def __init__(self, user):
        self.user = user
        self.rows = {}
        self.originals = {}
        self.updates = []
        self.storage_calls = []

    def get(self, table, item_id):
        if item_id not in self.rows:
            raise HTTPException(404, "Document not found")
        return dict(self.rows[item_id])

    def insert(self, table, body):
        row = {"chunk_count": 0, "error": None, **body}
        self.rows[row["id"]] = row
        return dict(row)

    def update(self, table, item_id, body):
        self.updates.append(dict(body))
        self.rows[item_id].update(body)
        return dict(self.rows[item_id])

    def storage(self, method, path, **kwargs):
        self.storage_calls.append((method, path))
        if method == "POST":
            self.originals[path] = kwargs["content"]
        elif method == "DELETE":
            self.originals.pop(path, None)
        return SimpleNamespace(content=self.originals.get(path, b""))

    def delete(self, table, item_id):
        self.rows.pop(item_id)

    def list(self, table, **kwargs):
        return [dict(row) for row in self.rows.values()]


@pytest.fixture
def file_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "settings", replace(jobs.settings, qdrant_all_minilm_l6_v2=False))
    indexing, deleting = QueuedExecutor(), QueuedExecutor()
    monkeypatch.setattr(jobs, "pool", indexing)
    monkeypatch.setattr(jobs, "delete_pool", deleting)
    monkeypatch.setattr(jobs, "activities", {})
    monkeypatch.setattr(jobs, "cache", tmp_path / "uploads")
    monkeypatch.setattr(jobs, "cleanup_path", tmp_path / "cleanup.json")
    monkeypatch.setattr(jobs, "start_cleanup", lambda: None)
    monkeypatch.setattr(jobs, "embedding_tokenizer", lambda: None)
    monkeypatch.setattr(jobs, "extract", lambda *args, **kwargs: [Chunk("Evidence from the original.")])
    monkeypatch.setattr(jobs, "vector_store", lambda: pytest.fail("Unexpected vector connection"))
    db = FakeDatabase(Identity("alice", "test-token"))
    document = db.insert("documents", {"id": str(uuid4()), "title": "report.pdf", "source": "report.pdf",
        "storage_path": "alice/report.pdf", "mime_type": "application/pdf", "status": "failed"})
    return db, document, indexing, deleting


@pytest.mark.parametrize("status,body", [(200, []), (404, {}), (400, {"statusCode": "404"})])
def test_storage_removal_uses_bucket_endpoint_and_prefixes(monkeypatch, status, body):
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(status, json=body)
    monkeypatch.setattr(database, "settings", replace(database.settings,
        supabase_url="https://storage.example", supabase_key="public-test", storage_bucket="research-documents"))
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(database, "client", client)
        Database(Identity("alice", "test-token")).storage("DELETE", "alice/document/source.pdf")
    request = requests[0]
    assert str(request.url) == "https://storage.example/storage/v1/object/research-documents"
    assert json.loads(request.content) == {"prefixes": ["alice/document/source.pdf"]}
    assert request.headers["authorization"] == "Bearer test-token"


def test_storage_permission_failure_is_not_treated_as_success(monkeypatch):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(403, json={}))) as client:
        monkeypatch.setattr(database, "client", client)
        with pytest.raises(HTTPException) as error:
            Database(Identity("alice", "test-token")).storage("DELETE", "alice/report.pdf")
    assert error.value.status_code == 403


@pytest.mark.parametrize("url,port", [("https://cluster.example", 443),
    ("https://cluster.example:8443", 8443), ("http://localhost:6333", 6333)])
def test_qdrant_uses_https_default_and_preserves_explicit_ports(monkeypatch, url, port):
    calls = []
    monkeypatch.setattr(vectors, "settings", replace(vectors.settings, qdrant_url=url))
    monkeypatch.setattr(vectors, "QdrantClient", lambda **kwargs: calls.append(kwargs) or object())
    vectors.VectorStore()
    assert calls[0]["port"] == port


def test_legacy_collection_indexes_exist_before_strict_version_filters():
    schema = {"user_id": True, "document_id": True}
    calls = []
    expected = {"user_id", "document_id", "index_revision", "embedding_model", "chunking_version"}
    class StrictClient:
        def collection_exists(self, name):
            return True
        def get_collection(self, name):
            return SimpleNamespace(payload_schema=schema,
                config=SimpleNamespace(params=SimpleNamespace(vectors=models.VectorParams(size=3, distance="Cosine"))))
        def create_payload_index(self, name, field, kind, wait):
            calls.append((field, kind))
            schema[field] = kind
        def scroll(self, *args, **kwargs):
            assert set(schema) == expected
            return [SimpleNamespace(payload={"document_chunk_count": 1, "index_revision": "current"})], None
        def count(self, name, count_filter, exact):
            assert set(schema) == expected
            return SimpleNamespace(count=0 if count_filter.must_not else 1)
    store = vectors.VectorStore(client=StrictClient())
    assert not store.needs_reindex("alice", "doc")
    assert ("chunking_version", models.PayloadSchemaType.INTEGER) in calls
    store.ensure(3)
    assert len(calls) == 3  # Existing indexes are preserved and new indexes are created once.


def test_upload_retry_and_delete_are_accepted_before_processing(file_jobs, monkeypatch):
    db, _, indexing, deleting = file_jobs
    monkeypatch.setattr(routes, "Database", lambda user: db)
    app.dependency_overrides[authenticate] = lambda: db.user
    try:
        client = TestClient(app)
        response = client.post("/api/documents", files={"file": ("research.pdf", b"%PDF-original", "application/pdf")})
        assert response.status_code == 202
        document = response.json()
        assert document["status"] == "processing" and document["activity"]["phase"] == "Queued"
        assert len(indexing.calls) == 1 and db.storage_calls == []
        response = client.delete(f"/api/documents/{document['id']}")
        assert response.status_code == 202 and len(deleting.calls) == 1
        assert indexing.calls[0][1][-1].cancelled.is_set()
        assert db.storage_calls == []
        failed = db.insert("documents", {"id": str(uuid4()), "source": "report.pdf", "status": "failed"})
        response = client.post(f"/api/documents/{failed['id']}/retry")
        assert response.status_code == 202 and len(indexing.calls) == 2
    finally:
        app.dependency_overrides.clear()


def test_index_failure_retains_original_and_retry_uses_it(file_jobs, monkeypatch):
    db, document, indexing, _ = file_jobs
    class UnavailableStore:
        def index(self, *args, **kwargs):
            raise httpx.ConnectError("offline")
    monkeypatch.setattr(jobs, "vector_store", lambda: UnavailableStore())
    jobs.schedule_index(db, document, b"original PDF")
    indexing.run()
    failed = db.get("documents", document["id"])
    assert failed["status"] == "failed" and "HTTPS" in failed["error"]
    assert db.originals[document["storage_path"]] == b"original PDF"
    assert not jobs.activities
    class AvailableStore:
        def index(self, *args, **kwargs):
            pass
    monkeypatch.setattr(jobs, "vector_store", lambda: AvailableStore())
    jobs.schedule_index(db, failed)
    indexing.run()
    assert db.get("documents", document["id"])["status"] == "ready"
    assert db.storage_calls[-1][0] == "GET"


def test_cloud_file_job_skips_local_model_for_chunking(file_jobs, monkeypatch):
    db, document, indexing, _ = file_jobs
    calls = []
    monkeypatch.setattr(jobs, "settings", replace(jobs.settings, qdrant_all_minilm_l6_v2=True))
    monkeypatch.setattr(jobs, "embed", lambda *args, **kwargs: pytest.fail("Cloud ingestion loaded local embeddings"))
    def extract(*args, **kwargs):
        assert kwargs["semantic_embed"] is None
        return [Chunk("Cloud evidence")]
    monkeypatch.setattr(jobs, "extract", extract)
    monkeypatch.setattr(jobs, "vector_store", lambda: SimpleNamespace(index=lambda *args, **kwargs: calls.append(args)))
    jobs.schedule_index(db, document, b"original")
    indexing.run()
    assert db.get("documents", document["id"])["status"] == "ready"
    assert calls[0][3][0].text == "Cloud evidence"


def test_cancelling_active_index_cannot_resurrect_removed_document(file_jobs, monkeypatch):
    db, document, indexing, deleting = file_jobs
    started, release = threading.Event(), threading.Event()
    def extract(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return [Chunk("Evidence")]
    monkeypatch.setattr(jobs, "extract", extract)
    jobs.schedule_index(db, document, b"original")
    thread = threading.Thread(target=indexing.run)
    thread.start()
    try:
        assert started.wait(3)
        jobs.schedule_delete(db, db.get("documents", document["id"]))
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    deleting.run()
    assert document["id"] not in db.rows and document["storage_path"] not in db.originals
    assert not any(update.get("status") in {"ready", "failed"} for update in db.updates)
    assert not jobs.activities
    assert json.loads(jobs.cleanup_path.read_text()) == [{"user_id": "alice", "document_id": document["id"]}]


def test_failed_file_can_be_removed_when_qdrant_is_unavailable(file_jobs):
    db, document, _, deleting = file_jobs
    jobs.schedule_delete(db, document)
    deleting.run()
    assert document["id"] not in db.rows
    assert json.loads(jobs.cleanup_path.read_text())[0]["document_id"] == document["id"]
    assert not jobs.cleanup_path.with_suffix(".tmp").exists()


def test_ready_file_deletion_failure_remains_retryable(file_jobs, monkeypatch):
    db, document, _, deleting = file_jobs
    document = db.update("documents", document["id"], {"status": "ready", "chunk_count": 3})
    monkeypatch.setattr(jobs, "vector_store", lambda: SimpleNamespace(delete=lambda *args: (_ for _ in ()).throw(httpx.ConnectError("offline"))))
    jobs.schedule_delete(db, document)
    deleting.run()
    row = db.get("documents", document["id"])
    assert row["status"] == "deleting" and row["error"] and not jobs.activities
    monkeypatch.setattr(jobs, "vector_store", lambda: SimpleNamespace(delete=lambda *args: None))
    jobs.schedule_delete(db, row)
    deleting.run()
    assert document["id"] not in db.rows


def test_chat_source_check_does_not_wait_for_another_files_index(file_jobs, monkeypatch):
    db, document, _, _ = file_jobs
    jobs.schedule_index(db, document, b"original")
    ready = db.insert("documents", {"id": str(uuid4()), "status": "ready"})
    monkeypatch.setattr(routes, "vector_store", lambda: SimpleNamespace(needs_reindex=lambda *args: False))
    routes.ensure_current_sources(db, [ready["id"]])
    lock = jobs.file_lock(db.user.id, document["id"])
    with lock:
        with pytest.raises(HTTPException) as error:
            routes.ensure_current_sources(db, [document["id"]])
    assert error.value.status_code == 409


def test_cannot_restart_index_of_a_deleting_document(file_jobs):
    db, document, _, _ = file_jobs
    document["status"] = "deleting"
    with pytest.raises(HTTPException) as error:
        jobs.schedule_index(db, document, b"original")
    assert error.value.status_code == 409


@pytest.mark.parametrize("status", ["processing", "deleting"])
def test_interrupted_file_jobs_resume_after_restart(file_jobs, status):
    db, document, indexing, deleting = file_jobs
    db.update("documents", document["id"], {"status": status,
        "updated_at": (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()})
    rows = jobs.list_documents(db)
    assert rows[0]["activity"]["phase"] == "Queued"
    assert len(indexing.calls if status == "processing" else deleting.calls) == 1
    jobs.list_documents(db)
    assert len(indexing.calls if status == "processing" else deleting.calls) == 1


def test_opening_original_during_upload_gives_a_retryable_message():
    class UploadingDatabase:
        def storage(self, *args):
            raise HTTPException(404, "Original unavailable")
    with pytest.raises(HTTPException) as error:
        routes.original_file(UploadingDatabase(), {"storage_path": "alice/source.pdf", "status": "processing"})
    assert error.value.status_code == 409 and "still uploading" in error.value.detail


def test_embedding_batches_release_access_and_observe_cancellation(monkeypatch):
    batches, progress = [], []
    class Model:
        def encode(self, texts, **kwargs):
            batches.append(texts)
            return SimpleNamespace(tolist=lambda: [[1.0, 0.0] for text in texts])
    monkeypatch.setattr(vectors, "embedder", lambda: Model())
    monkeypatch.setattr(vectors, "settings", replace(vectors.settings, embedding_batch_size=2, qdrant_all_minilm_l6_v2=False))
    result = vectors.embed(["a", "b", "c"], progress=lambda done, total: progress.append((done, total)))
    assert len(result) == 3 and batches == [["a", "b"], ["c"]] and progress == [(2, 3), (3, 3)]
    assert not vectors._embedding_active
    with pytest.raises(InterruptedError):
        vectors.embed(["a", "b", "c"], cancelled=lambda: len(batches) >= 3)
    assert batches[-1] == ["a", "b"] and not vectors._embedding_active
