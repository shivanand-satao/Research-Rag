import pytest
from dataclasses import replace
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.main import app
from app.processing import Chunk
from app.rag import INSUFFICIENT, validate_citations
from app import vectors, routes


@pytest.fixture(autouse=True)
def local_embedding_mode(monkeypatch):
    # Tenant-isolation tests run in local Qdrant regardless of the developer's .env.
    monkeypatch.setattr(vectors, "settings", replace(vectors.settings, qdrant_all_minilm_l6_v2=False))


@pytest.mark.parametrize("path", ["/api/documents", "/api/conversations"])
def test_workspace_requires_a_verified_session(path):
    response = TestClient(app).get(path)
    assert response.status_code == 401


def test_retrieval_never_searches_without_owner_and_documents():
    with pytest.raises(ValueError):
        vectors.scoped_filter("", ["doc-a"])
    with pytest.raises(ValueError):
        vectors.scoped_filter("alice", [])


@pytest.mark.parametrize("mode", ["vector", "text", "hybrid"])
def test_search_and_delete_are_isolated_by_owner_and_document(monkeypatch, mode):
    monkeypatch.setattr(vectors, "embed", lambda texts, **kwargs: [[1.0, 0.0, 0.0] for text in texts])
    store = vectors.VectorStore(client=QdrantClient(":memory:"))
    store.index("alice", "shared-id", "alice.txt", [Chunk("orchid evidence for Alice")])
    store.index("bob", "shared-id", "bob.txt", [Chunk("orchid confidential evidence for Bob")])
    store.index("alice", "other-id", "other.txt", [Chunk("orchid unrelated selected scope")])
    results = store.search("orchid", "alice", ["shared-id"], mode, 6)
    assert len(results) == 1
    assert results[0]["source"] == "alice.txt"
    store.delete("alice", "shared-id")
    assert store.search("orchid", "alice", ["shared-id"], mode, 6) == []
    assert store.search("orchid", "bob", ["shared-id"], mode, 6)[0]["source"] == "bob.txt"
    store.client.close()


def test_invented_and_missing_citations_are_rejected():
    chunks = [{"document_id": "doc", "source": "source.pdf", "text": "Evidence", "page": 2}]
    assert validate_citations("Unsupported [7]", chunks) == (INSUFFICIENT, [], False)
    assert validate_citations("Unsupported with no citation", chunks) == (INSUFFICIENT, [], False)
    text, citations, grounded = validate_citations("Supported [1]", chunks)
    assert grounded and text == "Supported [1]"
    assert citations[0]["page"] == 2


def test_public_config_does_not_expose_backend_secrets():
    response = TestClient(app).get("/api/config")
    assert response.status_code == 200
    assert set(response.json()) == {"supabase_url", "supabase_key", "redirect_url"}


def test_service_key_is_never_published(monkeypatch):
    monkeypatch.setattr(routes, "settings", replace(routes.settings, supabase_key="sb_secret_test"))
    response = TestClient(app).get("/api/config")
    assert response.status_code == 503
    assert "sb_secret_test" not in response.text


def test_reindex_replaces_old_points_without_duplication(monkeypatch):
    monkeypatch.setattr(vectors, "embed", lambda texts, **kwargs: [[1.0, 0.0, 0.0] for text in texts])
    store = vectors.VectorStore(client=QdrantClient(":memory:"))
    store.index("alice", "document", "source.txt", [Chunk("Old content"), Chunk("Other old content")])
    store.index("alice", "document", "source.txt", [Chunk("Updated content and context")])
    hits = store.search("Updated", "alice", ["document"], "text", 40)
    assert len(hits) == 1
    assert store.client.count(store.collection).count == 1
    assert not store.needs_reindex("alice", "document")
    store.client.close()

