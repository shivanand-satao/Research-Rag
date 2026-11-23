"""Hosted inference requests, token limits, and model-safe retrieval."""
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from qdrant_client import QdrantClient

from app import processing, rag, vectors
from app.chunking import SemanticChunker
from app.config import QDRANT_MINILM_MODEL
from app.processing import Chunk
from app.schemas import ChatRequest


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)


@pytest.fixture
def cloud_mode(monkeypatch):
    settings = replace(vectors.settings, qdrant_url="https://cluster.example")
    monkeypatch.setattr(vectors, "settings", settings)
    monkeypatch.setattr(vectors, "_minilm_tokenizer", lambda: Tokenizer())
    return settings


@pytest.mark.parametrize("child_tokens,ceiling", [(360, 254), (128, 128)])
def test_minilm_token_budget_caps_large_chunks_and_preserves_smaller_limits(child_tokens, ceiling):
    settings = replace(vectors.settings, child_tokens=child_tokens)
    assert settings.embedding_model == QDRANT_MINILM_MODEL and settings.embedding_child_tokens == ceiling


def test_sdk_always_uses_cloud_inference(monkeypatch):
    calls = []
    monkeypatch.setattr(vectors, "settings", replace(vectors.settings,
        qdrant_url="https://cluster.example"))
    monkeypatch.setattr(vectors, "QdrantClient", lambda **kwargs: calls.append(kwargs) or object())
    vectors.VectorStore()
    assert calls[0]["cloud_inference"] is True and calls[0]["port"] == 443


def test_cloud_mode_requires_a_cloud_url(monkeypatch):
    monkeypatch.setattr(vectors, "settings", replace(vectors.settings, qdrant_url=""))
    with pytest.raises(ValueError, match="require QDRANT_URL"):
        vectors.VectorStore()


@pytest.fixture
def cloud_wire(cloud_mode, monkeypatch):
    requests, dimensions = [], []
    def handle(request):
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, body))
        if request.url.path.endswith("/exists"):
            result = {"exists": True}
        elif request.url.path.endswith("/query"):
            result = {"points": [{"id": 1, "version": 1, "score": .9, "payload": {
                "user_id": "alice", "document_id": "doc", "embedding_model": QDRANT_MINILM_MODEL,
                "text": "Evidence", "source": "report.pdf"}}]}
        else:
            result = {"operation_id": 1, "status": "completed"}
        return httpx.Response(200, json={"result": result, "status": "ok", "time": .001})
    # Exercise the installed SDK's real serialization while replacing only HTTP transport.
    client = QdrantClient(url=cloud_mode.qdrant_url, cloud_inference=True, check_compatibility=False)
    client._client.openapi_client.client._client.close()
    client._client.openapi_client.client._client = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(client, "_embed_models", lambda *args, **kwargs: pytest.fail("SDK attempted local inference"))
    store = vectors.VectorStore(client=client)
    monkeypatch.setattr(store, "ensure", lambda size: dimensions.append(size))
    yield store, requests, dimensions
    client.close()


def test_cloud_upsert_and_query_send_document_text_to_qdrant(cloud_wire):
    store, requests, dimensions = cloud_wire
    store.index("alice", "doc", "report.pdf", [Chunk("The detector reached 27 millivolts.", page=4)])
    assert dimensions == [384]
    upsert = next(body for method, path, body in requests if method == "PUT" and path.endswith("/points"))
    point = upsert["points"][0]
    assert point["vector"] == {"text": "The detector reached 27 millivolts.", "model": QDRANT_MINILM_MODEL}
    assert point["payload"]["embedding_model"] == QDRANT_MINILM_MODEL
    assert point["payload"]["page"] == 4
    assert store.search("What was the sensitivity?", "alice", ["doc"], "vector", 40)[0]["text"] == "Evidence"
    query = next(body for method, path, body in requests if path.endswith("/query"))
    assert query["limit"] == 40
    assert query["query"]["nearest"] == {"text": "What was the sensitivity?", "model": QDRANT_MINILM_MODEL}
    assert {condition["key"] for condition in query["filter"]["must"]} == {"user_id", "document_id", "embedding_model"}


def test_cancelled_cloud_upsert_keeps_the_previous_revision(cloud_wire):
    store, requests, _ = cloud_wire
    def cancelled():
        return any(method == "PUT" and path.endswith("/points") for method, path, body in requests)
    with pytest.raises(InterruptedError):
        store.index("alice", "doc", "report.pdf", [Chunk(f"Evidence {index}") for index in range(65)], cancelled=cancelled)
    assert len([request for request in requests if request[0] == "PUT"]) == 1
    assert not any(path.endswith("/delete") for method, path, body in requests)
    assert store.verified == {}


def test_cloud_query_respects_256_token_model_window(cloud_wire):
    store, requests, _ = cloud_wire
    store.search("detector " * 500, "alice", ["doc"], "vector", 8)
    query = next(body for method, path, body in requests if path.endswith("/query"))
    assert len(query["query"]["nearest"]["text"].split()) == 254
    assert query["query"]["nearest"]["model"] == QDRANT_MINILM_MODEL


def test_legacy_models_require_reindex_even_with_identical_dimensions(qdrant_memory):
    store = vectors.VectorStore(client=qdrant_memory)
    store.index("alice", "doc", "report.txt", [Chunk("detector sensitivity evidence")])
    points, _ = qdrant_memory.scroll(store.collection, limit=1)
    qdrant_memory.set_payload(store.collection, {"embedding_model": "legacy-embedding-model"}, points=[points[0].id])
    current = vectors.VectorStore(client=qdrant_memory)
    assert current.needs_reindex("alice", "doc")
    assert current.search("detector", "alice", ["doc"], "text", 8) == []
    qdrant_memory.set_payload(store.collection, {"embedding_model": QDRANT_MINILM_MODEL}, points=[points[0].id])
    assert not current.needs_reindex("alice", "doc")
    assert current.search("detector", "alice", ["doc"], "text", 8)[0]["embedding_model"] == QDRANT_MINILM_MODEL


def test_cloud_chunks_respect_minilm_context_without_local_embeddings(cloud_mode, monkeypatch):
    monkeypatch.setattr(processing, "settings", cloud_mode)
    text = (("The calibrated detector measured repeatable sensitivity. " * 70) + "\n\n" +
        ("Bond yields influence financial market pricing. " * 65))
    chunks = processing.extract(text.encode(), "research.txt", tokenizer=Tokenizer())
    assert len(chunks) > 1 and all(len(chunk.text.split()) <= 254 for chunk in chunks)
    assert all(chunk.parent_text and chunk.parent_id for chunk in chunks)


def test_cloud_chunking_detects_vocabulary_changes_without_a_model():
    paragraphs = ["Cellular respiration biological ATP synthesis. " * 18,
        "Cellular respiration biological ATP synthesis. " * 10,
        "Financial bonds interest prices yield analysis. " * 15]
    chunker = SemanticChunker(tokenizer=Tokenizer(), child_tokens=254, minimum_tokens=50)
    chunks = chunker.chunk("\n\n".join(paragraphs))
    assert len(chunks) == 2 and "Financial" not in chunks[0].text
    assert chunks[0].parent_id == chunks[1].parent_id
    assert len(chunks[0].text) != len(chunks[1].text)


def test_cloud_answer_context_uses_minilm_tokenizer(cloud_mode, monkeypatch):
    monkeypatch.setattr(rag, "settings", replace(rag.settings, rerank=False))
    child = {"id": "hit", "document_id": "doc", "source": "report.pdf", "page": 2,
        "text": "Sensitivity reached 27 millivolts.", "parent_text": "A calibrated detector. Sensitivity reached 27 millivolts."}
    monkeypatch.setattr(rag, "vector_store", lambda: SimpleNamespace(search=lambda *args: [child]))
    monkeypatch.setattr(rag, "complete", lambda *args, **kwargs: "Sensitivity reached 27 millivolts [1].")
    request = ChatRequest(request_id="12345678-1234-1234-1234-123456789012", content="What sensitivity was measured?")
    assert rag.answer_question(request, "alice", ["doc"], [])["grounded"]
