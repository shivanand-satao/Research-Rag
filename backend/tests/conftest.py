"""Qdrant storage checks with hosted embedding responses substituted at the client boundary."""
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient, models

from app import vectors
from app.config import QDRANT_MINILM_MODEL


@pytest.fixture
def qdrant_memory(monkeypatch):
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split(),
        decode=lambda tokens, **kwargs: " ".join(tokens))
    monkeypatch.setattr(vectors, "_minilm_tokenizer", lambda: tokenizer)
    client = QdrantClient(":memory:")
    upsert, query_points = client.upsert, client.query_points
    vector = [1.0] + [0.0] * 383

    def hosted_upsert(collection_name, points, **kwargs):
        for point in points:
            assert isinstance(point.vector, models.Document)
            assert point.vector.model == QDRANT_MINILM_MODEL
        numeric_points = [point.model_copy(update={"vector": vector}) for point in points]
        return upsert(collection_name, numeric_points, **kwargs)

    def hosted_query(collection_name, query, **kwargs):
        assert isinstance(query, models.Document)
        assert query.model == QDRANT_MINILM_MODEL
        return query_points(collection_name, query=vector, **kwargs)

    monkeypatch.setattr(client, "upsert", hosted_upsert)
    monkeypatch.setattr(client, "query_points", hosted_query)
    yield client
    client.close()
