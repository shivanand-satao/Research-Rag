"""Qdrant Cloud embeddings and tenant-filtered retrieval."""
import math
import hashlib
import json
import re
import threading
import time
from collections import Counter
from functools import lru_cache
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from .config import BACKEND, settings
from .processing import Chunk
from .chunking import CHUNKING_VERSION


@lru_cache(maxsize=1)
def _minilm_tokenizer():
    # Only tokenizer files are downloaded; model weights and inference stay in Qdrant.
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    path = hf_hub_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", filename="tokenizer.json",
        cache_dir=str(BACKEND / ".cache" / "tokenizers"))
    tokenizer = Tokenizer.from_file(path)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return _CloudTokenizer(tokenizer)


class _CloudTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)


def embedding_tokenizer():
    return _minilm_tokenizer()


def scoped_filter(user_id: str, document_ids: list[str]) -> models.Filter:
    if not user_id or not document_ids:
        raise ValueError("Retrieval requires an owner and explicit document scope")
    return models.Filter(must=[
        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
        models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids)),
    ])


def lexical_rank(query: str, records: list[dict], limit: int) -> list[dict]:
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or", "in", "on", "for", "with", "what", "which", "how", "does", "do", "can", "this", "that", "these", "those"}
    terms = lambda text: [term for term in re.findall(r"[\w-]+", text.lower()) if term not in stopwords]
    query_terms = set(terms(query))
    corpora = [terms(item["text"]) for item in records]
    frequency = Counter(term for corpus in corpora for term in set(corpus))
    average = sum(map(len, corpora)) / max(len(corpora), 1)
    ranked = []
    for item, corpus in zip(records, corpora):
        counts, score = Counter(corpus), 0.0
        for term in query_terms:
            count = counts[term]
            if count:
                idf = math.log(1 + (len(records) - frequency[term] + .5) / (frequency[term] + .5))
                score += idf * count * 2.5 / (count + 1.5 * (.25 + .75 * len(corpus) / max(average, 1)))
        if score > 0:
            ranked.append({**item, "score": score})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]


def fuse(rankings: list[list[dict]], limit: int) -> list[dict]:
    merged = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, 1):
            match = merged.setdefault(item["id"], {**item, "score": 0.0})
            match["score"] += 1 / (60 + position)
    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:limit]


class VectorStore:
    def __init__(self, client=None):
        if client is None and not settings.qdrant_url:
            raise ValueError("Qdrant Cloud embeddings require QDRANT_URL for a cluster with inference enabled.")
        url = urlparse(settings.qdrant_url)
        port = url.port or (443 if url.scheme == "https" else 6333)
        self.client = client or QdrantClient(url=settings.qdrant_url, port=port, api_key=settings.qdrant_key,
            cloud_inference=True, timeout=30)
        self.collection = settings.collection
        self.embedding_model = settings.embedding_model
        self.lock = threading.Lock()
        self.verified: dict[tuple[str, str], float] = {}

    def ensure(self, dimensions: int):
        with self.lock:
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(self.collection,
                    vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE))
            configured = self.client.get_collection(self.collection).config.params.vectors
            if not isinstance(configured, models.VectorParams) or configured.size != dimensions:
                raise ValueError("Embedding dimensions differ from this collection. Configure a new RESEARCH_QDRANT_COLLECTION.")
            self._ensure_payload_indexes()

    def _ensure_payload_indexes(self):
        schema = self.client.get_collection(self.collection).payload_schema or {}
        for field, kind in {"user_id": models.PayloadSchemaType.KEYWORD,
            "document_id": models.PayloadSchemaType.KEYWORD, "index_revision": models.PayloadSchemaType.KEYWORD,
            "embedding_model": models.PayloadSchemaType.KEYWORD, "chunking_version": models.PayloadSchemaType.INTEGER}.items():
            if field not in schema:
                self.client.create_payload_index(self.collection, field, kind, wait=True)

    def index(self, user_id: str, document_id: str, filename: str, chunks: list[Chunk], *, cancelled=None, progress=None):
        if not chunks:
            raise ValueError("Cannot index an empty document")
        if cancelled and cancelled():
            raise InterruptedError("File processing cancelled")
        self.ensure(384)
        revision = hashlib.sha256(json.dumps({"version": CHUNKING_VERSION, "model": self.embedding_model,
            "chunks": [(c.text, c.parent_text, c.page, c.section) for c in chunks]}, ensure_ascii=False).encode()).hexdigest()
        for start in range(0, len(chunks), 64):
            if cancelled and cancelled():
                raise InterruptedError("File processing cancelled")
            points = [models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{user_id}/{document_id}/{revision}/{index}")),
                vector=models.Document(text=chunk.text, model=self.embedding_model),
                payload={"user_id": user_id, "document_id": document_id, "source": filename,
                    "text": chunk.text, "page": chunk.page, "section": chunk.section, "chunk_index": index,
                    "parent_text": chunk.parent_text or chunk.text, "parent_id": chunk.parent_id or str(index),
                    "chunking_version": CHUNKING_VERSION, "index_revision": revision, "document_chunk_count": len(chunks),
                    "embedding_model": self.embedding_model})
                for index, chunk in enumerate(chunks[start:start + 64], start)]
            self.client.upsert(self.collection, points, wait=True)
            if progress:
                progress(min(start + 64, len(chunks)), len(chunks))
        # Replace a document only after all of its new points have been acknowledged.
        obsolete = scoped_filter(user_id, [document_id])
        obsolete.must_not = [models.FieldCondition(key="index_revision", match=models.MatchValue(value=revision))]
        self.client.delete(self.collection, points_selector=obsolete, wait=True)
        if len(self.verified) > 512:
            self.verified.clear()
        self.verified[(user_id, document_id)] = time.monotonic()

    def needs_reindex(self, user_id: str, document_id: str) -> bool:
        checked = self.verified.get((user_id, document_id), 0)
        if time.monotonic() - checked < 300:
            return False
        if not self.client.collection_exists(self.collection):
            return True
        # Older collections may predate the version/revision indexes required by strict mode.
        with self.lock:
            self._ensure_payload_indexes()
        points, _ = self.client.scroll(self.collection, scroll_filter=scoped_filter(user_id, [document_id]),
            limit=1, with_payload=True, with_vectors=False)
        if not points:
            return True
        payload = points[0].payload or {}
        expected = payload.get("document_chunk_count")
        if not expected or self.client.count(self.collection, count_filter=scoped_filter(user_id, [document_id]), exact=True).count != expected:
            return True
        outdated = scoped_filter(user_id, [document_id])
        outdated.must_not = [models.Filter(must=[
            models.FieldCondition(key="chunking_version", match=models.MatchValue(value=CHUNKING_VERSION)),
            models.FieldCondition(key="embedding_model", match=models.MatchValue(value=self.embedding_model)),
            models.FieldCondition(key="index_revision", match=models.MatchValue(value=payload.get("index_revision", ""))),
        ])]
        if self.client.count(self.collection, count_filter=outdated, exact=True).count:
            return True
        self.verified[(user_id, document_id)] = time.monotonic()
        return False

    def delete(self, user_id: str, document_id: str):
        self.verified.pop((user_id, document_id), None)
        if self.client.collection_exists(self.collection):
            self.client.delete(self.collection, points_selector=scoped_filter(user_id, [document_id]), wait=True)

    def search(self, query: str, user_id: str, document_ids: list[str], mode: str, limit: int) -> list[dict]:
        security_filter = scoped_filter(user_id, document_ids)
        # Exclude incompatible vectors left by older embedding models.
        security_filter.must.append(models.FieldCondition(key="embedding_model", match=models.MatchValue(value=self.embedding_model)))
        if not self.client.collection_exists(self.collection):
            return []
        dense, lexical = [], []
        if mode in {"vector", "hybrid"}:
            tokenizer = embedding_tokenizer()
            tokens = tokenizer.encode(query, add_special_tokens=False)
            if len(tokens) > 254:
                query_text = tokenizer.decode(tokens[:254], skip_special_tokens=True)
            else:
                query_text = query
            vector = models.Document(text=query_text, model=self.embedding_model)
            result = self.client.query_points(self.collection, query=vector, query_filter=security_filter,
                limit=limit, with_payload=True, search_params=models.SearchParams(hnsw_ef=128))
            dense = [{**hit.payload, "id": str(hit.id), "score": hit.score} for hit in result.points]
        if mode in {"text", "hybrid"}:
            records, offset = [], None
            while True:
                points, offset = self.client.scroll(self.collection, scroll_filter=security_filter,
                    limit=256, offset=offset, with_payload=True, with_vectors=False)
                records.extend({**point.payload, "id": str(point.id)} for point in points)
                # Bound memory rather than silently returning a partial lexical corpus.
                if len(records) > 30000:
                    raise ValueError("Text search supports up to 30,000 scoped chunks. Select fewer documents or use vector mode.")
                if offset is None:
                    break
            lexical = lexical_rank(query, records, limit)
        if mode == "hybrid":
            return fuse([dense, lexical], limit)
        return (dense if mode == "vector" else lexical)[:limit]


@lru_cache(maxsize=1)
def vector_store() -> VectorStore:
    return VectorStore()

