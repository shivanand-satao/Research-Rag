from dataclasses import replace
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from app import rag, generation
from app.schemas import ChatRequest


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)


@pytest.fixture
def retrieval(monkeypatch):
    child = {"id": "child", "document_id": "doc", "source": "experiment.pdf", "page": 4,
        "text": "Sensitivity improved to 27 millivolts per unit.", "parent_id": "parent",
        "parent_text": "The baseline was 18 millivolts per unit. Sensitivity improved to 27 millivolts per unit. The measurement used a calibrated titanium detector.", "score": .8}
    queries = []
    class Store:
        def search(self, query, user, documents, mode, limit):
            queries.append((query, user, documents, limit))
            return [child]
    monkeypatch.setattr(rag, "vector_store", lambda: Store())
    monkeypatch.setattr(rag, "embedding_tokenizer", lambda: Tokenizer())
    monkeypatch.setattr(rag, "settings", replace(rag.settings, rerank=False))
    return child, queries


@pytest.mark.parametrize("top_k", [8, 16])
@pytest.mark.parametrize("scope", ["auto", "focused", "all_documents"])
def test_answer_uses_parent_context_and_fixed_candidate_pool(retrieval, monkeypatch, top_k, scope):
    _, queries = retrieval
    prompts = []
    def completion(messages, **kwargs):
        prompts.append(messages)
        return "Sensitivity improved from 18 to 27 millivolts per unit [1]."
    monkeypatch.setattr(rag, "complete", completion)
    result = rag.answer_question(ChatRequest(request_id="12345678-1234-1234-1234-123456789012", content="What was the improved sensitivity?", top_k=top_k, scope=scope), "alice", ["doc", "other"], [])
    assert result["grounded"]
    assert "baseline was 18" in prompts[0][-1]["content"]
    assert len(queries) == 1
    assert queries[0][2:] == (["doc", "other"], 40)
    assert result["citations"][0]["page"] == 4


@pytest.mark.parametrize("followup", [False, True])
def test_reranker_receives_40_candidates_before_selecting_8_contexts(retrieval, monkeypatch, followup):
    child, queries = retrieval
    documents = ["doc", *[f"doc-{index}" for index in range(1, 6)]]
    prompts, reranked = [], []
    class Store:
        def search(self, query, user, selected, mode, limit):
            queries.append((query, user, selected, limit))
            return [{**child, "id": f"{len(queries)}-{index}", "parent_id": f"{len(queries)}-{index}",
                "document_id": documents[index % len(documents)],
                "text": f"Sensitivity measurement {len(queries)}-{index}.",
                "parent_text": f"Sensitivity measurement {len(queries)}-{index}."} for index in range(40)]
    def predict(pairs, **kwargs):
        reranked.extend(text for _, text in pairs)
        # Make the last candidate most relevant to exercise selection after reranking.
        return list(range(len(pairs)))
    monkeypatch.setattr(rag, "vector_store", lambda: Store())
    monkeypatch.setattr(rag, "settings", replace(rag.settings, rerank=True))
    monkeypatch.setattr(rag, "reranker", lambda: SimpleNamespace(predict=predict))
    monkeypatch.setattr(rag, "complete", lambda messages, **kwargs: prompts.append(messages) or "The sensitivity was measured [1].")
    history = [{"role": "user", "content": "Tell me about the detector"}] if followup else []
    request = ChatRequest(request_id="12345678-1234-1234-1234-123456789012", content="What was its sensitivity?", scope="focused")
    result = rag.answer_question(request, "alice", documents, history)
    assert result["grounded"]
    assert len(queries) == (2 if followup else 1)
    assert all(selected == documents and limit == 40 for _, _, selected, limit in queries)
    assert len(reranked) == 40
    context = json.loads(prompts[0][-1]["content"])["source_excerpts"]
    assert [excerpt["text"] for excerpt in context] == list(reversed(reranked[-8:]))


def test_missing_citation_is_repaired_instead_of_claiming_no_evidence(retrieval, monkeypatch):
    responses = iter(["The improved sensitivity was 27 millivolts per unit.", "The improved sensitivity was 27 millivolts per unit [1]."])
    monkeypatch.setattr(rag, "complete", lambda *args, **kwargs: next(responses))
    result = rag.answer_question(ChatRequest(request_id="12345678-1234-1234-1234-123456789012", content="What was the improved sensitivity?"), "alice", ["doc"], [])
    assert result["grounded"] and "27" in result["content"]
    assert result["content"] != rag.INSUFFICIENT


def test_bad_citation_retry_reports_generation_error_not_missing_pdf_content(retrieval, monkeypatch):
    monkeypatch.setattr(rag, "complete", lambda *args, **kwargs: "A badly cited answer [99].")
    with pytest.raises(HTTPException, match="valid source references"):
        rag.answer_question(ChatRequest(request_id="12345678-1234-1234-1234-123456789012", content="What was the improved sensitivity?"), "alice", ["doc"], [])


def test_query_rewrite_failure_keeps_original_query(retrieval, monkeypatch):
    _, queries = retrieval
    def completion(messages, **kwargs):
        if "Rewrite" in messages[0]["content"]:
            raise HTTPException(503, "Rewrite provider unavailable")
        return "It was 27 millivolts per unit [1]."
    monkeypatch.setattr(rag, "complete", completion)
    result = rag.answer_question(ChatRequest(request_id="12345678-1234-1234-1234-123456789012", content="What was its sensitivity?"), "alice", ["doc"], [{"role": "user", "content": "Tell me about the detector"}])
    assert result["grounded"] and queries[0][0] == "What was its sensitivity?"


def test_compound_citation_formats_are_supported():
    assert rag.normalize_citations("Supported [1, 2] and 【3】 and [Source 4].") == "Supported [1][2] and [3] and [4]."


def test_empty_reasoning_completion_retries_with_a_larger_budget(monkeypatch):
    calls = []
    def post(*args, **kwargs):
        calls.append(dict(kwargs["json"]))
        content = "" if len(calls) == 1 else "The improved sensitivity was 27 [1]."
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
    monkeypatch.setattr(generation.httpx, "post", post)
    monkeypatch.setattr(generation, "settings", replace(generation.settings, groq_key="test"))
    assert "27" in rag.complete([{"role": "user", "content": "Question"}], max_tokens=1000)
    assert calls[1]["max_completion_tokens"] == 2000


def test_context_deduplicates_parents_and_centers_the_relevant_passage(retrieval, monkeypatch):
    child, _ = retrieval
    monkeypatch.setattr(rag, "settings", replace(rag.settings, context_tokens=200, parent_tokens=200))
    child["parent_text"] = "context " * 400 + child["text"] + " trailing " * 400
    result = rag.expand_context([child, {**child, "id": "other-child"}], ["doc"], 8)
    assert len(result) == 1 and "27 millivolts" in result[0]["text"]
    assert len(result[0]["text"].split()) <= 200


def test_structured_answer_renders_model_source_ids_without_a_repair_call(retrieval, monkeypatch):
    calls = []
    def completion(messages, **kwargs):
        calls.append((messages, kwargs))
        return json.dumps({"grounded": True, "blocks": [{"text": "The detector's sensitivity increased to 27 millivolts.", "source_ids": [1]}]})
    monkeypatch.setattr(rag, "complete", completion)
    result = rag.answer_question(ChatRequest(request_id="12345678-1234-1234-1234-123456789012", content="What was the sensitivity?"), "alice", ["doc"], [])
    assert result["content"].endswith("[1]") and result["grounded"]
    assert result["citations"][0]["document_id"] == "doc" and result["citations"][0]["page"] == 4
    assert len(calls) == 1 and calls[0][1]["json_mode"]


@pytest.mark.parametrize("source_ids", [[99], [0], [True], [], ["other-source"]])
def test_structured_answer_does_not_fabricate_source_ids(retrieval, source_ids):
    child, _ = retrieval
    with pytest.raises(ValueError):
        rag.parse_answer(json.dumps({"grounded": True, "blocks": [{"text": "An explanation.", "source_ids": source_ids}]}), [child])


def test_source_id_format_variants_and_unsupported_answer_are_handled(retrieval):
    child, _ = retrieval
    text, citations, grounded = rag.parse_answer(json.dumps({"grounded": True, "blocks": [{"text": "Supported explanation.", "source_ids": ["source_1", "S1"]}]}), [child])
    assert text == "Supported explanation. [1]" and len(citations) == 1 and grounded
    result = rag.parse_answer(json.dumps({"grounded": False, "blocks": [{"text": "The excerpt does not report costs.", "source_ids": []}]}), [child])
    assert result == ("The excerpt does not report costs.", [], False)


def test_structured_blocks_preserve_distinct_source_pages(retrieval):
    child, _ = retrieval
    other = {**child, "document_id": "other", "page": 9}
    text, citations, grounded = rag.parse_answer(json.dumps({"grounded": True, "blocks": [
        {"text": "First finding.", "source_ids": [1]}, {"text": "Second finding.", "source_ids": [2]}]}), [child, other])
    assert text == "First finding. [1]\n\nSecond finding. [2]"
    assert [citation["page"] for citation in citations] == [4, 9]


def test_followup_search_context_needs_only_one_generation_request(retrieval, monkeypatch):
    _, queries = retrieval
    calls = []
    monkeypatch.setattr(rag, "complete", lambda *args, **kwargs: calls.append(args) or 'Sensitivity reached 27 [1].')
    rag.answer_question(ChatRequest(request_id="12345678-1234-1234-1234-123456789012", content="What was its sensitivity?"), "alice", ["doc"], [{"role": "user", "content": "Tell me about the titanium detector"}])
    assert len(calls) == 1 and any("titanium detector" in query[0] for query in queries)


def test_large_parent_context_and_history_fit_answer_budget(retrieval, monkeypatch):
    child, _ = retrieval
    child["parent_text"] = "context " * 4000 + child["text"] + " trailing " * 4000
    prompts = []
    monkeypatch.setattr(rag, "complete", lambda messages, **kwargs: prompts.append(messages) or 'Sensitivity reached 27 [1].')
    rag.answer_question(ChatRequest(request_id="12345678-1234-1234-1234-123456789012", content="What was the improved sensitivity?"), "alice", ["doc"], [{"role": "user", "content": "conversation " * 4000}])
    assert generation.estimate_messages(prompts[0]) <= rag.settings.answer_input_tokens
    assert "27 millivolts" in prompts[0][-1]["content"]
