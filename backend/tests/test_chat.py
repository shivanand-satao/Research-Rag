from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.database import authenticate, Identity
from app.main import app
from app import routes


@pytest.fixture
def chat_client(monkeypatch):
    cid = str(uuid4())
    conversation = {"id": cid, "document_ids": [str(uuid4())], "title": "New research"}
    calls = []
    existing = []

    class FakeDatabase:
        def __init__(self, user):
            assert user.id == "alice"

        def get(self, table, item_id):
            if item_id != cid:
                raise HTTPException(404, "Conversation not found")
            return conversation

        def request(self, method, path, **kwargs):
            calls.append((method, path, kwargs))
            if method == "GET":
                return existing
            return {"conversation": conversation, "messages": [{"role": "user"}, {"role": "assistant"}]}

        def validate_documents(self, ids):
            assert ids == conversation["document_ids"]
            return ids

        def messages(self, item_id, **kwargs):
            assert item_id == cid
            return []

    monkeypatch.setattr(routes, "Database", FakeDatabase)
    monkeypatch.setattr(routes, "ensure_current_sources", lambda db, ids: None)
    app.dependency_overrides[authenticate] = lambda: Identity("alice", "test-token")
    yield TestClient(app), cid, calls, existing, conversation
    app.dependency_overrides.clear()


def test_turn_saves_both_messages_through_atomic_rpc(chat_client, monkeypatch):
    client, cid, calls, _, _ = chat_client
    monkeypatch.setattr(routes, "answer_question", lambda *args: {"content": "Evidence [1]", "citations": [], "grounded": True, "model": "test"})
    response = client.post(f"/api/conversations/{cid}/chat", json={"request_id": str(uuid4()), "content": "What does it say?"})
    assert response.status_code == 200
    writes = [call for call in calls if call[0] == "POST"]
    assert len(writes) == 1 and writes[0][1] == "rpc/research_save_turn"
    assert writes[0][2]["body"]["p_question"] == "What does it say?"
    assert writes[0][2]["body"]["p_answer"] == "Evidence [1]"


def test_generation_failure_does_not_write_partial_history(chat_client, monkeypatch):
    client, cid, calls, _, _ = chat_client
    def fail(*args):
        raise HTTPException(503, "Provider unavailable")
    monkeypatch.setattr(routes, "answer_question", fail)
    response = client.post(f"/api/conversations/{cid}/chat", json={"request_id": str(uuid4()), "content": "Explain"})
    assert response.status_code == 503
    assert not any(method == "POST" for method, _, _ in calls)


def test_provider_rate_limit_preserves_retry_after_without_saving_turn(chat_client, monkeypatch):
    client, cid, calls, _, _ = chat_client
    def fail(*args):
        raise HTTPException(429, "Groq rate limit. Retry in 60 seconds.", headers={"Retry-After": "60"})
    monkeypatch.setattr(routes, "answer_question", fail)
    response = client.post(f"/api/conversations/{cid}/chat", json={"request_id": str(uuid4()), "content": "Explain"})
    assert response.status_code == 429 and response.headers["retry-after"] == "60"
    assert not any(method == "POST" for method, _, _ in calls)


def test_retry_returns_saved_turn_without_generating_again(chat_client, monkeypatch):
    client, cid, calls, existing, _ = chat_client
    existing.extend([{"role": "user", "content": "Explain"}, {"role": "assistant", "content": "Saved answer"}])
    monkeypatch.setattr(routes, "answer_question", lambda *args: pytest.fail("Retry should not regenerate"))
    response = client.post(f"/api/conversations/{cid}/chat", json={"request_id": str(uuid4()), "content": "Explain"})
    assert response.status_code == 200 and len(response.json()["messages"]) == 2
    assert not any(method == "POST" for method, _, _ in calls)


def test_foreign_conversation_is_not_accessible(chat_client):
    client, _, _, _, _ = chat_client
    response = client.post(f"/api/conversations/{uuid4()}/chat", json={"request_id": str(uuid4()), "content": "Explain"})
    assert response.status_code == 404


def test_chat_rejects_empty_source_scope(chat_client):
    client, cid, _, _, conversation = chat_client
    conversation["document_ids"] = []
    response = client.post(f"/api/conversations/{cid}/chat", json={"request_id": str(uuid4()), "content": "Explain"})
    assert response.status_code == 400

