"""Answer-provider quota handling without live requests, real sleeps, or discarded user turns."""
import json
from dataclasses import replace

import httpx
import pytest
from fastapi import HTTPException

from app import generation


@pytest.fixture(autouse=True)
def provider(monkeypatch):
    monkeypatch.setattr(generation, "settings", replace(generation.settings, groq_key="test-key",
        generation_retries=2, generation_retry_wait=8, answer_input_tokens=3500, answer_max_tokens=1400))
    monkeypatch.setattr(generation, "_cooldowns", {})
    clock, sleeps = [100.0], []
    monkeypatch.setattr(generation.time, "monotonic", lambda: clock[0])
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(generation.time, "sleep", sleep)
    return clock, sleeps


def success(content="Supported answer [1].", **kwargs):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, **kwargs}]})


def test_short_rate_limit_retries_at_provider_deadline(provider, monkeypatch):
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        return httpx.Response(429, headers={"Retry-After": "2"}, json={"error": {"code": "rate_limit_exceeded"}}) if len(calls) == 1 else success()
    monkeypatch.setattr(generation.httpx, "post", post)
    assert generation.complete([{"role": "user", "content": "Question"}]) == "Supported answer [1]."
    assert len(calls) == 2 and provider[1] == [2]


def test_long_rate_limit_returns_countdown_and_stops_repeat_requests(provider, monkeypatch):
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        return httpx.Response(429, headers={"Retry-After": "61"}, json={"error": {"code": "rate_limit_exceeded"}})
    monkeypatch.setattr(generation.httpx, "post", post)
    for remaining in (61, 51):
        with pytest.raises(HTTPException) as error:
            generation.complete([{"role": "user", "content": "Question"}])
        assert error.value.status_code == 429 and error.value.headers["Retry-After"] == str(remaining)
        assert "Groq" in error.value.detail and "not been saved" in error.value.detail
        provider[0][0] += 10
    assert len(calls) == 1 and provider[1] == []


def test_token_reset_header_is_respected(provider, monkeypatch):
    monkeypatch.setattr(generation.httpx, "post", lambda *args, **kwargs: httpx.Response(429,
        headers={"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "1m12.5s"}, json={"error": {}}))
    with pytest.raises(HTTPException) as error:
        generation.complete([{"role": "user", "content": "Question"}])
    assert error.value.headers["Retry-After"] == "73" and provider[1] == []


def test_retry_message_duration_is_used(provider, monkeypatch):
    monkeypatch.setattr(generation.httpx, "post", lambda *args, **kwargs: httpx.Response(429,
        json={"error": {"message": "Rate limit reached. Please try again in 2m10.5s."}}))
    with pytest.raises(HTTPException) as error:
        generation.complete([{"role": "user", "content": "Question"}])
    assert error.value.headers["Retry-After"] == "131"


def test_too_large_request_is_not_retried_or_applied_to_other_questions(provider, monkeypatch):
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        return httpx.Response(429, json={"error": {"message": "Request too large: Limit 8000, Requested 9000"}}) if len(calls) == 1 else success()
    monkeypatch.setattr(generation.httpx, "post", post)
    with pytest.raises(HTTPException) as error:
        generation.complete([{"role": "user", "content": "Question"}])
    assert error.value.status_code == 413 and provider[1] == []
    assert generation.complete([{"role": "user", "content": "Smaller question"}]) == "Supported answer [1]."


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_transient_provider_failure_is_retried(provider, monkeypatch, status):
    responses = iter([httpx.Response(status, json={}), success()])
    monkeypatch.setattr(generation.httpx, "post", lambda *args, **kwargs: next(responses))
    assert generation.complete([{"role": "user", "content": "Question"}]) == "Supported answer [1]."
    assert provider[1] == [.5]


def test_network_retries_are_bounded(provider, monkeypatch):
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(generation.httpx, "post", post)
    with pytest.raises(HTTPException) as error:
        generation.complete([{"role": "user", "content": "Question"}])
    assert error.value.status_code == 503 and len(calls) == 3
    assert provider[1] == [.5, 1]


def test_json_mode_and_output_reservation_are_explicit(provider, monkeypatch):
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs["json"])
        return success('{"grounded":true,"blocks":[]}')
    monkeypatch.setattr(generation.httpx, "post", post)
    generation.complete([{"role": "user", "content": "Question"}], json_mode=True)
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["max_completion_tokens"] == 1400
    assert calls[0]["reasoning_effort"] == "low"


def test_truncated_json_retries_with_room_inside_total_budget(provider, monkeypatch):
    calls = []
    def post(*args, **kwargs):
        calls.append(json.loads(json.dumps(kwargs["json"])))
        return success('{"grounded":true', finish_reason="length") if len(calls) == 1 else success('{"grounded":false,"blocks":[]}')
    monkeypatch.setattr(generation.httpx, "post", post)
    messages = [{"role": "user", "content": "source " * 1400}]
    generation.complete(messages, json_mode=True)
    assert len(calls) == 2
    assert generation.estimate_messages(messages) + calls[-1]["max_completion_tokens"] <= 4900


def test_oversized_input_is_rejected_before_calling_provider(provider, monkeypatch):
    monkeypatch.setattr(generation.httpx, "post", lambda *args, **kwargs: pytest.fail("Oversized prompt sent to provider"))
    with pytest.raises(HTTPException) as error:
        generation.complete([{"role": "user", "content": "source " * 3000}])
    assert error.value.status_code == 400
