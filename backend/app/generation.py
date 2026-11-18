"""Budgeted answer-provider requests with bounded retries and shared rate-limit cooldowns."""
import logging
import math
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
from fastapi import HTTPException

from .config import settings

logger = logging.getLogger(__name__)
_cooldowns: dict[tuple[str, str], float] = {}
_cooldown_lock = threading.Lock()


def estimate_tokens(text: str) -> int:
    # Conservative UTF-8 estimate; exact usage comes from the answer provider's response.
    return math.ceil(len(text.encode("utf-8")) / 3)


def estimate_messages(messages: list[dict]) -> int:
    return 16 + sum(12 + estimate_tokens(message["content"]) for message in messages)


def _duration(value: str | None) -> float | None:
    if not value:
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except ValueError:
        pass
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", value.lower())
    if parts:
        return sum(float(number) * {"ms": .001, "s": 1, "m": 60, "h": 3600}[unit] for number, unit in parts)
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(0, (date - datetime.now(timezone.utc)).total_seconds())
    except (ValueError, TypeError, OverflowError):
        return None


def _retry_delay(response: httpx.Response, error: dict, attempt: int) -> float:
    delay = _duration(response.headers.get("retry-after"))
    if delay is not None:
        return max(.1, delay)
    waits = []
    for resource in ("tokens", "requests"):
        if response.headers.get(f"x-ratelimit-remaining-{resource}") == "0":
            reset = _duration(response.headers.get(f"x-ratelimit-reset-{resource}"))
            if reset is not None:
                waits.append(reset)
    if waits:
        return max(.1, max(waits))
    match = re.search(r"try again in ((?:\d+(?:\.\d+)?\s*(?:ms|s|m|h)\s*)+)", str(error.get("message", "")), re.I)
    delay = _duration(match[1]) if match else None
    return max(.1, delay) if delay is not None else min(2 ** attempt, 8)


def _rate_error(delay: float) -> HTTPException:
    seconds = max(1, math.ceil(delay))
    return HTTPException(429,
        f"Groq's answer-generation rate limit was reached. Try again in {seconds} seconds. Your question has not been saved.",
        headers={"Retry-After": str(seconds)})


def complete(messages: list[dict], max_tokens: int | None = None, *, json_mode=False) -> str:
    if not settings.groq_key:
        raise HTTPException(503, "Add GROQ_API_KEY to backend/.env to generate research answers.")
    requested = settings.answer_max_tokens if max_tokens is None else max_tokens
    input_estimate = estimate_messages(messages)
    if input_estimate > settings.answer_input_tokens:
        raise HTTPException(400, "The question and source context exceed the answer input budget. Shorten the question or reduce the selected sources.")
    reservation = settings.answer_input_tokens + settings.answer_max_tokens
    output_ceiling = max(128, reservation - input_estimate)
    body = {"model": settings.generation_model, "messages": messages,
        "max_completion_tokens": min(requested, output_ceiling)}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if "gpt-oss" in settings.generation_model:
        body.update({"reasoning_effort": "low", "include_reasoning": False})
    key = (settings.groq_url, settings.generation_model)
    retries, empty_retry = 0, False
    while True:
        with _cooldown_lock:
            remaining = _cooldowns.get(key, 0) - time.monotonic()
        if remaining > 0:
            raise _rate_error(remaining)
        try:
            response = httpx.post(f"{settings.groq_url}/chat/completions", timeout=120,
                headers={"Authorization": f"Bearer {settings.groq_key}", "User-Agent": "research-rag/1.0"}, json=body)
        except httpx.RequestError as exc:
            if retries < settings.generation_retries:
                time.sleep(min(.5 * 2 ** retries, settings.generation_retry_wait))
                retries += 1
                continue
            raise HTTPException(503, "Groq's answer service is unavailable. Your question has not been saved; please retry.") from exc
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not response.is_success:
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            error = error if isinstance(error, dict) else {}
            logger.warning("Answer provider returned HTTP %s", response.status_code)
            if response.status_code == 429:
                if "request too large" in str(error.get("message", "")).lower():
                    raise HTTPException(413, "The answer request exceeds Groq's token allowance. Lower RESEARCH_ANSWER_INPUT_TOKENS or select fewer sources; your question has not been saved.")
                delay = _retry_delay(response, error, retries)
                with _cooldown_lock:
                    deadline = time.monotonic() + delay
                    _cooldowns[key] = max(_cooldowns.get(key, 0), deadline)
                if retries < settings.generation_retries and delay <= settings.generation_retry_wait:
                    time.sleep(delay)
                    retries += 1
                    continue
                raise _rate_error(delay)
            if response.status_code in {500, 502, 503, 504} and retries < settings.generation_retries:
                time.sleep(min(.5 * 2 ** retries, settings.generation_retry_wait))
                retries += 1
                continue
            if response.status_code in {401, 403}:
                raise HTTPException(503, "Groq denied access to answer generation. Check GROQ_API_KEY and model access in the backend.")
            raise HTTPException(503, f"Groq's answer service could not complete this request (HTTP {response.status_code}). Your question has not been saved; please retry.")
        try:
            choice = payload["choices"][0]
            text = choice["message"].get("content") or ""
            if not isinstance(text, str):
                raise TypeError("Nontext answer")
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            usage = payload.get("usage", {})
            logger.info("Answer usage: prompt=%s completion=%s finish=%s",
                usage.get("prompt_tokens"), usage.get("completion_tokens"), choice.get("finish_reason"))
            truncated_json = json_mode and choice.get("finish_reason") == "length"
            if text and not truncated_json:
                return text
            if not empty_retry:
                # Increase room for reasoning once, within the total request reservation.
                body["max_completion_tokens"] = min(body["max_completion_tokens"] * 2, output_ceiling)
                empty_retry = True
                continue
            if text:
                return text
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise HTTPException(503, "Groq returned an invalid answer response. Your question has not been saved; please retry.") from exc
        raise HTTPException(503, "Groq returned an empty answer after retrying. Your question has not been saved; please retry.")
