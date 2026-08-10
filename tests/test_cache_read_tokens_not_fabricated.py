"""Regression guard: Ollama must never report a fabricated cache_read_tokens.

Prior state (the bug this test guards against): `llm:response` event
emission guarded `cache_read_tokens` with
`if chat_response.usage.cache_read_tokens is not None:` at both the
non-streaming and streaming call sites, but `Usage(...)` was never
constructed with a `cache_read_tokens` argument at either usage-building
site. The field therefore always defaulted to `None` and the guard was
provably dead code -- it made the provider look instrumented for cache
telemetry when it never was.

Investigation finding (verified against the installed `ollama` SDK's actual
response type, not just documentation): `ollama._types.BaseGenerateResponse`
(SDK 0.6.1, the base class of `ChatResponse`) exposes only: model,
created_at, done, done_reason, total_duration, load_duration,
prompt_eval_count, prompt_eval_duration, eval_count, eval_duration. There is
no field distinguishing prompt tokens served from Ollama's local KV-cache
reuse (a process-internal llama.cpp implementation detail) from freshly
evaluated prompt tokens. Unlike Anthropic/OpenAI/Gemini, Ollama's chat API
provides no real number here at all.

Given that, the honest fix is to remove the dead guard (done in
amplifier_module_provider_ollama/__init__.py) rather than synthesize a value
Ollama doesn't report. This test locks in both halves: the dead branch stays
removed, and Usage.cache_read_tokens stays None (never fabricated as 0 or
derived from prompt_eval_count deltas).
"""

from unittest.mock import AsyncMock, patch

import pytest


def test_usage_metadata_type_has_no_cache_field():
    """Source-level confirmation: the installed `ollama` SDK's actual
    response type has no cache-related field at all. Pins the investigation
    finding itself, independent of how this provider reads it, so an SDK
    upgrade that *adds* real cache telemetry is caught here instead of the
    provider silently continuing to report None forever.
    """
    from ollama._types import BaseGenerateResponse

    field_names = set(BaseGenerateResponse.model_fields)
    cache_like_fields = {name for name in field_names if "cache" in name.lower()}

    assert cache_like_fields == set(), (
        "ollama SDK now exposes a cache-related field on BaseGenerateResponse: "
        f"{cache_like_fields}. The 'Ollama reports no cache metric' finding in "
        "amplifier_module_provider_ollama/__init__.py needs re-investigation -- "
        "if this is a real, measured value, it should be surfaced as "
        "Usage.cache_read_tokens instead of left unset."
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_non_streaming_cache_read_tokens_is_none(
    make_provider, simple_request, mock_response
):
    """Non-streaming complete(): Usage.cache_read_tokens must be None, and
    the llm:response event's usage dict must not contain a cache_read_tokens
    key (there is nothing real to report).
    """
    provider = make_provider()
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        response = await provider.complete(simple_request(metadata={"stream": False}))

    assert response.usage is not None
    assert response.usage.cache_read_tokens is None
    assert response.usage.cache_write_tokens is None

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1
    usage = response_events[0].get("usage", {})
    assert "cache_read_tokens" not in usage, (
        "llm:response usage dict must not contain cache_read_tokens -- Ollama "
        "reports no such metric, so a present key here (even set to a real-"
        "looking number) would be fabricated telemetry."
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_streaming_cache_read_tokens_is_none(
    make_provider, simple_request, mock_response
):
    """Streaming complete(): same guarantee as the non-streaming path --
    Usage.cache_read_tokens is None and the emitted event carries no
    cache_read_tokens key.
    """

    async def _fake_stream():
        chunk = mock_response()
        chunk["message"]["content"] = "ok"
        yield chunk

    provider = make_provider()
    provider.client.chat = AsyncMock(return_value=_fake_stream())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        response = await provider.complete(simple_request(metadata={"stream": True}))

    assert response.usage is not None
    assert response.usage.cache_read_tokens is None
    assert response.usage.cache_write_tokens is None

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1
    usage = response_events[0].get("usage", {})
    assert "cache_read_tokens" not in usage, (
        "Streaming llm:response usage dict must not contain cache_read_tokens "
        "-- Ollama reports no such metric."
    )
