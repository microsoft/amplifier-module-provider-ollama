"""Tests for llm:response ordering fix — canonical usage keys.

Verifies:
- llm:response event is emitted AFTER _convert_to_chat_response()
- usage keys use canonical names: input_tokens, output_tokens (not input/output)
- usage values come from chat_response.usage fields
- streaming path (_complete_streaming) uses same canonical keys
"""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_response_uses_canonical_input_tokens_key(
    make_provider, simple_request, mock_response
):
    """llm:response usage dict must use 'input_tokens' key (not 'input')."""
    provider = make_provider()
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request())

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1, "llm:response (ok) event should be emitted"

    usage = response_events[0].get("usage", {})
    assert "input_tokens" in usage, (
        "llm:response usage must have 'input_tokens' key (canonical), got: "
        + str(list(usage.keys()))
    )
    assert "input" not in usage, (
        "llm:response usage must NOT have 'input' key (non-canonical)"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_response_uses_canonical_output_tokens_key(
    make_provider, simple_request, mock_response
):
    """llm:response usage dict must use 'output_tokens' key (not 'output')."""
    provider = make_provider()
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request())

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1, "llm:response (ok) event should be emitted"

    usage = response_events[0].get("usage", {})
    assert "output_tokens" in usage, (
        "llm:response usage must have 'output_tokens' key (canonical), got: "
        + str(list(usage.keys()))
    )
    assert "output" not in usage, (
        "llm:response usage must NOT have 'output' key (non-canonical)"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_response_usage_values_match_chat_response(
    make_provider, simple_request, mock_response
):
    """llm:response usage values must match the converted ChatResponse usage."""
    provider = make_provider()
    # mock_response returns prompt_eval_count=10, eval_count=5
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await provider.complete(simple_request())

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1, "llm:response (ok) event should be emitted"

    usage = response_events[0].get("usage", {})
    # Values should match chat_response.usage (prompt_eval_count=10, eval_count=5)
    assert usage.get("input_tokens") == result.usage.input_tokens, (
        "llm:response input_tokens must match chat_response.usage.input_tokens"
    )
    assert usage.get("output_tokens") == result.usage.output_tokens, (
        "llm:response output_tokens must match chat_response.usage.output_tokens"
    )


# ── Streaming path tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio(loop_scope="function")
async def test_streaming_llm_response_uses_canonical_input_tokens_key(
    make_provider, simple_request
):
    """Streaming llm:response usage must use 'input_tokens' key (not 'input')."""
    provider = make_provider()

    async def fake_stream():
        yield {"message": {"content": "hello"}, "done": False}
        yield {
            "message": {"content": ""},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
            "model": "llama3.2:3b",
        }

    provider.client.chat = AsyncMock(return_value=fake_stream())

    await provider.complete(simple_request(stream=True))

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1, "llm:response (ok) event should be emitted"

    usage = response_events[0].get("usage", {})
    assert "input_tokens" in usage, (
        "streaming llm:response usage must have 'input_tokens' key (canonical), got: "
        + str(list(usage.keys()))
    )
    assert "input" not in usage, (
        "streaming llm:response usage must NOT have 'input' key (non-canonical)"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_streaming_llm_response_uses_canonical_output_tokens_key(
    make_provider, simple_request
):
    """Streaming llm:response usage must use 'output_tokens' key (not 'output')."""
    provider = make_provider()

    async def fake_stream():
        yield {"message": {"content": "hello"}, "done": False}
        yield {
            "message": {"content": ""},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
            "model": "llama3.2:3b",
        }

    provider.client.chat = AsyncMock(return_value=fake_stream())

    await provider.complete(simple_request(stream=True))

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1, "llm:response (ok) event should be emitted"

    usage = response_events[0].get("usage", {})
    assert "output_tokens" in usage, (
        "streaming llm:response usage must have 'output_tokens' key (canonical), got: "
        + str(list(usage.keys()))
    )
    assert "output" not in usage, (
        "streaming llm:response usage must NOT have 'output' key (non-canonical)"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_streaming_llm_response_usage_values_match_chat_response(
    make_provider, simple_request
):
    """Streaming llm:response usage values must match _build_streaming_response usage."""
    provider = make_provider()

    async def fake_stream():
        yield {"message": {"content": "hello"}, "done": False}
        yield {
            "message": {"content": ""},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
            "model": "llama3.2:3b",
        }

    provider.client.chat = AsyncMock(return_value=fake_stream())

    result = await provider.complete(simple_request(stream=True))

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1, "llm:response (ok) event should be emitted"

    usage = response_events[0].get("usage", {})
    # Values must match the ChatResponse built by _build_streaming_response
    assert usage.get("input_tokens") == result.usage.input_tokens, (
        "streaming llm:response input_tokens must match chat_response.usage.input_tokens"
    )
    assert usage.get("output_tokens") == result.usage.output_tokens, (
        "streaming llm:response output_tokens must match chat_response.usage.output_tokens"
    )
