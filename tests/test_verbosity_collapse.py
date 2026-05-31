"""Tests for CP-V Provider Verbosity Collapse (Task 13e).

Verifies:
- `raw` flag replaces `debug` + `raw_debug` config flags
- `llm:request` and `llm:response` events get optional `raw` field instead of tiered emissions
- No `llm:request:debug`, `llm:request:raw`, `llm:response:debug`, `llm:response:raw` events emitted
- Both non-streaming and streaming paths follow the collapsed pattern
- `_truncate_values` helper is removed (only used for debug tier)
"""

from unittest.mock import AsyncMock, patch

import pytest

# (OllamaProvider accessed via make_provider fixture from conftest.py)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _fake_stream(content: str = "hello"):
    """Minimal async generator simulating ollama streaming response."""
    yield {"message": {"content": content}, "done": False}
    yield {
        "message": {"content": ""},
        "done": True,
        "prompt_eval_count": 5,
        "eval_count": 2,
        "model": "llama3.2:3b",
    }


# ---------------------------------------------------------------------------
# Config flag tests
# ---------------------------------------------------------------------------


def test_raw_flag_attribute_exists_on_provider(make_provider):
    """OllamaProvider should have a `raw` attribute from config."""
    provider = make_provider(raw=True)
    assert hasattr(provider, "raw"), "Provider should have 'raw' attribute"
    assert provider.raw is True  # type: ignore[attr-defined]


def test_raw_flag_defaults_to_false(make_provider):
    """OllamaProvider.raw should default to False."""
    provider = make_provider()
    assert hasattr(provider, "raw"), "Provider should have 'raw' attribute"
    assert provider.raw is False  # type: ignore[attr-defined]


def test_debug_flag_removed(make_provider):
    """Provider should NOT have a `debug` attribute (replaced by `raw`)."""
    provider = make_provider()
    assert not hasattr(provider, "debug"), (
        "Provider should not have 'debug' attribute — replaced by 'raw'"
    )


def test_raw_debug_flag_removed(make_provider):
    """Provider should NOT have a `raw_debug` attribute (replaced by `raw`)."""
    provider = make_provider()
    assert not hasattr(provider, "raw_debug"), (
        "Provider should not have 'raw_debug' attribute — replaced by 'raw'"
    )


def test_truncate_values_removed():
    """Module-level `_truncate_values` function should be removed (only used for debug tier)."""
    import amplifier_module_provider_ollama as mod

    assert not hasattr(mod, "_truncate_values"), (
        "_truncate_values should be removed — it was only used for the debug tier"
    )


# ---------------------------------------------------------------------------
# Non-streaming: llm:request event tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_request_event_emitted_without_raw_field_by_default(
    make_provider, simple_request, mock_response
):
    """llm:request event should be emitted without a `raw` field by default (non-streaming)."""
    provider = make_provider()
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    request_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:request"
    ]
    assert len(request_events) >= 1, "llm:request event should be emitted"
    assert "raw" not in request_events[0], (
        "llm:request event should NOT have 'raw' field when raw=False"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_request_event_has_raw_field_when_raw_true(
    make_provider, simple_request, mock_response
):
    """llm:request event should include a `raw` field when raw=True (non-streaming)."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    request_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:request"
    ]
    assert len(request_events) >= 1, "llm:request event should be emitted"
    assert "raw" in request_events[0], (
        "llm:request event should have 'raw' field when raw=True"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_request_debug_event_never_emitted(
    make_provider, simple_request, mock_response
):
    """llm:request:debug event should NEVER be emitted (tiered events removed)."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    debug_events = [
        name
        for name, _ in provider.coordinator.hooks.events
        if name == "llm:request:debug"
    ]
    assert len(debug_events) == 0, (
        "llm:request:debug event should NEVER be emitted (collapsed pattern)"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_request_raw_event_never_emitted(
    make_provider, simple_request, mock_response
):
    """llm:request:raw event should NEVER be emitted (tiered events removed)."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    raw_events = [
        name
        for name, _ in provider.coordinator.hooks.events
        if name == "llm:request:raw"
    ]
    assert len(raw_events) == 0, (
        "llm:request:raw event should NEVER be emitted (collapsed pattern)"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_request_base_payload_fields_present(
    make_provider, simple_request, mock_response
):
    """llm:request event should still contain the expected base fields (non-streaming)."""
    provider = make_provider()
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    request_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:request"
    ]
    assert len(request_events) >= 1
    payload = request_events[0]
    assert payload["provider"] == "ollama"
    assert "model" in payload
    assert "message_count" in payload


# ---------------------------------------------------------------------------
# Non-streaming: llm:response event tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_response_event_emitted_without_raw_field_by_default(
    make_provider, simple_request, mock_response
):
    """llm:response event should be emitted without a `raw` field by default (non-streaming)."""
    provider = make_provider()
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1, "llm:response (ok) event should be emitted"
    assert "raw" not in response_events[0], (
        "llm:response event should NOT have 'raw' field when raw=False"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_response_event_has_raw_field_when_raw_true(
    make_provider, simple_request, mock_response
):
    """llm:response event should include a `raw` field when raw=True (non-streaming)."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1, "llm:response (ok) event should be emitted"
    assert "raw" in response_events[0], (
        "llm:response event should have 'raw' field when raw=True"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_response_debug_event_never_emitted(
    make_provider, simple_request, mock_response
):
    """llm:response:debug event should NEVER be emitted (tiered events removed)."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    debug_events = [
        name
        for name, _ in provider.coordinator.hooks.events
        if name == "llm:response:debug"
    ]
    assert len(debug_events) == 0, (
        "llm:response:debug event should NEVER be emitted (collapsed pattern)"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_response_raw_event_never_emitted(
    make_provider, simple_request, mock_response
):
    """llm:response:raw event should NEVER be emitted (tiered events removed)."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    raw_events = [
        name
        for name, _ in provider.coordinator.hooks.events
        if name == "llm:response:raw"
    ]
    assert len(raw_events) == 0, (
        "llm:response:raw event should NEVER be emitted (collapsed pattern)"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_response_base_payload_fields_present(
    make_provider, simple_request, mock_response
):
    """llm:response event should still contain the expected base fields (non-streaming)."""
    provider = make_provider()
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1
    payload = response_events[0]
    assert payload["provider"] == "ollama"
    assert "model" in payload
    assert payload["status"] == "ok"
    assert "duration_ms" in payload


# ---------------------------------------------------------------------------
# Non-streaming: event count sanity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_only_two_base_llm_events_non_streaming_raw_false(
    make_provider, simple_request, mock_response
):
    """With raw=False, exactly one llm:request and one llm:response (ok) emitted (non-streaming)."""
    provider = make_provider()
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    llm_events = [
        name for name, _ in provider.coordinator.hooks.events if name.startswith("llm:")
    ]
    assert llm_events.count("llm:request") == 1
    assert "llm:request:debug" not in llm_events
    assert "llm:request:raw" not in llm_events
    assert "llm:response:debug" not in llm_events
    assert "llm:response:raw" not in llm_events


@pytest.mark.asyncio(loop_scope="function")
async def test_only_two_base_llm_events_non_streaming_raw_true(
    make_provider, simple_request, mock_response
):
    """With raw=True, still one llm:request and one llm:response (with raw field, no tiered variants)."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    llm_events = [
        name for name, _ in provider.coordinator.hooks.events if name.startswith("llm:")
    ]
    assert llm_events.count("llm:request") == 1
    assert "llm:request:debug" not in llm_events
    assert "llm:request:raw" not in llm_events
    assert "llm:response:debug" not in llm_events
    assert "llm:response:raw" not in llm_events


# ---------------------------------------------------------------------------
# Streaming: llm:request event tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_streaming_llm_request_no_raw_field_by_default(
    make_provider, simple_request
):
    """llm:request event should be emitted without `raw` field by default (streaming)."""
    provider = make_provider()
    provider.client.chat = AsyncMock(return_value=_fake_stream())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(stream=True))

    request_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:request"
    ]
    assert len(request_events) >= 1, "llm:request event should be emitted"
    assert "raw" not in request_events[0], (
        "llm:request event should NOT have 'raw' field when raw=False (streaming)"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_streaming_llm_request_has_raw_field_when_raw_true(
    make_provider, simple_request
):
    """llm:request event should include a `raw` field when raw=True (streaming)."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=_fake_stream())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(stream=True))

    request_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:request"
    ]
    assert len(request_events) >= 1, "llm:request event should be emitted"
    assert "raw" in request_events[0], (
        "llm:request event should have 'raw' field when raw=True (streaming)"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_streaming_llm_request_debug_event_never_emitted(
    make_provider, simple_request
):
    """llm:request:debug event should NEVER be emitted in streaming path."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=_fake_stream())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(stream=True))

    debug_events = [
        name
        for name, _ in provider.coordinator.hooks.events
        if name == "llm:request:debug"
    ]
    assert len(debug_events) == 0, (
        "llm:request:debug should NEVER be emitted in streaming path"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_streaming_llm_request_raw_event_never_emitted(
    make_provider, simple_request
):
    """llm:request:raw event should NEVER be emitted in streaming path."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=_fake_stream())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(stream=True))

    raw_events = [
        name
        for name, _ in provider.coordinator.hooks.events
        if name == "llm:request:raw"
    ]
    assert len(raw_events) == 0, (
        "llm:request:raw should NEVER be emitted in streaming path"
    )


# ---------------------------------------------------------------------------
# Streaming: llm:response event tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_streaming_llm_response_no_raw_field_by_default(
    make_provider, simple_request
):
    """llm:response event should be emitted without `raw` field by default (streaming)."""
    provider = make_provider()
    provider.client.chat = AsyncMock(return_value=_fake_stream())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(stream=True))

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1, "llm:response (ok) event should be emitted"
    assert "raw" not in response_events[0], (
        "llm:response event should NOT have 'raw' field when raw=False (streaming)"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_streaming_llm_response_has_raw_field_when_raw_true(
    make_provider, simple_request
):
    """llm:response event should include a `raw` field when raw=True (streaming)."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=_fake_stream())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(stream=True))

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1, "llm:response (ok) event should be emitted"
    assert "raw" in response_events[0], (
        "llm:response event should have 'raw' field when raw=True (streaming)"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_streaming_llm_response_debug_event_never_emitted(
    make_provider, simple_request
):
    """llm:response:debug event should NEVER be emitted in streaming path."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=_fake_stream())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(stream=True))

    debug_events = [
        name
        for name, _ in provider.coordinator.hooks.events
        if name == "llm:response:debug"
    ]
    assert len(debug_events) == 0, (
        "llm:response:debug should NEVER be emitted in streaming path"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_streaming_llm_response_raw_event_never_emitted(
    make_provider, simple_request
):
    """llm:response:raw event should NEVER be emitted in streaming path."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=_fake_stream())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(stream=True))

    raw_events = [
        name
        for name, _ in provider.coordinator.hooks.events
        if name == "llm:response:raw"
    ]
    assert len(raw_events) == 0, (
        "llm:response:raw should NEVER be emitted in streaming path"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_streaming_no_tiered_events_raw_false(make_provider, simple_request):
    """With raw=False, no tiered llm events in streaming path."""
    provider = make_provider()
    provider.client.chat = AsyncMock(return_value=_fake_stream())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(stream=True))

    llm_events = [
        name for name, _ in provider.coordinator.hooks.events if name.startswith("llm:")
    ]
    assert "llm:request:debug" not in llm_events
    assert "llm:request:raw" not in llm_events
    assert "llm:response:debug" not in llm_events
    assert "llm:response:raw" not in llm_events


@pytest.mark.asyncio(loop_scope="function")
async def test_streaming_no_tiered_events_raw_true(make_provider, simple_request):
    """With raw=True, still no tiered llm events in streaming path (raw goes on base event)."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=_fake_stream())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(stream=True))

    llm_events = [
        name for name, _ in provider.coordinator.hooks.events if name.startswith("llm:")
    ]
    assert "llm:request:debug" not in llm_events
    assert "llm:request:raw" not in llm_events
    assert "llm:response:debug" not in llm_events
    assert "llm:response:raw" not in llm_events
