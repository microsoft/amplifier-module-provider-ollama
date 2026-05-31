"""Tests for the raw observability flag (CP-V verbosity collapse).

After Task 13e, the `debug` + `raw_debug` flags are replaced by a single `raw`
flag. Raw API I/O is exposed as an optional field on the base llm:request and
llm:response events — never as separate :raw or :debug sub-events.
"""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio(loop_scope="function")
async def test_no_tiered_raw_events_when_raw_false(
    make_provider, simple_request, mock_response
):
    """With raw=False (default), no :raw sub-events are ever emitted."""
    provider = make_provider(raw=False)
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    raw_events = [
        name for name, _ in provider.coordinator.hooks.events if name.endswith(":raw")
    ]
    assert raw_events == [], f"Expected no :raw events but got {raw_events}"


@pytest.mark.asyncio(loop_scope="function")
async def test_no_tiered_raw_events_when_raw_true(
    make_provider, simple_request, mock_response
):
    """With raw=True, raw data is on the base event — still no :raw sub-events."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    raw_events = [
        name for name, _ in provider.coordinator.hooks.events if name.endswith(":raw")
    ]
    assert raw_events == [], (
        f"Expected no :raw sub-events (raw goes on base event) but got {raw_events}"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_no_tiered_debug_events_when_raw_true(
    make_provider, simple_request, mock_response
):
    """With raw=True, no :debug sub-events are emitted either."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    debug_events = [
        name for name, _ in provider.coordinator.hooks.events if name.endswith(":debug")
    ]
    assert debug_events == [], f"Expected no :debug events but got {debug_events}"


@pytest.mark.asyncio(loop_scope="function")
async def test_raw_field_on_base_events_when_raw_true(
    make_provider, simple_request, mock_response
):
    """With raw=True, the base llm:request and llm:response events carry a `raw` field."""
    provider = make_provider(raw=True)
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request(metadata={"stream": False}))

    request_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:request"
    ]
    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]

    assert len(request_events) >= 1, "llm:request should be emitted"
    assert "raw" in request_events[0], (
        "llm:request should have 'raw' field when raw=True"
    )

    assert len(response_events) >= 1, "llm:response (ok) should be emitted"
    assert "raw" in response_events[0], (
        "llm:response should have 'raw' field when raw=True"
    )
