"""Tests for cost_usd JSON serializability and contributor behavior (Ollama).

Covers:
  1. llm:response event payload is fully JSON serializable (cost_usd is None for Ollama)
  2. cost_usd is None in the llm:response event usage dict
  3. cost_usd round-trips through json.dumps/loads as None (JSON null)
  4. contributor lambda always returns None for Ollama (cost never accumulates)
  5. Usage model stores None internally after _convert_to_chat_response()

Ollama is self-hosted — cost is always indeterminate (None), never a Decimal.
The contributor lambda fix ensures that if cost_usd were ever a non-None Decimal,
it would be serialized as str() rather than leaking an unserializable type.
"""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest


# ---------------------------------------------------------------------------
# Test 1: llm:response event is fully JSON serializable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_response_event_is_json_serializable(
    make_provider, simple_request, mock_response
):
    """llm:response event payload must be fully JSON serializable.

    Ollama cost is always None (self-hosted); ensure no Decimal leaks into
    the event payload that would raise TypeError on json.dumps.
    """
    provider = make_provider()
    provider.client.chat = AsyncMock(return_value=mock_response())

    await provider.complete(simple_request(metadata={"stream": False}))

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1, "llm:response (ok) event must be emitted"

    payload = response_events[0]
    # Must not raise TypeError (e.g. Decimal is not JSON serializable)
    serialized = json.dumps(payload)
    assert serialized, "json.dumps(payload) must return a non-empty string"

    usage = payload.get("usage", {})
    assert usage.get("cost_usd") is None, (
        f"cost_usd must be None for Ollama (self-hosted), got: {usage.get('cost_usd')!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: cost_usd is None in the llm:response event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_response_event_cost_usd_is_none(
    make_provider, simple_request, mock_response
):
    """cost_usd in the llm:response usage dict must be None for Ollama.

    Ollama does not have a public pricing API; compute_cost() always returns
    None. The event must reflect this — not $0.00 (free), but None (unknown).
    """
    provider = make_provider()
    provider.client.chat = AsyncMock(return_value=mock_response())

    await provider.complete(simple_request(metadata={"stream": False}))

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1, "llm:response (ok) event must be emitted"

    usage = response_events[0].get("usage", {})
    assert "cost_usd" in usage, "usage dict must contain the cost_usd key"
    assert usage["cost_usd"] is None, (
        f"Ollama cost_usd must be None (indeterminate), got {usage['cost_usd']!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: cost_usd round-trips through JSON as None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
async def test_llm_response_event_cost_usd_round_trips_through_json(
    make_provider, simple_request, mock_response
):
    """cost_usd must survive a json.dumps → json.loads round-trip as None.

    JSON null (Python None) must remain None after deserialization.
    This guards against a Decimal accidentally being serialized as a string
    and then coming back as a string instead of None.
    """
    provider = make_provider()
    provider.client.chat = AsyncMock(return_value=mock_response())

    await provider.complete(simple_request(metadata={"stream": False}))

    response_events = [
        payload
        for name, payload in provider.coordinator.hooks.events
        if name == "llm:response" and payload.get("status") == "ok"
    ]
    assert len(response_events) >= 1, "llm:response (ok) event must be emitted"

    payload = response_events[0]
    round_tripped = json.loads(json.dumps(payload))
    cost_usd = round_tripped.get("usage", {}).get("cost_usd")
    assert cost_usd is None, (
        f"cost_usd must round-trip through JSON as None (null), got {cost_usd!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: contributor lambda always returns None for Ollama
# ---------------------------------------------------------------------------


def test_contributor_returns_none_always():
    """Contributor lambda must return None for Ollama unconditionally.

    Ollama is self-hosted — cost is always indeterminate (None).
    compute_cost() returns None for all Ollama models, so _add_cost(None)
    never flips has_data to True. The fixed contributor lambda must therefore
    always return None (the else-branch of `if _totals["has_data"]`).

    This test mirrors the mount() closure directly to exercise the fixed lambda
    pattern without needing a running Ollama server.
    """
    _totals: dict = {"cost_usd": Decimal(0), "has_data": False}

    def _add_cost(cost: Decimal | None) -> None:
        if cost is not None:
            _totals["cost_usd"] += cost
            _totals["has_data"] = True

    # Fixed contributor lambda — mirrors the post-fix version in mount()
    contributor = lambda: (  # noqa: E731
        {
            "cost_usd": (
                str(_totals["cost_usd"]) if _totals["cost_usd"] is not None else None
            )
        }
        if _totals["has_data"]
        else None
    )

    # Before any completions: no data accumulated yet
    assert contributor() is None, "Contributor must return None before any responses"

    # Simulate multiple Ollama completions — compute_cost() always yields None
    _add_cost(None)
    _add_cost(None)
    _add_cost(None)

    # has_data is still False — contributor still returns None
    assert _totals["has_data"] is False, (
        "has_data must stay False for Ollama (cost is always None)"
    )
    assert contributor() is None, (
        "Contributor must return None after Ollama responses (cost always indeterminate)"
    )


# ---------------------------------------------------------------------------
# Test 5: usage model stores None internally after _convert_to_chat_response
# ---------------------------------------------------------------------------


def test_usage_model_stores_none_internally(make_provider, mock_response):
    """_convert_to_chat_response must populate usage.cost_usd as None.

    Ollama compute_cost() always returns None (self-hosted, indeterminate cost).
    The Usage model should store None — not Decimal(0) (free) or any other value.
    """
    provider = make_provider()
    # mock_response(): prompt_eval_count=10, eval_count=5, model="llama3.2:3b"
    response_dict = mock_response()

    result = provider._convert_to_chat_response(response_dict)

    assert result.usage is not None, "_convert_to_chat_response must populate usage"
    cost_usd = getattr(result.usage, "cost_usd", "MISSING")
    assert cost_usd != "MISSING", "usage must have a cost_usd attribute"
    assert cost_usd is None, (
        f"Ollama usage.cost_usd must be None (indeterminate), got {cost_usd!r}"
    )
