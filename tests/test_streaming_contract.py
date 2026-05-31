"""
TDD tests for the Ollama provider streaming contract.

Asserts the five-event contract defined in provider-streaming-contract.md:
  llm:stream_block_start, llm:stream_block_delta, llm:stream_thinking_delta,
  llm:stream_block_end, llm:stream_aborted.

All assertions are against exact event names and required payload keys.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_ollama import OllamaProvider


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_provider() -> OllamaProvider:
    """Minimal wired provider backed by FakeCoordinator."""
    from tests.conftest import FakeCoordinator
    from typing import cast
    from amplifier_core import ModuleCoordinator

    p = OllamaProvider(host="http://localhost:11434")
    p.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return p


def _events(provider: OllamaProvider) -> list[tuple[str, dict[str, Any]]]:
    return provider.coordinator.hooks.events  # type: ignore[union-attr]


def _stream_events(provider: OllamaProvider) -> list[tuple[str, dict[str, Any]]]:
    return [(n, p) for n, p in _events(provider) if n.startswith("llm:stream")]


def _request(metadata: dict | None = None) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="hello")],
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Fake stream builders
# ---------------------------------------------------------------------------


async def _text_stream(chunks: list[str]):
    """Async generator yielding plain text chunks then a done chunk."""
    for text in chunks:
        yield {"message": {"content": text, "thinking": ""}, "done": False}
    yield {
        "message": {"content": "", "thinking": ""},
        "done": True,
        "prompt_eval_count": 5,
        "eval_count": 3,
        "model": "llama3.2:3b",
    }


async def _thinking_then_text_stream(thinking_chunks: list[str], text_chunks: list[str]):
    """Thinking deltas, then text deltas, then done."""
    for t in thinking_chunks:
        yield {"message": {"content": "", "thinking": t}, "done": False}
    for c in text_chunks:
        yield {"message": {"content": c, "thinking": ""}, "done": False}
    yield {
        "message": {"content": "", "thinking": ""},
        "done": True,
        "prompt_eval_count": 5,
        "eval_count": 3,
        "model": "llama3.2:3b",
    }


async def _tool_call_stream(tool_name: str, tool_args: dict):
    """Yields a single done chunk containing a tool call (no content)."""
    yield {
        "message": {
            "content": "",
            "thinking": "",
            "tool_calls": [
                {"function": {"name": tool_name, "arguments": tool_args}}
            ],
        },
        "done": True,
        "prompt_eval_count": 2,
        "eval_count": 1,
        "model": "llama3.2:3b",
    }


async def _error_mid_stream():
    """Yields one text chunk then raises RuntimeError."""
    yield {"message": {"content": "hello", "thinking": ""}, "done": False}
    raise RuntimeError("mid-stream failure")


# ---------------------------------------------------------------------------
# Test 1: default routing uses streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_routing_uses_streaming_path():
    """complete() with no metadata opt-out must go through the streaming path,
    evidenced by llm:stream_block_* events appearing."""
    provider = _make_provider()
    provider.client.chat = AsyncMock(return_value=_text_stream(["hello"]))

    await provider.complete(_request())

    names = [n for n, _ in _stream_events(provider)]
    assert "llm:stream_block_start" in names, (
        "Default routing must use streaming path (emit llm:stream_block_start)"
    )


# ---------------------------------------------------------------------------
# Test 2: metadata stream=False uses non-streaming path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_stream_false_uses_nonstreaming():
    """request.metadata['stream'] is False must route to _complete_chat_request,
    which emits NO llm:stream_* events."""

    def _mock_dict_response():
        return {
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
            "model": "llama3.2:3b",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

    provider = _make_provider()
    provider.client.chat = AsyncMock(return_value=_mock_dict_response())

    await provider.complete(_request(metadata={"stream": False}))

    stream_names = [n for n, _ in _stream_events(provider)]
    assert stream_names == [], (
        f"Non-streaming path must not emit llm:stream_* events, got: {stream_names}"
    )


# ---------------------------------------------------------------------------
# Test 3: single request_id constant across all events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_request_id_across_all_events():
    """All llm:stream_* events for one call must share the same request_id."""
    provider = _make_provider()
    provider.client.chat = AsyncMock(
        return_value=_thinking_then_text_stream(["think"], ["text"])
    )

    await provider.complete(_request())

    stream_evts = _stream_events(provider)
    assert stream_evts, "Must emit at least one stream event"

    request_ids = {p["request_id"] for _, p in stream_evts if "request_id" in p}
    assert len(request_ids) == 1, (
        f"All stream events must share one request_id, got {len(request_ids)}: {request_ids}"
    )
    (rid,) = request_ids
    assert isinstance(rid, str) and len(rid) == 36, (
        f"request_id must be a UUID4 string, got {rid!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: thinking-then-text produces correct block sequence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_then_text_produces_correct_event_sequence():
    """A stream with thinking then text chunks must emit:
      block_start(thinking, idx=0) -> thinking_deltas -> block_end(thinking, idx=0)
      -> block_start(text, idx=1) -> block_deltas -> block_end(text, idx=1)
    """
    provider = _make_provider()
    provider.client.chat = AsyncMock(
        return_value=_thinking_then_text_stream(["t1", "t2"], ["c1", "c2"])
    )

    await provider.complete(_request())

    evts = _stream_events(provider)
    names = [n for n, _ in evts]

    # Must have all event types
    assert "llm:stream_block_start" in names
    assert "llm:stream_thinking_delta" in names
    assert "llm:stream_block_delta" in names
    assert "llm:stream_block_end" in names

    # Extract the block_start and block_end events in order
    starts = [(n, p) for n, p in evts if n == "llm:stream_block_start"]
    ends = [(n, p) for n, p in evts if n == "llm:stream_block_end"]

    assert len(starts) == 2, f"Expected 2 block_starts, got {len(starts)}"
    assert len(ends) == 2, f"Expected 2 block_ends, got {len(ends)}"

    # First block: thinking at index 0
    _, s0 = starts[0]
    assert s0["block_type"] == "thinking", f"First block should be thinking, got {s0}"
    assert s0["block_index"] == 0

    # Second block: text at index 1
    _, s1 = starts[1]
    assert s1["block_type"] == "text", f"Second block should be text, got {s1}"
    assert s1["block_index"] == 1

    # block_end types match their starts
    _, e0 = ends[0]
    _, e1 = ends[1]
    assert e0["block_index"] == 0 and e0["block_type"] == "thinking"
    assert e1["block_index"] == 1 and e1["block_type"] == "text"

    # Ordering: thinking block_start must precede text block_start
    thinking_start_pos = next(
        i for i, (n, p) in enumerate(evts)
        if n == "llm:stream_block_start" and p.get("block_type") == "thinking"
    )
    text_start_pos = next(
        i for i, (n, p) in enumerate(evts)
        if n == "llm:stream_block_start" and p.get("block_type") == "text"
    )
    assert thinking_start_pos < text_start_pos, "thinking block must come before text block"


# ---------------------------------------------------------------------------
# Test 5: per-block sequences reset at block boundaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_block_sequences_reset_at_block_boundary():
    """sequence counter must restart at 0 for each new block_index."""
    provider = _make_provider()
    provider.client.chat = AsyncMock(
        return_value=_thinking_then_text_stream(["ta", "tb", "tc"], ["ca", "cb"])
    )

    await provider.complete(_request())

    evts = _stream_events(provider)

    # thinking deltas (block_index=0)
    thinking_seqs = [
        p["sequence"]
        for n, p in evts
        if n == "llm:stream_thinking_delta"
    ]
    # text deltas (block_index=1)
    text_seqs = [
        p["sequence"]
        for n, p in evts
        if n == "llm:stream_block_delta"
    ]

    assert thinking_seqs == [0, 1, 2], f"thinking sequences wrong: {thinking_seqs}"
    assert text_seqs == [0, 1], f"text sequences wrong: {text_seqs}"


# ---------------------------------------------------------------------------
# Test 6: empty fragments never emitted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_fragments_not_emitted():
    """Chunks with empty content/thinking strings must produce no delta events."""
    provider = _make_provider()

    async def _empty_then_real():
        yield {"message": {"content": "", "thinking": ""}, "done": False}
        yield {"message": {"content": "hello", "thinking": ""}, "done": False}
        yield {
            "message": {"content": "", "thinking": ""},
            "done": True,
            "prompt_eval_count": 2,
            "eval_count": 1,
            "model": "llama3.2:3b",
        }

    provider.client.chat = AsyncMock(return_value=_empty_then_real())

    await provider.complete(_request())

    deltas = [
        (n, p) for n, p in _stream_events(provider)
        if n in ("llm:stream_block_delta", "llm:stream_thinking_delta")
    ]
    assert len(deltas) == 1, f"Only one non-empty delta expected, got {len(deltas)}: {deltas}"
    _, payload = deltas[0]
    assert payload["text"] == "hello"


# ---------------------------------------------------------------------------
# Test 7: tool calls emitted as atomic blocks (start+end, no deltas)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_calls_emitted_as_atomic_blocks():
    """Tool call in final chunk must emit block_start(tool_use, name=...) + block_end,
    no intermediate deltas."""
    provider = _make_provider()
    provider.client.chat = AsyncMock(
        return_value=_tool_call_stream("get_weather", {"city": "NYC"})
    )

    await provider.complete(_request())

    evts = _stream_events(provider)

    tool_starts = [
        (n, p) for n, p in evts
        if n == "llm:stream_block_start" and p.get("block_type") == "tool_use"
    ]
    tool_ends = [
        (n, p) for n, p in evts
        if n == "llm:stream_block_end" and p.get("block_type") == "tool_use"
    ]

    assert len(tool_starts) == 1, f"Expected 1 tool_use block_start, got {len(tool_starts)}"
    assert len(tool_ends) == 1, f"Expected 1 tool_use block_end, got {len(tool_ends)}"

    _, sp = tool_starts[0]
    assert sp.get("name") == "get_weather", (
        f"tool_use block_start must carry 'name', got {sp}"
    )

    # No delta events for tool calls
    deltas = [n for n, _ in evts if n in ("llm:stream_block_delta", "llm:stream_thinking_delta")]
    assert deltas == [], f"tool_use blocks must not emit deltas, got {deltas}"


# ---------------------------------------------------------------------------
# Test 8: error after partial emit -> llm:stream_aborted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_after_partial_emit_produces_stream_aborted():
    """RuntimeError mid-stream (after at least one delta was emitted) must
    emit llm:stream_aborted{request_id, error:{type, msg}} and then re-raise."""
    provider = _make_provider()
    provider.client.chat = AsyncMock(return_value=_error_mid_stream())

    with pytest.raises(Exception):
        await provider.complete(_request())

    aborted = [(n, p) for n, p in _events(provider) if n == "llm:stream_aborted"]
    assert len(aborted) == 1, f"Expected exactly 1 llm:stream_aborted, got {len(aborted)}"

    _, payload = aborted[0]
    assert "request_id" in payload, "llm:stream_aborted must have request_id"
    assert "error" in payload, "llm:stream_aborted must have error"
    error = payload["error"]
    assert "type" in error and "msg" in error, (
        f"error must have 'type' and 'msg' keys, got {error}"
    )
    assert error["type"] == "RuntimeError"
    assert "mid-stream failure" in error["msg"]


# ---------------------------------------------------------------------------
# Test 9: no stream_aborted when error before any delta
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_stream_aborted_when_error_before_any_delta():
    """If an error occurs before any delta is emitted, llm:stream_aborted must NOT
    be emitted (partial_emitted is False)."""
    provider = _make_provider()
    provider.client.chat = AsyncMock(side_effect=ConnectionError("refused"))

    with pytest.raises(Exception):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await provider.complete(_request())

    aborted = [(n, p) for n, p in _events(provider) if n == "llm:stream_aborted"]
    assert len(aborted) == 0, (
        f"llm:stream_aborted must not be emitted when no delta was sent, got {aborted}"
    )


# ---------------------------------------------------------------------------
# Test 10: block_delta payload shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_block_delta_payload_shape():
    """Each llm:stream_block_delta must have: request_id, block_index, sequence, text."""
    provider = _make_provider()
    provider.client.chat = AsyncMock(return_value=_text_stream(["hello", " world"]))

    await provider.complete(_request())

    deltas = [(n, p) for n, p in _stream_events(provider) if n == "llm:stream_block_delta"]
    assert deltas, "Must have at least one block_delta"

    for _, payload in deltas:
        assert "request_id" in payload, f"block_delta missing request_id: {payload}"
        assert "block_index" in payload, f"block_delta missing block_index: {payload}"
        assert "sequence" in payload, f"block_delta missing sequence: {payload}"
        assert "text" in payload, f"block_delta missing text: {payload}"
        assert payload["text"], f"block_delta must not have empty text: {payload}"


# ---------------------------------------------------------------------------
# Test 11: thinking_delta payload shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_delta_payload_shape():
    """Each llm:stream_thinking_delta must have: request_id, block_index, sequence, text."""
    provider = _make_provider()
    provider.client.chat = AsyncMock(
        return_value=_thinking_then_text_stream(["think1"], ["text1"])
    )

    await provider.complete(_request())

    thinking_deltas = [
        (n, p) for n, p in _stream_events(provider)
        if n == "llm:stream_thinking_delta"
    ]
    assert thinking_deltas, "Must emit at least one thinking_delta for thinking stream"

    for _, payload in thinking_deltas:
        assert "request_id" in payload
        assert "block_index" in payload
        assert "sequence" in payload
        assert "text" in payload
        assert payload["text"], f"thinking_delta must not have empty text: {payload}"


# ---------------------------------------------------------------------------
# Test 12: block_start and block_end payload shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_block_start_end_payload_shape():
    """block_start/end must carry: request_id, block_index, block_type."""
    provider = _make_provider()
    provider.client.chat = AsyncMock(return_value=_text_stream(["hello"]))

    await provider.complete(_request())

    for event_name in ("llm:stream_block_start", "llm:stream_block_end"):
        events_of_type = [p for n, p in _stream_events(provider) if n == event_name]
        assert events_of_type, f"Must emit at least one {event_name}"
        for payload in events_of_type:
            assert "request_id" in payload, f"{event_name} missing request_id"
            assert "block_index" in payload, f"{event_name} missing block_index"
            assert "block_type" in payload, f"{event_name} missing block_type"


# ---------------------------------------------------------------------------
# Test 13: block_index is shared across all block types
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_block_index_shared_across_block_types():
    """block_index must be a shared counter: first block=0, second block=1."""
    provider = _make_provider()
    provider.client.chat = AsyncMock(
        return_value=_thinking_then_text_stream(["think"], ["text"])
    )

    await provider.complete(_request())

    evts = _stream_events(provider)
    starts = [(n, p) for n, p in evts if n == "llm:stream_block_start"]

    indices = [p["block_index"] for _, p in starts]
    assert indices == [0, 1], f"Shared block_index must be [0, 1], got {indices}"
