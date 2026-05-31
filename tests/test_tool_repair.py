"""Tests for tool result repair and infinite loop prevention."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import Message
from amplifier_core.message_models import ToolCallBlock
from amplifier_module_provider_ollama import OllamaProvider


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def make_mock_response(content: str = "Test response"):
    """Create a mock ollama response."""
    return {
        "message": {"role": "assistant", "content": content},
        "done": True,
        "model": "llama3.2:3b",
        "prompt_eval_count": 10,
        "eval_count": 5,
    }


def test_tool_call_sequence_missing_tool_message_is_repaired():
    """Missing tool results should be repaired with synthetic results and emit event."""
    provider = OllamaProvider(host="http://localhost:11434")
    provider.client.chat = AsyncMock(return_value=make_mock_response())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_1", name="do_something", input={"value": 1})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages, metadata={"stream": False})

    asyncio.run(provider.complete(request))

    # Should succeed (not raise validation error)
    provider.client.chat.assert_awaited_once()

    # Should not emit validation error
    assert all(
        event_name != "provider:validation_error"
        for event_name, _ in fake_coordinator.hooks.events
    )

    # Should emit repair event
    repair_events = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["provider"] == "ollama"
    assert repair_events[0][1]["repair_count"] == 1
    assert repair_events[0][1]["repairs"][0]["tool_name"] == "do_something"


def test_repaired_tool_ids_are_not_detected_again():
    """Repaired tool IDs should be tracked and not trigger infinite detection loops.

    This test verifies the fix for the infinite loop bug where:
    1. Missing tool results are detected and synthetic results are injected
    2. Synthetic results are NOT persisted to message store
    3. On next iteration, same missing tool results are detected again
    4. This creates an infinite loop of detection -> injection -> detection

    The fix tracks repaired tool IDs to skip re-detection.
    """
    provider = OllamaProvider(host="http://localhost:11434")
    provider.client.chat = AsyncMock(return_value=make_mock_response())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Create a request with missing tool result
    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_abc123", name="grep", input={"pattern": "test"})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages, metadata={"stream": False})

    # First call - should detect and repair
    asyncio.run(provider.complete(request))

    # Verify repair happened
    assert "call_abc123" in provider._repaired_tool_ids  # pyright: ignore[reportAttributeAccessIssue]
    repair_events_1 = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events_1) == 1

    # Clear events for second call
    fake_coordinator.hooks.events.clear()

    # Second call with SAME messages (simulating message store not persisting synthetic results)
    # This would previously cause infinite loop detection
    messages_2 = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_abc123", name="grep", input={"pattern": "test"})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request_2 = ChatRequest(messages=messages_2, metadata={"stream": False})

    asyncio.run(provider.complete(request_2))

    # Should NOT emit another repair event for the same tool ID
    repair_events_2 = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events_2) == 0, "Should not re-detect already-repaired tool IDs"


def test_multiple_missing_tool_results_all_tracked():
    """Multiple missing tool results should all be tracked to prevent infinite loops."""
    provider = OllamaProvider(host="http://localhost:11434")
    provider.client.chat = AsyncMock(return_value=make_mock_response())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Create request with 3 parallel tool calls, none with results
    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_1", name="grep", input={"pattern": "a"}),
                ToolCallBlock(id="call_2", name="grep", input={"pattern": "b"}),
                ToolCallBlock(id="call_3", name="grep", input={"pattern": "c"}),
            ],
        ),
        Message(role="user", content="No tool results"),
    ]
    request = ChatRequest(messages=messages, metadata={"stream": False})

    asyncio.run(provider.complete(request))

    # All 3 should be tracked
    assert provider._repaired_tool_ids == {"call_1", "call_2", "call_3"}  # pyright: ignore[reportAttributeAccessIssue]

    # Verify repair event has all 3
    repair_events = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["repair_count"] == 3


def test_synthetic_results_inserted_before_subsequent_user_message():
    """Synthetic tool results must be inserted immediately after the assistant
    message, NOT appended at the end after subsequent user messages.

    Before the fix, the staging list + extend() pattern placed synthetics at
    the very end, after any user messages that came after the assistant turn.
    This caused Ollama to reject the request due to invalid message ordering.
    """
    provider = OllamaProvider(host="http://localhost:11434")
    provider.client.chat = AsyncMock(return_value=make_mock_response())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    messages = [
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_xyz", name="search", input={"q": "test"})],
        ),
        Message(role="user", content="What did the search return?"),
    ]
    request = ChatRequest(messages=messages, metadata={"stream": False})

    asyncio.run(provider.complete(request))

    provider.client.chat.assert_awaited_once()
    call_kwargs = provider.client.chat.call_args
    ollama_msgs = call_kwargs.kwargs.get(
        "messages", call_kwargs.args[0] if call_kwargs.args else []
    )

    # Pull out role sequence from the Ollama-format message list
    roles = [m["role"] for m in ollama_msgs]

    # The tool result must appear BEFORE the user message — not at the end
    assert "tool" in roles, "Synthetic tool result should be present"
    tool_idx = next(i for i, m in enumerate(ollama_msgs) if m["role"] == "tool")
    user_idx = next(
        i
        for i, m in enumerate(ollama_msgs)
        if m["role"] == "user" and m.get("content") == "What did the search return?"
    )
    assert tool_idx < user_idx, (
        f"Synthetic tool result (idx={tool_idx}) must come BEFORE the subsequent "
        f"user message (idx={user_idx}). Got roles: {roles}"
    )


def test_fm3_synthetic_assistant_inserted_before_next_user_message():
    """After synthetic tool results are inserted, a synthetic assistant message
    (FM3) should be added immediately before the next real user message so the
    conversation structure is valid: assistant → tool(s) → assistant → user.
    """
    provider = OllamaProvider(host="http://localhost:11434")
    provider.client.chat = AsyncMock(return_value=make_mock_response())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_fm3", name="read_file", input={"path": "/tmp/x"})
            ],
        ),
        Message(role="user", content="Continue please"),
    ]
    request = ChatRequest(messages=messages, metadata={"stream": False})

    asyncio.run(provider.complete(request))

    provider.client.chat.assert_awaited_once()
    call_kwargs = provider.client.chat.call_args
    ollama_msgs = call_kwargs.kwargs.get(
        "messages", call_kwargs.args[0] if call_kwargs.args else []
    )

    roles = [m["role"] for m in ollama_msgs]

    # Must have: tool result somewhere, then assistant (FM3), then user
    assert "tool" in roles, "Synthetic tool result should be present"
    tool_idx = next(i for i, m in enumerate(ollama_msgs) if m["role"] == "tool")

    # The message right after the tool result should be a synthetic assistant (FM3)
    assert tool_idx + 1 < len(ollama_msgs), (
        "There should be a message after the tool result"
    )
    assert ollama_msgs[tool_idx + 1]["role"] == "assistant", (
        f"FM3: expected assistant message after tool result at idx {tool_idx + 1}, "
        f"got '{ollama_msgs[tool_idx + 1]['role']}'. Full roles: {roles}"
    )

    # And the real user message should come after FM3
    user_idx = next(
        i
        for i, m in enumerate(ollama_msgs)
        if m["role"] == "user" and m.get("content") == "Continue please"
    )
    assert user_idx == tool_idx + 2, (
        f"Real user message should be at tool_idx+2={tool_idx + 2}, "
        f"got {user_idx}. Roles: {roles}"
    )


def test_fm3_not_inserted_when_no_following_user_message():
    """FM3 synthetic assistant should NOT be inserted when there is no real
    user message immediately following the injected tool results.

    For example, if the assistant turn with missing tool calls is the last
    message in the conversation, only the synthetic tool results are added —
    no FM3 assistant message.
    """
    provider = OllamaProvider(host="http://localhost:11434")
    provider.client.chat = AsyncMock(return_value=make_mock_response())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Assistant tool call is the LAST message — no subsequent user message
    messages = [
        Message(role="user", content="Please do something"),
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_last", name="run_cmd", input={"cmd": "ls"})
            ],
        ),
    ]
    request = ChatRequest(messages=messages, metadata={"stream": False})

    asyncio.run(provider.complete(request))

    provider.client.chat.assert_awaited_once()
    call_kwargs = provider.client.chat.call_args
    ollama_msgs = call_kwargs.kwargs.get(
        "messages", call_kwargs.args[0] if call_kwargs.args else []
    )

    roles = [m["role"] for m in ollama_msgs]

    # Synthetic tool result should be present
    assert "tool" in roles, "Synthetic tool result should be present"
    tool_idx = next(i for i, m in enumerate(ollama_msgs) if m["role"] == "tool")

    # The message after the tool result (if any) should NOT be a synthetic
    # assistant — since we're at the end there's nothing after
    if tool_idx + 1 < len(ollama_msgs):
        # If something follows, it must not be an FM3 synthetic assistant
        # (there's no real user message to precede)
        next_role = ollama_msgs[tool_idx + 1]["role"]
        assert next_role != "assistant", (
            f"FM3 should NOT be inserted when there is no following user message. "
            f"Got '{next_role}' at idx {tool_idx + 1}. Roles: {roles}"
        )
