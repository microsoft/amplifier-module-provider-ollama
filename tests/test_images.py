"""Offline image regression and native SDK transport fixtures (not server proof)."""

import base64
import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import ollama
import pytest
from amplifier_core.message_models import ChatRequest, Message, ToolCallBlock


PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+aN1sAAAAASUVORK5CYII="
)
PIXELS = base64.b64decode(PNG)


def image(data=PNG):
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


def history():
    return [
        Message(role="user", content=[
            {"type": "text", "text": "Inspect the file"}, image(),
        ]),
        Message(
            role="assistant",
            content=[
                {"type": "text", "text": "Reading"}, image(),
                ToolCallBlock(id="read-1", name="read_file", input={}),
            ],
        ),
        Message(
            role="tool",
            tool_call_id="read-1",
            content=[{"type": "text", "text": "before"}, image(),
                     {"type": "text", "text": "after"}],
        ),
    ]


@pytest.mark.parametrize("role", ["user", "tool", "assistant", "system", "developer"])
def test_native_images_stay_in_original_message(make_provider, role):
    provider = make_provider()
    original = [{"role": role, "tool_call_id": "read-1", "name": "read_file",
                 "content": ["before", image(), "after", image()]}]
    snapshot = copy.deepcopy(original)
    converted = provider._convert_messages(original)
    assert len(converted) == 1
    assert converted[0]["images"] == [PIXELS, PIXELS]
    assert converted[0]["role"] == ("user" if role == "developer" else role)
    assert "before\nafter" in converted[0]["content"]
    if role == "tool":
        assert converted[0]["tool_call_id"] == "read-1"
        assert converted[0]["tool_name"] == "read_file"
    assert original == snapshot


def test_tool_names_come_from_matching_call_not_nearest_tool(make_provider):
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "tool": "first", "arguments": {}},
            {"id": "b", "tool": "second", "arguments": {}},
        ]},
        {"role": "tool", "tool_call_id": "b", "content": [image()]},
        {"role": "tool", "tool_call_id": "a", "content": "plain"},
    ]
    result = make_provider()._convert_messages(messages)
    assert [m["role"] for m in result] == ["assistant", "tool", "tool"]
    assert result[1]["tool_name"] == "second"
    assert result[2]["tool_name"] == "first"
    assert result[1]["tool_call_id"] == "b"
    assert result[2]["tool_call_id"] == "a"


@pytest.mark.parametrize("source", [
    None, "base64", [], {}, {"type": "base64"}, {"type": "base64", "data": ""},
    {"type": "base64", "data": "/not/an/image.png"},
    {"type": "base64", "data": 123},
    {"type": "url", "url": "https://example.invalid/private?secret=do-not-emit"},
    {"type": "file", "path": "/private/image.png"},
])
def test_unavailable_image_is_honest_and_does_not_poison_later_text(
    make_provider, source
):
    messages = [
        {"role": "tool", "tool_call_id": "a",
         "content": ["before", {"type": "image", "source": source}, "after"]},
        {"role": "user", "content": "still here"},
    ]
    before = copy.deepcopy(messages)
    result = make_provider()._convert_messages(messages)
    assert result[0]["content"].startswith("before\n[Image unavailable:")
    assert result[0]["content"].endswith("\nafter")
    assert "images" not in result[0]
    assert "do-not-emit" not in repr(result)
    assert result[1] == {"role": "user", "content": "still here"}
    assert messages == before


def test_base64_never_uses_sdk_path_interpretation(make_provider, monkeypatch):
    # "AAAA" is valid base64 AND a legal local filename. The provider must pass
    # bytes, not allow the SDK to open a path with this name.
    converted = make_provider()._convert_messages(
        [{"role": "user", "content": [image("AAAA")]}]
    )
    reads = []
    def record_read(*args, **kwargs):
        reads.append(args)
        return b"wrong local file"
    monkeypatch.setattr(Path, "read_bytes", record_read)
    monkeypatch.setattr(Path, "exists", lambda *args: True)
    assert ollama.Image(value=converted[0]["images"][0]).model_dump() == "AAAA"
    assert reads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_native_sdk_transport_and_redacted_events(make_provider, stream):
    captured = []

    def transport(request):
        assert request.url.host == "ollama-fixture.invalid"
        captured.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "vision", "tools"]})
        assert request.url.path == "/api/chat"
        response = {"message": {"role": "assistant", "content": "ok"}, "done": True,
                    "prompt_eval_count": 10, "eval_count": 1}
        if stream:
            return httpx.Response(200, content=json.dumps(response) + "\n",
                                  headers={"Content-Type": "application/x-ndjson"})
        return httpx.Response(200, json=response)

    provider = make_provider(raw=True, num_ctx=4096, use_streaming=stream)
    await provider.client._client.aclose()
    provider._client = ollama.AsyncClient(
        host="http://ollama-fixture.invalid", transport=httpx.MockTransport(transport)
    )
    emit = provider.coordinator.hooks.emit

    async def mutating_hook(name, data):
        await emit(name, data)
        if name == "llm:request":
            # A diagnostic consumer cannot corrupt the pending SDK request.
            data["raw"]["messages"][-1]["content"] = "diagnostic mutation"

    provider.coordinator.hooks.emit = mutating_hook
    messages = [
        Message(role="system", content=[image(), {"type": "text", "text": "system"}]),
        Message(role="developer", content=[image(), {"type": "text", "text": "context"}]),
        *history(),
    ]
    request = ChatRequest(messages=messages)
    snapshot = request.model_dump()
    try:
        await provider.complete(request)
    finally:
        await provider.client._client.aclose()
    payload = next(data for path, data in captured if path == "/api/chat")
    assert [m["role"] for m in payload["messages"]] == [
        "system", "user", "user", "assistant", "tool",
    ]
    for index in range(5):
        assert payload["messages"][index]["images"] == [PNG]
    assert payload["messages"][3]["tool_calls"][0]["function"]["name"] == "read_file"
    assert [path for path, _ in captured] == ["/api/chat"]
    assert payload["messages"][-1]["tool_name"] == "read_file"
    # Native Ollama correlates by tool_name + message ordering, not OpenAI IDs.
    assert "tool_call_id" not in payload["messages"][-1]
    assert request.model_dump() == snapshot
    events = provider.coordinator.hooks.events
    assert PNG not in repr(events)
    assert repr(PIXELS) not in repr(events)
    raw = next(data["raw"] for name, data in events if name == "llm:request")
    assert raw["messages"][-1]["images"] == ["[REDACTED image]"]
    raw["messages"][-1]["content"] = "changed diagnostic copy"
    assert payload["messages"][-1]["content"] == "before\nafter"


@pytest.mark.asyncio
@pytest.mark.parametrize("capabilities", [
    ["completion", "tools"], [], None, ["completion", "vision", "tools"],
])
@pytest.mark.parametrize("stream", [False, True])
async def test_model_capability_is_not_guessed_from_name(make_provider, capabilities, stream):
    provider = make_provider(use_streaming=stream)
    provider.client.show = AsyncMock(return_value={"capabilities": capabilities})
    response = {
        "message": {"role": "assistant", "content": "ok"}, "done": True,
    }

    async def chunks():
        yield response

    provider.client.chat = AsyncMock(return_value=chunks() if stream else response)
    try:
        await provider.complete(ChatRequest(messages=history()), model="llava-vision-custom")
    finally:
        await provider.client._client.aclose()
    messages = provider.client.chat.call_args.kwargs["messages"]
    if capabilities is None or "vision" in capabilities:
        # Older servers lack capability metadata. Forward native bytes rather
        # than inventing a capability or silently suppressing working user images.
        assert messages[-1]["images"] == [PIXELS]
    else:
        assert "images" not in messages[-1]
        assert "[Image unavailable:" in messages[-1]["content"]
    assert messages[-1]["tool_call_id"] == "read-1"
    provider.client.show.assert_awaited_once_with("llava-vision-custom")


@pytest.mark.asyncio
async def test_plain_text_never_queries_image_capability(make_provider):
    provider = make_provider(use_streaming=False, num_ctx=4096)
    provider.client.show = AsyncMock(side_effect=AssertionError("no image query"))
    provider.client.chat = AsyncMock(return_value={
        "message": {"role": "assistant", "content": "ok"}, "done": True,
    })
    try:
        await provider.complete(ChatRequest(messages=[Message(role="user", content="hello")]))
    finally:
        await provider.client._client.aclose()
    assert provider.client.chat.call_args.kwargs["messages"] == [
        {"role": "user", "content": "hello"},
    ]
    provider.client.show.assert_not_awaited()


def test_actual_core_tool_call_block_keeps_image_parent(make_provider):
    original = [
        Message(role="assistant", content=[
            image(),
            ToolCallBlock(id="parent", name="read_file", input={"file_path": "image.png"}),
        ]),
        Message(role="tool", tool_call_id="parent", content=[image()]),
    ]
    result = make_provider()._convert_messages([m.model_dump() for m in original])
    assert result[0]["tool_calls"][0]["function"] == {
        "name": "read_file", "arguments": {"file_path": "image.png"},
    }
    assert result[1]["tool_name"] == "read_file"
    assert result[1]["tool_call_id"] == "parent"
    assert [m["images"] for m in result] == [[PIXELS], [PIXELS]]


def test_multiple_images_preserve_original_bytes_without_size_policy(make_provider):
    # No decoder/transcoder or arbitrary size cap at this provider boundary.
    original = b"\x00\xff" * (1024 * 1024)
    encoded = base64.b64encode(original).decode("ascii")
    result = make_provider()._convert_messages([{
        "role": "user", "content": [image(encoded), image(PNG)],
    }])
    assert result[0]["images"] == [original, PIXELS]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_image_server_rejection_does_not_replay_or_relocate(make_provider, stream):
    from amplifier_core import InvalidRequestError

    requests = []

    def reject(request):
        data = json.loads(request.content)
        requests.append(data)
        assert data["messages"][-1]["role"] == "tool"
        assert data["messages"][-1]["images"] == [PNG]
        return httpx.Response(400, json={"error": "this model does not support images"})

    provider = make_provider(num_ctx=4096, use_streaming=stream)
    provider._client = ollama.AsyncClient(
        host="http://ollama-fixture.invalid", transport=httpx.MockTransport(reject)
    )
    request = ChatRequest(messages=history())
    before = request.model_dump()
    try:
        with pytest.raises(InvalidRequestError):
            await provider.complete(request)
    finally:
        await provider.client._client.aclose()
    assert len(requests) == 1
    assert request.model_dump() == before


@pytest.mark.parametrize("next_info", [{}, {"capabilities": None}, {"capabilities": [1]}])
def test_missing_metadata_replaces_stale_negative_vision_evidence(make_provider, next_info):
    provider = make_provider()
    provider._record_image_capabilities("custom", {"capabilities": ["completion"]})
    assert provider._model_vision["custom"] is False
    provider._record_image_capabilities("custom", next_info)
    messages = provider._convert_messages([{"role": "user", "content": [image()]}])
    provider._apply_image_capabilities("custom", messages)
    assert messages[0]["images"] == [PIXELS]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_mid_stream", [False, True])
async def test_real_sdk_multichunk_preserves_thinking_tools_and_no_replay(
    make_provider, fail_mid_stream
):
    from amplifier_core import LLMError

    requests = []
    chunks = [
        {"message": {"role": "assistant", "thinking": "inspect"}, "done": False},
        {"message": {"role": "assistant", "content": "answer"}, "done": False},
        {"message": {"role": "assistant", "tool_calls": [{
            "function": {"name": "next_tool", "arguments": {"value": 1}},
        }]}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True,
         "model": "fixture-model", "prompt_eval_count": 10, "eval_count": 3},
    ]

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in chunks[:3] if fail_mid_stream else chunks:
                yield (json.dumps(chunk) + "\n").encode()
            if fail_mid_stream:
                raise httpx.ReadError("fixture interrupted after partial output")

    def transport(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, stream=Body(),
                              headers={"Content-Type": "application/x-ndjson"})

    provider = make_provider(num_ctx=4096, raw=True)
    provider._client = ollama.AsyncClient(
        host="http://ollama-fixture.invalid", transport=httpx.MockTransport(transport)
    )
    request = ChatRequest(messages=history())
    before = request.model_dump()
    try:
        if fail_mid_stream:
            with pytest.raises(LLMError):
                await provider.complete(request)
            assert any(name == "llm:stream_aborted"
                       for name, _ in provider.coordinator.hooks.events)
        else:
            response = await provider.complete(request)
            assert response.text == "answer"
            assert response.thinking_content == "inspect"
            assert response.tool_calls[0].name == "next_tool"
            assert response.tool_calls[0].arguments == {"value": 1}
            assert isinstance(response.raw_response, dict)
    finally:
        await provider.client._client.aclose()
    assert len(requests) == 1
    assert requests[0]["messages"][-1]["images"] == [PNG]
    assert request.model_dump() == before

@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_echoed_pixels_private_in_retry_and_terminal_errors(make_provider, caplog, raw, stream):
    import traceback
    from amplifier_core import InvalidRequestError

    attempts = []

    def handler(request):
        assert json.loads(request.content)["messages"][-1]["images"] == [PNG]
        attempts.append(request.url.path)
        # Ollama streaming opens the HTTP connection during iteration.
        status = 503 if not stream and len(attempts) == 1 else 400
        return httpx.Response(status, json={"error": f"image validation failed for {PNG}"})

    instance = make_provider(raw=raw, num_ctx=4096, use_streaming=stream, max_retries=1,
                             min_retry_delay=0.001)
    instance._client = ollama.AsyncClient(
        host="http://ollama-fixture.invalid", transport=httpx.MockTransport(handler)
    )
    req = ChatRequest(messages=history())
    before = req.model_dump()
    try:
        with pytest.raises(InvalidRequestError) as caught:
            await instance.complete(req)
    finally:
        await instance.client._client.aclose()
    events = instance.coordinator.hooks.events
    assert len(attempts) == (1 if stream else 2)
    assert sum(name == "provider:retry" for name, _ in events) == (0 if stream else 1)
    assert caught.value.status_code == 400 and not caught.value.retryable
    assert "image validation failed" in str(caught.value)
    assert PNG not in "".join(traceback.format_exception(caught.value)) + caplog.text + repr(events)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert req.model_dump() == before


def test_image_encoded_once_per_request_across_long_diagnostics(monkeypatch):
    from amplifier_module_provider_ollama import _ImageDiagnostics

    count = 0
    encode = base64.b64encode

    def counting(data):
        nonlocal count
        count += 1
        return encode(data)

    monkeypatch.setattr(base64, "b64encode", counting)
    messages = [{"content": f"text {index}"} for index in range(800)]
    messages.extend([{"images": [PIXELS]}, {"images": [PIXELS]}])
    diagnostic = _ImageDiagnostics(messages)
    diagnostic.copy(messages)
    assert PNG not in diagnostic.copy(f"echo {PNG}")
    assert "[REDACTED image]" == diagnostic.copy(PNG.replace("/", r"\/"))
    assert count == 1
    assert _ImageDiagnostics([]).copy(PNG) == PNG
