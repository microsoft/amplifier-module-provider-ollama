"""Default model waits survive virtual elapsed time; cancellation/errors remain live."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core.llm_errors import LLMError, LLMTimeoutError
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_ollama import OllamaProvider
from tests.conftest import FakeCoordinator


def setup_call(streaming, config=None, failure=None):
    provider = OllamaProvider(
        host="http://test.invalid",
        config={
            "use_streaming": streaming,
            "max_retries": 0,
            **(config or {}),
        },
    )
    provider.coordinator = FakeCoordinator()
    entered = asyncio.Event()
    release = asyncio.Event()
    closed = []

    async def wait():
        entered.set()
        await release.wait()
        if failure:
            raise failure

    async def stream():
        try:
            await wait()
            yield {
                "message": {"content": "done"},
                "done": True,
                "prompt_eval_count": 1,
                "eval_count": 1,
                "model": "test-model",
            }
        finally:
            closed.append(True)

    async def create(**kwargs):
        if streaming and not config:
            return stream()
        await wait()
        if streaming:
            return stream()
        return {
            "message": {"content": "done"},
            "done": True,
            "prompt_eval_count": 1,
            "eval_count": 1,
            "model": "test-model",
        }

    client = MagicMock()
    provider._client = client
    call = client.chat = AsyncMock(side_effect=create)

    request = ChatRequest(messages=[Message(role="user", content="hello")])
    return provider, request, entered, release, closed, call


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_default_wait_survives_an_hour_then_completes(monkeypatch, streaming):
    provider, request, entered, release, closed, _call = setup_call(streaming)
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    loop = asyncio.get_running_loop()
    clock = loop.time
    # No real sleeping: move beyond every former model-work deadline while
    # the mock provider remains healthy but silent.
    with monkeypatch.context() as patch:
        patch.setattr(loop, "time", lambda: clock() + 3600)
        for _ in range(5):
            await asyncio.sleep(0)
        assert not task.done()
    release.set()
    await task
    if streaming:
        assert closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_explicit_cancellation_propagates_without_retry(streaming):
    provider, request, entered, _release, closed, call = setup_call(streaming)
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert call.call_count == 1
    if streaming:
        assert closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_explicit_deadline_still_stops_model_work(monkeypatch, streaming):
    provider, request, entered, _release, _closed, _call = setup_call(
        streaming, {"timeout": 10}
    )
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    loop = asyncio.get_running_loop()
    clock = loop.time
    with monkeypatch.context() as patch:
        patch.setattr(loop, "time", lambda: clock() + 11)
        with pytest.raises(LLMTimeoutError):
            await task
    assert provider.timeout == 10  # Stream was not established yet.


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_transport_failure_is_still_reported(streaming):
    provider, request, entered, release, closed, call = setup_call(
        streaming, failure=ConnectionError("transport disconnected")
    )
    task = asyncio.create_task(provider.complete(request))
    await entered.wait()
    release.set()
    with pytest.raises(LLMError):
        await task
    assert call.call_count == 1
    if streaming:
        assert closed


@pytest.mark.parametrize("timeout", [None, 30])
def test_sdk_read_policy_preserves_explicit_stream_timeout(timeout):
    provider = OllamaProvider(host="http://test.invalid", config={"timeout": timeout})
    sdk_timeout = provider.client._client.timeout
    assert sdk_timeout.read == timeout
    assert sdk_timeout.connect == sdk_timeout.pool == 5
    assert provider.get_info().defaults["timeout"] is None
