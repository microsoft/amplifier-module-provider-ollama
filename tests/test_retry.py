"""Tests for RetryConfig integration and retry behavior in the Ollama provider."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from amplifier_core import (
    InvalidRequestError,
    NotFoundError,
    ProviderUnavailableError,
)
from amplifier_core.utils.retry import RetryConfig
from ollama import ResponseError  # pyright: ignore[reportAttributeAccessIssue]

from amplifier_module_provider_ollama import OllamaProvider


# ── TestRetryConfigAttribute ────────────────────────────────────────────────────


class TestRetryConfigAttribute:
    """Verify that OllamaProvider constructs a RetryConfig from config."""

    def test_retry_config_exists_and_is_retry_config(self, make_provider):
        provider = make_provider()
        assert hasattr(provider, "_retry_config")
        assert isinstance(provider._retry_config, RetryConfig)

    def test_retry_config_values_from_config(self, make_provider):
        provider = make_provider(
            max_retries=7,
            min_retry_delay=2.0,
            max_retry_delay=120.0,
            retry_jitter=True,
        )
        assert provider._retry_config.max_retries == 7
        assert provider._retry_config.initial_delay == 2.0
        assert provider._retry_config.max_delay == 120.0
        assert provider._retry_config.jitter == 0.2  # True -> 0.2 via Rust getter

    def test_retry_config_defaults(self, make_provider):
        provider = make_provider()
        assert provider._retry_config.max_retries == 3
        assert provider._retry_config.initial_delay == 1.0
        assert provider._retry_config.max_delay == 60.0
        assert provider._retry_config.jitter == 0.2  # default True -> 0.2

    def test_retry_config_jitter_bool_passthrough(self, make_provider):
        """jitter=bool is passed directly to RetryConfig, no float compat."""
        p1 = make_provider(retry_jitter=True)
        assert p1._retry_config.jitter == 0.2  # Rust getter: True -> 0.2

        p2 = make_provider(retry_jitter=False)
        assert p2._retry_config.jitter == 0.0  # Rust getter: False -> 0.0

    def test_no_jitter_float_compat_code(self):
        """Verify the jitter bool/float compat code has been removed from source."""
        import inspect

        source = inspect.getsource(OllamaProvider.__init__)
        assert "raw_jitter" not in source
        assert "isinstance(raw_jitter" not in source

    def test_uses_initial_delay_not_min_delay(self):
        """RetryConfig construction uses initial_delay=, not min_delay=."""
        import inspect

        source = inspect.getsource(OllamaProvider.__init__)
        assert "initial_delay=" in source
        assert "min_delay=" not in source


# ── TestOldRetryCodeRemoved ─────────────────────────────────────────────────────


class TestOldRetryCodeRemoved:
    """Verify that legacy retry constants and methods are removed."""

    def test_no_max_retries_class_var(self):
        assert not hasattr(OllamaProvider, "MAX_RETRIES")

    def test_no_base_retry_delay_class_var(self):
        assert not hasattr(OllamaProvider, "BASE_RETRY_DELAY")

    def test_no_private_retry_with_backoff_method(self, make_provider):
        provider = make_provider()
        assert not hasattr(provider, "_retry_with_backoff")


# ── TestRetryBehavior ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRetryBehavior:
    """Verify retry behavior through the complete() call path."""

    async def test_5xx_error_is_retried(
        self, make_provider, simple_request, mock_response
    ):
        """THE PRIMARY BUG FIX: 500 ResponseError should be retried."""
        provider = make_provider(max_retries=2)
        err = ResponseError("internal server error")
        err.status_code = 500
        provider.client.chat = AsyncMock(side_effect=[err, mock_response()])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await provider.complete(simple_request(metadata={"stream": False}))

        assert result is not None
        assert provider.client.chat.await_count == 2

    async def test_5xx_exhausted_raises_provider_unavailable(
        self, make_provider, simple_request
    ):
        """After exhausting retries on 500, raises ProviderUnavailableError."""
        provider = make_provider(max_retries=1)
        err = ResponseError("internal server error")
        err.status_code = 500
        provider.client.chat = AsyncMock(side_effect=err)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ProviderUnavailableError) as exc_info:
                await provider.complete(simple_request())

        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 500
        assert provider.client.chat.await_count == 2

    async def test_400_error_not_retried(self, make_provider, simple_request):
        """400 errors should raise immediately without retry."""
        provider = make_provider(max_retries=3)
        err = ResponseError("bad request")
        err.status_code = 400
        provider.client.chat = AsyncMock(side_effect=err)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(InvalidRequestError):
                await provider.complete(simple_request())

        assert provider.client.chat.await_count == 1

    async def test_404_error_not_retried(self, make_provider, simple_request):
        """404 errors should raise immediately without retry."""
        provider = make_provider(max_retries=3)
        err = ResponseError("model not found")
        err.status_code = 404
        provider.client.chat = AsyncMock(side_effect=err)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(NotFoundError):
                await provider.complete(simple_request())

        assert provider.client.chat.await_count == 1

    async def test_429_error_is_retried(
        self, make_provider, simple_request, mock_response
    ):
        """429 (rate limit) should be retried."""
        provider = make_provider(max_retries=2)
        err = ResponseError("rate limit exceeded")
        err.status_code = 429
        provider.client.chat = AsyncMock(side_effect=[err, mock_response()])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await provider.complete(simple_request(metadata={"stream": False}))

        assert result is not None
        assert provider.client.chat.await_count == 2

    async def test_timeout_error_is_retried(
        self, make_provider, simple_request, mock_response
    ):
        """TimeoutError should be retried."""
        provider = make_provider(max_retries=2)
        provider.client.chat = AsyncMock(
            side_effect=[asyncio.TimeoutError(), mock_response()]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await provider.complete(simple_request(metadata={"stream": False}))

        assert result is not None
        assert provider.client.chat.await_count == 2

    async def test_connection_error_is_retried(
        self, make_provider, simple_request, mock_response
    ):
        """ConnectionError should be retried."""
        provider = make_provider(max_retries=2)
        provider.client.chat = AsyncMock(
            side_effect=[ConnectionError("refused"), mock_response()]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await provider.complete(simple_request(metadata={"stream": False}))

        assert result is not None
        assert provider.client.chat.await_count == 2


# ── TestRetryEventEmission ──────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRetryEventEmission:
    """Verify that provider:retry events are emitted on retry."""

    async def test_provider_retry_event_emitted(
        self, make_provider, simple_request, mock_response
    ):
        """A single retry should emit one provider:retry event."""
        provider = make_provider(max_retries=2)
        err = ResponseError("internal server error")
        err.status_code = 500
        provider.client.chat = AsyncMock(side_effect=[err, mock_response()])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await provider.complete(simple_request(metadata={"stream": False}))

        hooks = provider.coordinator.hooks  # type: ignore[union-attr]
        retry_events = [(n, p) for n, p in hooks.events if n == "provider:retry"]
        assert len(retry_events) == 1

        _, payload = retry_events[0]
        assert payload["provider"] == "ollama"
        assert payload["attempt"] == 1
        assert payload["max_retries"] == 2
        assert "delay" in payload
        assert payload["error_type"] == "ProviderUnavailableError"
        assert "error_message" in payload

    async def test_multiple_retry_events(
        self, make_provider, simple_request, mock_response
    ):
        """Multiple retries should emit incrementing attempt events."""
        provider = make_provider(max_retries=4)
        err = ResponseError("internal server error")
        err.status_code = 500
        provider.client.chat = AsyncMock(side_effect=[err, err, err, mock_response()])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await provider.complete(simple_request(metadata={"stream": False}))

        hooks = provider.coordinator.hooks  # type: ignore[union-attr]
        retry_events = [(n, p) for n, p in hooks.events if n == "provider:retry"]
        assert len(retry_events) == 3

        attempts = [p["attempt"] for _, p in retry_events]
        assert attempts == [1, 2, 3]


# ── TestStreamingRetryBehavior ──────────────────────────────────────────────────


@pytest.mark.asyncio
class TestStreamingRetryBehavior:
    """Verify retry behavior through the streaming complete() path."""

    @staticmethod
    async def _fake_stream(content="hi"):
        """Create a simple async iterator yielding two chunks."""
        yield {"message": {"content": content}, "done": False}
        yield {
            "message": {"content": ""},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 2,
            "model": "llama3.2:3b",
        }

    async def test_streaming_5xx_is_retried(self, make_provider, simple_request):
        """500 on stream connect fails once then succeeds."""
        provider = make_provider(max_retries=2)
        err = ResponseError("internal server error")
        err.status_code = 500
        provider.client.chat = AsyncMock(side_effect=[err, self._fake_stream()])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await provider.complete(simple_request(stream=True))

        assert result is not None
        assert provider.client.chat.await_count == 2

    async def test_streaming_400_not_retried(self, make_provider, simple_request):
        """400 on stream connect raises InvalidRequestError immediately."""
        provider = make_provider(max_retries=3)
        err = ResponseError("bad request")
        err.status_code = 400
        provider.client.chat = AsyncMock(side_effect=err)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(InvalidRequestError):
                await provider.complete(simple_request(stream=True))

        assert provider.client.chat.await_count == 1

    async def test_streaming_connection_error_retried(
        self, make_provider, simple_request
    ):
        """ConnectionError fails once then succeeds."""
        provider = make_provider(max_retries=2)
        provider.client.chat = AsyncMock(
            side_effect=[ConnectionError("refused"), self._fake_stream()]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await provider.complete(simple_request(stream=True))

        assert result is not None
        assert provider.client.chat.await_count == 2

    async def test_streaming_retry_event_emitted(self, make_provider, simple_request):
        """500 fails once then succeeds, asserts 1 provider:retry event."""
        provider = make_provider(max_retries=2)
        err = ResponseError("internal server error")
        err.status_code = 500
        provider.client.chat = AsyncMock(side_effect=[err, self._fake_stream()])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await provider.complete(simple_request(stream=True))

        hooks = provider.coordinator.hooks  # type: ignore[union-attr]
        retry_events = [(n, p) for n, p in hooks.events if n == "provider:retry"]
        assert len(retry_events) == 1

        _, payload = retry_events[0]
        assert payload["provider"] == "ollama"
        assert payload["attempt"] == 1
        assert payload["max_retries"] == 2
        assert "delay" in payload
        assert payload["error_type"] == "ProviderUnavailableError"
        assert "error_message" in payload
