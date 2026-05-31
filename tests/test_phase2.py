"""Tests for Phase 2: error translation, reasoning_effort, and usage notes."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from amplifier_core import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    LLMError,
    LLMTimeoutError,
    NotFoundError,
    ProviderUnavailableError,
    RateLimitError,
)
from ollama import ResponseError  # pyright: ignore[reportAttributeAccessIssue]

from amplifier_module_provider_ollama import _translate_ollama_error


# ── _translate_ollama_error unit tests ───────────────────────────────────


class TestTranslateOllamaError:
    """Unit tests for the standalone error translation helper."""

    def test_response_error_401(self):
        err = ResponseError("unauthorized")
        err.status_code = 401
        result = _translate_ollama_error(err)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "ollama"
        assert result.status_code == 401

    def test_response_error_403(self):
        err = ResponseError("forbidden")
        err.status_code = 403
        result = _translate_ollama_error(err)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "ollama"
        assert result.status_code == 403

    def test_response_error_429(self):
        err = ResponseError("rate limit exceeded")
        err.status_code = 429
        result = _translate_ollama_error(err)
        assert isinstance(result, RateLimitError)
        assert result.provider == "ollama"
        assert result.status_code == 429
        assert result.retryable is True

    def test_response_error_400(self):
        err = ResponseError("bad request")
        err.status_code = 400
        result = _translate_ollama_error(err)
        assert isinstance(result, InvalidRequestError)
        assert result.provider == "ollama"
        assert result.status_code == 400

    def test_response_error_400_context_length(self):
        err = ResponseError("context length exceeded")
        err.status_code = 400
        result = _translate_ollama_error(err)
        assert isinstance(result, ContextLengthError)
        assert result.provider == "ollama"
        assert result.status_code == 400

    def test_response_error_400_content_filter(self):
        err = ResponseError("content blocked by safety filter")
        err.status_code = 400
        result = _translate_ollama_error(err)
        assert isinstance(result, ContentFilterError)
        assert result.provider == "ollama"
        assert result.status_code == 400

    def test_response_error_404(self):
        err = ResponseError("model not found")
        err.status_code = 404
        result = _translate_ollama_error(err)
        assert isinstance(result, NotFoundError)
        assert result.provider == "ollama"
        assert result.status_code == 404
        assert result.retryable is False

    def test_response_error_404_is_not_retryable(self):
        err = ResponseError("not found")
        err.status_code = 404
        result = _translate_ollama_error(err)
        assert result.retryable is False

    def test_response_error_500(self):
        err = ResponseError("internal server error")
        err.status_code = 500
        result = _translate_ollama_error(err)
        assert isinstance(result, ProviderUnavailableError)
        assert result.provider == "ollama"
        assert result.status_code == 500

    def test_response_error_503(self):
        err = ResponseError("service unavailable")
        err.status_code = 503
        result = _translate_ollama_error(err)
        assert isinstance(result, ProviderUnavailableError)
        assert result.status_code == 503

    def test_response_error_other_status(self):
        err = ResponseError("something else")
        err.status_code = 418
        result = _translate_ollama_error(err)
        assert isinstance(result, LLMError)
        assert result.retryable is True
        assert result.provider == "ollama"

    def test_connection_error(self):
        result = _translate_ollama_error(ConnectionError("refused"))
        assert isinstance(result, ProviderUnavailableError)
        assert result.retryable is True
        assert result.provider == "ollama"

    def test_os_error(self):
        result = _translate_ollama_error(OSError("network down"))
        assert isinstance(result, ProviderUnavailableError)
        assert result.retryable is True

    def test_timeout_error(self):
        result = _translate_ollama_error(asyncio.TimeoutError())
        assert isinstance(result, LLMTimeoutError)
        assert result.provider == "ollama"

    def test_timeout_error_is_retryable(self):
        result = _translate_ollama_error(asyncio.TimeoutError())
        assert isinstance(result, LLMTimeoutError)
        assert result.retryable is True

    def test_generic_exception(self):
        result = _translate_ollama_error(RuntimeError("boom"))
        assert isinstance(result, LLMError)
        assert result.retryable is True
        assert result.provider == "ollama"


# ── Error translation integration (through complete()) ──────────────────


@pytest.mark.asyncio
class TestErrorTranslationIntegration:
    """Verify that complete() raises translated kernel errors."""

    async def test_timeout_raises_llm_timeout_error(
        self, make_provider, simple_request
    ):
        provider = make_provider()
        provider.client.chat = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(LLMTimeoutError) as exc_info:
                await provider.complete(simple_request())

        assert exc_info.value.provider == "ollama"
        assert exc_info.value.__cause__ is not None

    async def test_timeout_error_is_retryable_through_complete(
        self, make_provider, simple_request
    ):
        provider = make_provider()
        provider.client.chat = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(LLMTimeoutError) as exc_info:
                await provider.complete(simple_request())

        assert exc_info.value.retryable is True

    async def test_response_error_401_raises_authentication_error(
        self, make_provider, simple_request
    ):
        provider = make_provider()
        err = ResponseError("unauthorized")
        err.status_code = 401
        provider.client.chat = AsyncMock(side_effect=err)

        with pytest.raises(AuthenticationError) as exc_info:
            await provider.complete(simple_request())

        assert exc_info.value.provider == "ollama"
        assert exc_info.value.__cause__ is err

    async def test_response_error_400_raises_invalid_request(
        self, make_provider, simple_request
    ):
        provider = make_provider()
        err = ResponseError("bad request")
        err.status_code = 400
        provider.client.chat = AsyncMock(side_effect=err)

        with pytest.raises(InvalidRequestError):
            await provider.complete(simple_request())

    async def test_response_error_500_raises_provider_unavailable(
        self, make_provider, simple_request
    ):
        provider = make_provider()
        err = ResponseError("server error")
        err.status_code = 500
        provider.client.chat = AsyncMock(side_effect=err)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ProviderUnavailableError) as exc_info:
                await provider.complete(simple_request())

        assert exc_info.value.status_code == 500

    async def test_response_error_404_raises_not_found(
        self, make_provider, simple_request
    ):
        provider = make_provider()
        err = ResponseError("model not found")
        err.status_code = 404
        provider.client.chat = AsyncMock(side_effect=err)

        with pytest.raises(NotFoundError) as exc_info:
            await provider.complete(simple_request())

        assert exc_info.value.status_code == 404
        assert exc_info.value.retryable is False
        assert exc_info.value.__cause__ is err

    async def test_connection_error_after_retry_raises_provider_unavailable(
        self, make_provider, simple_request
    ):
        """ConnectionError is retried by retry_with_backoff, then translated."""
        provider = make_provider()
        # retry_with_backoff retries 3 times; all fail → raises last error
        provider.client.chat = AsyncMock(side_effect=ConnectionError("refused"))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ProviderUnavailableError) as exc_info:
                await provider.complete(simple_request())

        assert exc_info.value.retryable is True

    async def test_cause_chain_preserved(self, make_provider, simple_request):
        provider = make_provider()
        original = ResponseError("original")
        original.status_code = 400
        provider.client.chat = AsyncMock(side_effect=original)

        with pytest.raises(InvalidRequestError) as exc_info:
            await provider.complete(simple_request())

        assert exc_info.value.__cause__ is original

    async def test_streaming_timeout_raises_llm_timeout_error(
        self, make_provider, simple_request
    ):
        provider = make_provider()
        provider.client.chat = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(LLMTimeoutError) as exc_info:
                await provider.complete(simple_request(stream=True))

        assert exc_info.value.provider == "ollama"

    async def test_streaming_response_error_raises_translated(
        self, make_provider, simple_request
    ):
        provider = make_provider()
        err = ResponseError("forbidden")
        err.status_code = 403
        provider.client.chat = AsyncMock(side_effect=err)

        with pytest.raises(AuthenticationError):
            await provider.complete(simple_request(stream=True))


# ── reasoning_effort support ────────────────────────────────────────────


@pytest.mark.asyncio
class TestReasoningEffort:
    """Verify reasoning_effort on ChatRequest enables thinking."""

    async def test_reasoning_effort_enables_thinking_for_thinking_model(
        self, make_provider, simple_request, mock_response
    ):
        """Non-None reasoning_effort should pass effort level to think param."""
        provider = make_provider(default_model="deepseek-r1:14b", enable_thinking=False)
        provider.client.chat = AsyncMock(return_value=mock_response())

        request = simple_request(reasoning_effort="high", metadata={"stream": False})
        await provider.complete(request, model="deepseek-r1:14b")

        call_kwargs = provider.client.chat.call_args
        # Ollama v0.9.0+ supports effort levels — value is passed through directly
        assert (
            call_kwargs.kwargs.get("think") == "high"
            or call_kwargs[1].get("think") == "high"
        )

    async def test_reasoning_effort_ignored_for_non_thinking_model(
        self, make_provider, simple_request, mock_response
    ):
        """reasoning_effort should have no effect on non-thinking models."""
        provider = make_provider(default_model="llama3.2:3b", enable_thinking=False)
        provider.client.chat = AsyncMock(return_value=mock_response())

        request = simple_request(reasoning_effort="medium", metadata={"stream": False})
        await provider.complete(request, model="llama3.2:3b")

        call_kwargs = provider.client.chat.call_args
        # think should not be in the params at all
        assert "think" not in (call_kwargs.kwargs or {})

    async def test_reasoning_effort_none_falls_through_to_config(
        self, make_provider, simple_request, mock_response
    ):
        """When reasoning_effort is None, existing config controls thinking."""
        provider = make_provider(
            default_model="qwen3:8b",
            enable_thinking=True,
            thinking_effort="low",
        )
        provider.client.chat = AsyncMock(return_value=mock_response())

        request = simple_request(metadata={"stream": False})  # reasoning_effort defaults to None
        await provider.complete(request, model="qwen3:8b")

        call_kwargs = provider.client.chat.call_args
        # Should use config effort "low" (not True from reasoning_effort)
        assert (
            call_kwargs.kwargs.get("think") == "low"
            or call_kwargs[1].get("think") == "low"
        )

    async def test_enable_thinking_takes_precedence_over_reasoning_effort(
        self, make_provider, simple_request, mock_response
    ):
        """request.enable_thinking (kwargs path) has higher priority."""
        provider = make_provider(
            default_model="qwen3:8b",
            enable_thinking=False,
            thinking_effort="high",
        )
        provider.client.chat = AsyncMock(return_value=mock_response())

        # Simulate enable_thinking on request (existing kwargs path)
        request = simple_request(reasoning_effort="medium", metadata={"stream": False})
        # Manually set enable_thinking to test precedence
        request.enable_thinking = True  # type: ignore[attr-defined]
        await provider.complete(request, model="qwen3:8b")

        call_kwargs = provider.client.chat.call_args
        # Should use config's thinking_effort "high" (from enable_thinking path)
        assert (
            call_kwargs.kwargs.get("think") == "high"
            or call_kwargs[1].get("think") == "high"
        )

    async def test_reasoning_effort_low_passes_through(
        self, make_provider, simple_request, mock_response
    ):
        """'low' reasoning_effort is passed through as effort level."""
        provider = make_provider(default_model="deepseek-r1:14b", enable_thinking=False)
        provider.client.chat = AsyncMock(return_value=mock_response())

        request = simple_request(reasoning_effort="low", metadata={"stream": False})
        await provider.complete(request, model="deepseek-r1:14b")

        call_kwargs = provider.client.chat.call_args
        # Ollama v0.9.0+ supports effort levels — value is passed through directly
        assert (
            call_kwargs.kwargs.get("think") == "low"
            or call_kwargs[1].get("think") == "low"
        )

    async def test_streaming_reasoning_effort_passes_through(
        self, make_provider, simple_request
    ):
        """reasoning_effort should pass effort level through in streaming path."""
        provider = make_provider(default_model="deepseek-r1:14b", enable_thinking=False)

        # Create an async iterator for streaming
        async def fake_stream():
            yield {"message": {"content": "hi"}, "done": False}
            yield {
                "message": {"content": ""},
                "done": True,
                "prompt_eval_count": 5,
                "eval_count": 2,
                "model": "deepseek-r1:14b",
            }

        provider.client.chat = AsyncMock(return_value=fake_stream())

        request = simple_request(stream=True, reasoning_effort="high")
        await provider.complete(request, model="deepseek-r1:14b")

        call_kwargs = provider.client.chat.call_args
        # Ollama v0.9.0+ supports effort levels — value is passed through directly
        assert (
            call_kwargs.kwargs.get("think") == "high"
            or call_kwargs[1].get("think") == "high"
        )
