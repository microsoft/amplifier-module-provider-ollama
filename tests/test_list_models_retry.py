"""Retry behavior tests for list_models().

Verifies that list_models() uses the same shared retry_with_backoff()/
_retry_config machinery as complete(): transient failures (connection
errors, timeouts, 5xx) are retried with backoff via the shared
_translate_ollama_error() classification, and non-retryable failures
(e.g. 401) skip retries.

Unlike the sibling fixes in provider-openai (PR #61), provider-anthropic
(PR #90), and provider-gemini (PR #39) -- which all *raise* to the caller
once retries are exhausted -- this module's list_models() has a
deliberate soft-failure contract: it never raises, it degrades to an
empty list. That contract is preserved here; retry is added *before* the
degrade, not instead of it. The key regression this guards is a
transient blip no longer causing an immediate (un-retried) empty-list
return.

See test_retry.py for the equivalent tests on the complete() path --
this file mirrors that call shape for list_models().
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from ollama import ResponseError  # pyright: ignore[reportAttributeAccessIssue]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_model(model_id: str, context_length: int = 8192) -> SimpleNamespace:
    """Create a fake ollama Model object (response.models entries)."""
    return SimpleNamespace(
        model=model_id,
        details=SimpleNamespace(context_length=context_length),
    )


def _fake_list_response(model_ids: list[str]) -> SimpleNamespace:
    """Create a fake ollama client.list() response."""
    return SimpleNamespace(models=[_fake_model(mid) for mid in model_ids])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestListModelsRetry:
    async def test_list_models_succeeds_first_try(self, make_provider):
        """No transient failure: exactly one API call, result unchanged."""
        provider = make_provider()
        provider.client.list = AsyncMock(
            return_value=_fake_list_response(["llama3.2:3b"])
        )

        with pytest.MonkeyPatch.context() as mp:
            mock_sleep = AsyncMock()
            mp.setattr(asyncio, "sleep", mock_sleep)
            models = await provider.list_models()

            assert provider.client.list.await_count == 1
            mock_sleep.assert_not_awaited()

        assert len(models) == 1
        assert models[0].id == "llama3.2:3b"

    async def test_list_models_recovers_from_transient_connection_error(
        self, make_provider
    ):
        """THE KEY REGRESSION: a single transient ConnectionError is retried,
        then the call succeeds -- the degraded [] fallback must NOT be
        returned when the retry succeeds.
        """
        provider = make_provider(max_retries=2)
        provider.client.list = AsyncMock(
            side_effect=[
                ConnectionError("connection refused"),
                _fake_list_response(["llama3.2:3b"]),
            ]
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(asyncio, "sleep", AsyncMock())
            models = await provider.list_models()

        assert provider.client.list.await_count == 2
        # Degraded fallback NOT returned -- real models came back.
        assert len(models) == 1
        assert models[0].id == "llama3.2:3b"

    async def test_list_models_recovers_from_transient_500(self, make_provider):
        """A transient 5xx ResponseError is also retryable and recovers."""
        provider = make_provider(max_retries=2)
        err = ResponseError("internal server error")
        err.status_code = 500
        provider.client.list = AsyncMock(
            side_effect=[err, _fake_list_response(["llama3.2:3b"])]
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(asyncio, "sleep", AsyncMock())
            models = await provider.list_models()

        assert provider.client.list.await_count == 2
        assert len(models) == 1
        assert models[0].id == "llama3.2:3b"

    async def test_list_models_exhaustion_returns_degraded_fallback(
        self, make_provider, caplog
    ):
        """Persistent transient failure exhausts retries, then degrades to
        [] (not a raise) with an unmistakable WARNING naming the attempt
        count and the degraded return.
        """
        provider = make_provider(max_retries=2)
        provider.client.list = AsyncMock(
            side_effect=ConnectionError("connection refused")
        )

        with (
            pytest.MonkeyPatch.context() as mp,
            caplog.at_level(logging.WARNING),
        ):
            mp.setattr(asyncio, "sleep", AsyncMock())
            models = await provider.list_models()

        # 1 initial + 2 retries = 3 total attempts
        assert provider.client.list.await_count == 3
        # Soft-failure contract preserved: degrades, does not raise.
        assert models == []

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("list_models" in r.getMessage() for r in warnings)
        assert any("3 attempt" in r.getMessage() for r in warnings)
        assert any(
            "DEGRADED" in r.getMessage() or "empty" in r.getMessage().lower()
            for r in warnings
        )

    async def test_list_models_non_retryable_error_skips_retries(
        self, make_provider, caplog
    ):
        """A non-retryable error (401) is not retried -- one attempt only --
        but still degrades to [] rather than raising, per this module's
        soft-failure contract.
        """
        provider = make_provider(max_retries=3)
        err = ResponseError("invalid API key")
        err.status_code = 401
        provider.client.list = AsyncMock(side_effect=err)

        with (
            pytest.MonkeyPatch.context() as mp,
            caplog.at_level(logging.WARNING),
        ):
            mock_sleep = AsyncMock()
            mp.setattr(asyncio, "sleep", mock_sleep)
            models = await provider.list_models()

            assert provider.client.list.await_count == 1
            mock_sleep.assert_not_awaited()

        assert models == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("list_models" in r.getMessage() for r in warnings)
        assert any("1 attempt" in r.getMessage() for r in warnings)
