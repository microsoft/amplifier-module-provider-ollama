"""Behavioral tests for ollama provider.

Inherits authoritative tests from amplifier-core.
"""

from amplifier_core.validation.behavioral import ProviderBehaviorTests
import httpx
from ollama import AsyncClient
import pytest


class TestOllamaProviderBehavior(ProviderBehaviorTests):
    """Run standard provider behavioral tests for ollama.

    All tests from ProviderBehaviorTests run automatically.
    Add module-specific tests below if needed.
    """

    @pytest.mark.asyncio
    async def test_list_models_returns_list(self, provider_module):
        """The inherited fixture needs an available endpoint, not silent fallback."""
        original = provider_module._client
        client = AsyncClient(
            host="https://ollama.invalid",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"models": [{"model": "actual:Q4"}]}
                )
            ),
        )
        provider_module._client = client
        try:
            await super().test_list_models_returns_list(provider_module)
        finally:
            provider_module._client = original
            await client._client.aclose()
