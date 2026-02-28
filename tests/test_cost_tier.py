"""Tests for cost tier metadata in Ollama provider.

Validates:
1. list_models() adds cost_per_input/output_token = 0.0 (genuinely free)
2. list_models() adds metadata={"cost_tier": "free"} to all models
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_ollama import OllamaProvider


class TestOllamaCostTier:
    """Verify all Ollama models get free cost tier and zero cost."""

    @pytest.fixture
    def provider(self):
        return OllamaProvider(host="http://localhost:11434")

    def _make_mock_model(self, model_name: str):
        m = MagicMock()
        m.model = model_name
        m.details = MagicMock()
        m.details.context_length = 8192
        return m

    @pytest.mark.asyncio
    async def test_model_has_free_cost_tier(self, provider):
        mock_response = MagicMock()
        mock_response.models = [self._make_mock_model("llama3.2:3b")]

        provider._client = MagicMock()
        provider._client.list = AsyncMock(return_value=mock_response)

        models = await provider.list_models()

        assert len(models) == 1
        model = models[0]
        assert model.metadata == {"cost_tier": "free"}

    @pytest.mark.asyncio
    async def test_model_has_zero_cost(self, provider):
        mock_response = MagicMock()
        mock_response.models = [self._make_mock_model("deepseek-r1:14b")]

        provider._client = MagicMock()
        provider._client.list = AsyncMock(return_value=mock_response)

        models = await provider.list_models()

        assert len(models) == 1
        model = models[0]
        assert model.cost_per_input_token == 0.0
        assert model.cost_per_output_token == 0.0

    @pytest.mark.asyncio
    async def test_all_models_are_free(self, provider):
        """Every model from a local Ollama server should be free."""
        mock_response = MagicMock()
        mock_response.models = [
            self._make_mock_model("llama3.2:3b"),
            self._make_mock_model("deepseek-r1:14b"),
            self._make_mock_model("qwen3:8b"),
        ]

        provider._client = MagicMock()
        provider._client.list = AsyncMock(return_value=mock_response)

        models = await provider.list_models()

        assert len(models) == 3
        for model in models:
            assert model.metadata["cost_tier"] == "free"
            assert model.cost_per_input_token == 0.0
            assert model.cost_per_output_token == 0.0
