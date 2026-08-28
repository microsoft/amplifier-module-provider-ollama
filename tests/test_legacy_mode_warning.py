"""Tests for the mount-time warning on the removed legacy `mode` config key.

`mode` was removed in favor of host-based derivation (see test_cloud_auth.py
module docstring): `host` is now the single source of truth for
local-vs-cloud. A config that still carries a `mode` key is otherwise
silently ignored -- mount() should warn loudly, once, naming the key and
the replacement guidance, rather than silently discarding it.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_module_provider_ollama import mount


def _fake_coordinator() -> MagicMock:
    coord = MagicMock()
    coord.mount = AsyncMock()
    return coord


@pytest.mark.asyncio(loop_scope="function")
async def test_mount_warns_when_legacy_mode_key_present(caplog):
    """A legacy `mode` key in config should trigger a loud warning at mount."""
    coordinator = _fake_coordinator()
    with (
        patch("amplifier_module_provider_ollama.OllamaProvider._check_connection"),
        caplog.at_level(logging.WARNING),
    ):
        await mount(coordinator=coordinator, config={"mode": "local"})

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("mode" in r.message and "ignored" in r.message for r in warnings), (
        f"Expected a warning naming the ignored 'mode' key, got: "
        f"{[r.message for r in warnings]}"
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_mount_silent_when_mode_key_absent(caplog):
    """No `mode` key -> no warning about it (normal, silent config)."""
    coordinator = _fake_coordinator()
    with (
        patch("amplifier_module_provider_ollama.OllamaProvider._check_connection"),
        caplog.at_level(logging.WARNING),
    ):
        await mount(coordinator=coordinator, config={"host": "http://localhost:11434"})

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("mode" in r.message for r in warnings), (
        f"Did not expect a 'mode'-related warning, got: {[r.message for r in warnings]}"
    )
