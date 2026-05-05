"""Tests for Ollama Cloud authentication support.

Verifies that when an api_key is supplied, the underlying AsyncClient is
constructed with an Authorization: Bearer <key> header, and that local-only
usage (no api_key) preserves the prior unauthenticated behavior.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_module_provider_ollama import OllamaProvider


def test_init_without_api_key_has_no_headers():
    """Local Ollama use (no api_key) should not synthesize any headers."""
    provider = OllamaProvider(host="http://localhost:11434")
    assert provider._api_key is None
    assert provider._headers is None


def test_init_with_api_key_builds_bearer_header():
    """Cloud use should build an Authorization: Bearer <key> header dict."""
    provider = OllamaProvider(host="https://ollama.com", api_key="secret-key")
    assert provider._api_key == "secret-key"
    assert provider._headers == {"Authorization": "Bearer secret-key"}


def test_is_cloud_property():
    """`is_cloud` should detect ollama.com hosts and ignore local hosts."""
    cloud = OllamaProvider(host="https://ollama.com", api_key="k")
    local = OllamaProvider(host="http://localhost:11434")
    custom = OllamaProvider(host="https://my-internal.example/")
    assert cloud.is_cloud is True
    assert local.is_cloud is False
    assert custom.is_cloud is False


def test_is_cloud_handles_subdomains_and_ports():
    """Subdomains of ollama.com should match; port suffix shouldn't break detection."""
    sub = OllamaProvider(host="https://staging.ollama.com")
    with_port = OllamaProvider(host="https://ollama.com:443")
    assert sub.is_cloud is True
    assert with_port.is_cloud is True


def test_is_cloud_rejects_lookalike_hosts():
    """is_cloud must NOT match attacker-controlled lookalike hostnames."""
    # Substring-match would falsely accept these; urlparse-based check rejects them.
    evil = OllamaProvider(host="http://evil.ollama.com.attacker.io")
    path_lookalike = OllamaProvider(host="http://localhost/ollama.com")
    suffix_lookalike = OllamaProvider(host="https://notollama.com")
    empty_host = OllamaProvider(host="")
    assert evil.is_cloud is False
    assert path_lookalike.is_cloud is False
    assert suffix_lookalike.is_cloud is False
    assert empty_host.is_cloud is False


def test_client_property_passes_headers_to_async_client():
    """When api_key is set, AsyncClient must be constructed with the headers dict."""
    provider = OllamaProvider(host="https://ollama.com", api_key="abc123")
    with patch("amplifier_module_provider_ollama.AsyncClient") as mock_client_cls:
        _ = provider.client
        mock_client_cls.assert_called_once_with(
            host="https://ollama.com",
            headers={"Authorization": "Bearer abc123"},
        )


def test_client_property_passes_none_headers_for_local():
    """Local use must pass headers=None so behavior is unchanged."""
    provider = OllamaProvider(host="http://localhost:11434")
    with patch("amplifier_module_provider_ollama.AsyncClient") as mock_client_cls:
        _ = provider.client
        mock_client_cls.assert_called_once_with(
            host="http://localhost:11434",
            headers=None,
        )


def _fake_coordinator() -> MagicMock:
    """Build a minimal coordinator with an awaitable .mount() method."""
    coord = MagicMock()
    coord.mount = AsyncMock()
    return coord


@pytest.mark.asyncio(loop_scope="function")
async def test_mount_reads_api_key_from_env(monkeypatch):
    """mount() should pick up OLLAMA_API_KEY from the environment."""
    from amplifier_module_provider_ollama import mount

    monkeypatch.setenv("OLLAMA_API_KEY", "env-key")
    monkeypatch.setenv("OLLAMA_HOST", "https://ollama.com")

    coordinator = _fake_coordinator()
    # Skip the network connection check — we only care that the provider is
    # constructed with the right api_key.
    with patch("amplifier_module_provider_ollama.OllamaProvider._check_connection"):
        await mount(coordinator=coordinator, config={})

    coordinator.mount.assert_awaited_once()
    provider = coordinator.mount.await_args.args[1]
    assert isinstance(provider, OllamaProvider)
    assert provider._api_key == "env-key"
    assert provider._headers == {"Authorization": "Bearer env-key"}


@pytest.mark.asyncio(loop_scope="function")
async def test_mount_config_overrides_env(monkeypatch):
    """An explicit api_key in config should win over OLLAMA_API_KEY."""
    from amplifier_module_provider_ollama import mount

    monkeypatch.setenv("OLLAMA_API_KEY", "env-key")
    monkeypatch.setenv("OLLAMA_HOST", "https://ollama.com")

    coordinator = _fake_coordinator()
    with patch("amplifier_module_provider_ollama.OllamaProvider._check_connection"):
        await mount(coordinator=coordinator, config={"api_key": "config-key"})

    coordinator.mount.assert_awaited_once()
    provider = coordinator.mount.await_args.args[1]
    assert isinstance(provider, OllamaProvider)
    assert provider._api_key == "config-key"


def test_get_info_declares_credential_env_var():
    """get_info() must report OLLAMA_API_KEY as a credential env var."""
    info = OllamaProvider(host="http://localhost:11434").get_info()
    assert "OLLAMA_API_KEY" in info.credential_env_vars


def test_get_info_includes_api_key_config_field():
    """get_info() must expose an api_key ConfigField as a secret, gated to cloud mode."""
    info = OllamaProvider(host="http://localhost:11434").get_info()
    api_key_fields = [f for f in info.config_fields if f.id == "api_key"]
    assert len(api_key_fields) == 1
    field = api_key_fields[0]
    assert field.field_type == "secret"
    assert field.env_var == "OLLAMA_API_KEY"
    # api_key is required, but only shown when mode=cloud (via show_when)
    assert field.required is True
    assert field.show_when == {"mode": "cloud"}


def test_get_info_includes_mode_choice_field():
    """get_info() must expose a `mode` choice field with local/cloud options."""
    info = OllamaProvider(host="http://localhost:11434").get_info()
    mode_fields = [f for f in info.config_fields if f.id == "mode"]
    assert len(mode_fields) == 1
    field = mode_fields[0]
    assert field.field_type == "choice"
    assert field.choices == ["local", "cloud"]
    assert field.default == "local"
    assert field.required is True


def test_get_info_has_two_host_fields_gated_by_mode():
    """Two host ConfigFields exist — one for each mode, gated via show_when."""
    info = OllamaProvider(host="http://localhost:11434").get_info()
    host_fields = [f for f in info.config_fields if f.id == "host"]
    assert len(host_fields) == 2

    local_host = next(f for f in host_fields if f.show_when == {"mode": "local"})
    cloud_host = next(f for f in host_fields if f.show_when == {"mode": "cloud"})

    assert local_host.default == "http://localhost:11434"
    assert cloud_host.default == "https://ollama.com"


def test_get_info_auto_pull_only_in_local_mode():
    """auto_pull must be gated to local mode (cloud doesn't support pull)."""
    info = OllamaProvider(host="http://localhost:11434").get_info()
    auto_pull_fields = [f for f in info.config_fields if f.id == "auto_pull"]
    assert len(auto_pull_fields) == 1
    assert auto_pull_fields[0].show_when == {"mode": "local"}


def test_get_info_capabilities_local_when_local_host():
    """capabilities should advertise 'local' when host is local."""
    info = OllamaProvider(host="http://localhost:11434").get_info()
    assert "local" in info.capabilities
    assert "cloud" not in info.capabilities


def test_get_info_capabilities_cloud_when_cloud_host():
    """capabilities should advertise 'cloud' when host points at Ollama Cloud."""
    info = OllamaProvider(host="https://ollama.com", api_key="k").get_info()
    assert "cloud" in info.capabilities
    assert "local" not in info.capabilities


@pytest.mark.asyncio(loop_scope="function")
async def test_mount_mode_cloud_derives_host(monkeypatch):
    """mount() with mode=cloud and no host should default to https://ollama.com."""
    from amplifier_module_provider_ollama import mount

    # Clear env so we test the mode-driven fallback in isolation
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    coordinator = _fake_coordinator()
    with patch("amplifier_module_provider_ollama.OllamaProvider._check_connection"):
        await mount(coordinator=coordinator, config={"mode": "cloud", "api_key": "k"})

    provider = coordinator.mount.await_args.args[1]
    assert provider.host == "https://ollama.com"
    assert provider.is_cloud is True


@pytest.mark.asyncio(loop_scope="function")
async def test_mount_mode_local_derives_host(monkeypatch):
    """mount() with mode=local and no host should default to http://localhost:11434."""
    from amplifier_module_provider_ollama import mount

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    coordinator = _fake_coordinator()
    with patch("amplifier_module_provider_ollama.OllamaProvider._check_connection"):
        await mount(coordinator=coordinator, config={"mode": "local"})

    provider = coordinator.mount.await_args.args[1]
    assert provider.host == "http://localhost:11434"
    assert provider.is_cloud is False
    assert provider._headers is None


@pytest.mark.asyncio(loop_scope="function")
async def test_mount_explicit_host_overrides_mode(monkeypatch):
    """An explicit host should win over the mode-driven default."""
    from amplifier_module_provider_ollama import mount

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    coordinator = _fake_coordinator()
    with patch("amplifier_module_provider_ollama.OllamaProvider._check_connection"):
        await mount(
            coordinator=coordinator,
            config={
                "mode": "cloud",
                "host": "https://my-proxy.example",
                "api_key": "k",
            },
        )

    provider = coordinator.mount.await_args.args[1]
    assert provider.host == "https://my-proxy.example"


@pytest.mark.asyncio(loop_scope="function")
async def test_mount_no_mode_no_host_defaults_to_localhost(monkeypatch):
    """Existing configs with no mode/host must continue to work (backward compat)."""
    from amplifier_module_provider_ollama import mount

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    coordinator = _fake_coordinator()
    with patch("amplifier_module_provider_ollama.OllamaProvider._check_connection"):
        await mount(coordinator=coordinator, config={})

    provider = coordinator.mount.await_args.args[1]
    assert provider.host == "http://localhost:11434"
    assert provider.is_cloud is False
    assert provider._headers is None


@pytest.mark.asyncio(loop_scope="function")
async def test_mount_mode_cloud_default_model_is_gpt_oss(monkeypatch):
    """mount() with mode=cloud and no default_model should use gpt-oss:120b."""
    from amplifier_module_provider_ollama import mount

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    coordinator = _fake_coordinator()
    with patch("amplifier_module_provider_ollama.OllamaProvider._check_connection"):
        await mount(coordinator=coordinator, config={"mode": "cloud", "api_key": "k"})

    provider = coordinator.mount.await_args.args[1]
    assert provider.default_model == "gpt-oss:120b"


@pytest.mark.asyncio(loop_scope="function")
async def test_mount_mode_local_default_model_is_llama(monkeypatch):
    """mount() with mode=local should keep llama3.2:3b as default."""
    from amplifier_module_provider_ollama import mount

    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    coordinator = _fake_coordinator()
    with patch("amplifier_module_provider_ollama.OllamaProvider._check_connection"):
        await mount(coordinator=coordinator, config={"mode": "local"})

    provider = coordinator.mount.await_args.args[1]
    assert provider.default_model == "llama3.2:3b"


@pytest.mark.asyncio(loop_scope="function")
async def test_mount_explicit_default_model_wins_over_mode(monkeypatch):
    """An explicit default_model in config must override the mode-derived default."""
    from amplifier_module_provider_ollama import mount

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    coordinator = _fake_coordinator()
    with patch("amplifier_module_provider_ollama.OllamaProvider._check_connection"):
        await mount(
            coordinator=coordinator,
            config={
                "mode": "cloud",
                "api_key": "k",
                "default_model": "qwen3-coder-next",
            },
        )

    provider = coordinator.mount.await_args.args[1]
    assert provider.default_model == "qwen3-coder-next"


def test_detect_model_capabilities_cloud_when_cloud_host():
    """_detect_model_capabilities() should report 'cloud' when provider is cloud."""
    provider = OllamaProvider(host="https://ollama.com", api_key="k")
    caps = provider._detect_model_capabilities("gpt-oss:120b")
    assert "cloud" in caps
    assert "local" not in caps


def test_detect_model_capabilities_local_when_local_host():
    """_detect_model_capabilities() should report 'local' for the legacy local path."""
    provider = OllamaProvider(host="http://localhost:11434")
    caps = provider._detect_model_capabilities("llama3.2:3b")
    assert "local" in caps
    assert "cloud" not in caps


def test_get_info_defaults_model_reflects_cloud_default():
    """ProviderInfo.defaults['model'] must surface the cloud default for cloud providers."""
    provider = OllamaProvider(
        host="https://ollama.com", api_key="k", config={"mode": "cloud"}
    )
    info = provider.get_info()
    assert info.defaults["model"] == "gpt-oss:120b"


def test_get_info_defaults_model_reflects_local_default():
    """ProviderInfo.defaults['model'] must surface the local default for local providers."""
    provider = OllamaProvider(host="http://localhost:11434", config={"mode": "local"})
    info = provider.get_info()
    assert info.defaults["model"] == "llama3.2:3b"


def test_get_info_defaults_model_respects_explicit_override():
    """An explicit default_model in config must propagate to ProviderInfo.defaults['model']."""
    provider = OllamaProvider(
        host="https://ollama.com",
        api_key="k",
        config={"mode": "cloud", "default_model": "qwen3-coder-next"},
    )
    info = provider.get_info()
    assert info.defaults["model"] == "qwen3-coder-next"


@pytest.mark.asyncio(loop_scope="function")
async def test_close_resets_client_for_cloud_provider():
    """Sanity check that close() still works on a cloud-configured provider."""
    provider = OllamaProvider(host="https://ollama.com", api_key="key")
    # Force lazy client init
    with patch("amplifier_module_provider_ollama.AsyncClient"):
        _ = provider.client
    assert provider._client is not None
    await provider.close()
    assert provider._client is None
