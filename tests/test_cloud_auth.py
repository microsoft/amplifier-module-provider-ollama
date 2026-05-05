"""Tests for Ollama Cloud authentication and the host-as-SSOT design.

After PR-feedback refactor: ``host`` is the single source of truth for
local-vs-cloud. There is no ``mode`` field; ``is_cloud`` is URL-derived
and cached at construction; ``default_model``, capability tags, skip-pull
behavior, and Bearer header all key off the SAME source (host URL +
optional api_key).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_module_provider_ollama import OllamaProvider


# ---------- Bearer header / api_key plumbing ----------


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


def test_init_with_api_key_attaches_bearer_for_custom_proxy():
    """Bearer is attached whenever api_key is set, even for non-cloud hosts.

    Supports custom auth-proxy deployments (Bearer-auth proxy in front of a
    local Ollama). Semantic cloud-vs-local decisions key off is_cloud (host
    URL); raw header attachment keys off api_key alone.
    """
    provider = OllamaProvider(host="https://my-proxy.internal:8443", api_key="proxy-key")
    assert provider._headers == {"Authorization": "Bearer proxy-key"}
    assert provider.is_cloud is False  # proxy is not Ollama Cloud


# ---------- is_cloud URL detection (cached) ----------


def test_is_cloud_property_basic():
    cloud = OllamaProvider(host="https://ollama.com", api_key="k")
    local = OllamaProvider(host="http://localhost:11434")
    custom = OllamaProvider(host="https://my-internal.example/")
    assert cloud.is_cloud is True
    assert local.is_cloud is False
    assert custom.is_cloud is False


def test_is_cloud_handles_subdomains_and_ports():
    sub = OllamaProvider(host="https://staging.ollama.com")
    with_port = OllamaProvider(host="https://ollama.com:443")
    assert sub.is_cloud is True
    assert with_port.is_cloud is True


def test_is_cloud_rejects_lookalike_hosts():
    """Defends against attacker-controlled lookalike hostnames.

    These are exactly the hosts a ``show_when={"host": "contains:ollama.com"}``
    init gate WOULD prompt for, but runtime is_cloud correctly says False so
    cloud-only behaviors (skip-pull, etc.) are not triggered.
    """
    evil = OllamaProvider(host="http://evil.ollama.com.attacker.io")
    path_lookalike = OllamaProvider(host="http://localhost/ollama.com")
    suffix_lookalike = OllamaProvider(host="https://notollama.com")
    empty_host = OllamaProvider(host="")
    assert evil.is_cloud is False
    assert path_lookalike.is_cloud is False
    assert suffix_lookalike.is_cloud is False
    assert empty_host.is_cloud is False


def test_is_cloud_is_cached_at_construction():
    """is_cloud should be computed once at __init__, not re-parsed each access."""
    provider = OllamaProvider(host="https://ollama.com", api_key="k")
    assert hasattr(provider, "_is_cloud_cached")
    assert provider._is_cloud_cached is True
    assert provider.is_cloud is provider._is_cloud_cached


# ---------- AsyncClient header passthrough ----------


def test_client_property_passes_headers_to_async_client():
    provider = OllamaProvider(host="https://ollama.com", api_key="abc123")
    with patch("amplifier_module_provider_ollama.AsyncClient") as mock_client_cls:
        _ = provider.client
        mock_client_cls.assert_called_once_with(
            host="https://ollama.com",
            headers={"Authorization": "Bearer abc123"},
        )


def test_client_property_passes_none_headers_for_local():
    provider = OllamaProvider(host="http://localhost:11434")
    with patch("amplifier_module_provider_ollama.AsyncClient") as mock_client_cls:
        _ = provider.client
        mock_client_cls.assert_called_once_with(
            host="http://localhost:11434",
            headers=None,
        )


# ---------- mount() env-var + config integration ----------


def _fake_coordinator() -> MagicMock:
    coord = MagicMock()
    coord.mount = AsyncMock()
    return coord


@pytest.mark.asyncio(loop_scope="function")
async def test_mount_reads_api_key_from_env(monkeypatch):
    from amplifier_module_provider_ollama import mount

    monkeypatch.setenv("OLLAMA_API_KEY", "env-key")
    monkeypatch.setenv("OLLAMA_HOST", "https://ollama.com")

    coordinator = _fake_coordinator()
    with patch("amplifier_module_provider_ollama.OllamaProvider._check_connection"):
        await mount(coordinator=coordinator, config={})

    coordinator.mount.assert_awaited_once()
    provider = coordinator.mount.await_args.args[1]
    assert isinstance(provider, OllamaProvider)
    assert provider._api_key == "env-key"
    assert provider._headers == {"Authorization": "Bearer env-key"}


@pytest.mark.asyncio(loop_scope="function")
async def test_mount_config_overrides_env(monkeypatch):
    from amplifier_module_provider_ollama import mount

    monkeypatch.setenv("OLLAMA_API_KEY", "env-key")
    monkeypatch.setenv("OLLAMA_HOST", "https://ollama.com")

    coordinator = _fake_coordinator()
    with patch("amplifier_module_provider_ollama.OllamaProvider._check_connection"):
        await mount(coordinator=coordinator, config={"api_key": "config-key"})

    provider = coordinator.mount.await_args.args[1]
    assert provider._api_key == "config-key"


@pytest.mark.asyncio(loop_scope="function")
async def test_mount_default_host_is_localhost(monkeypatch):
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
async def test_mount_explicit_cloud_host(monkeypatch):
    from amplifier_module_provider_ollama import mount

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    coordinator = _fake_coordinator()
    with patch("amplifier_module_provider_ollama.OllamaProvider._check_connection"):
        await mount(
            coordinator=coordinator,
            config={"host": "https://ollama.com", "api_key": "k"},
        )

    provider = coordinator.mount.await_args.args[1]
    assert provider.host == "https://ollama.com"
    assert provider.is_cloud is True
    assert provider.default_model == "gpt-oss:120b"


# ---------- Backward-compat: legacy `mode` configs are silently ignored ----------


@pytest.mark.asyncio(loop_scope="function")
async def test_mount_legacy_mode_key_is_silently_ignored(monkeypatch):
    """Old configs containing `mode` keep working - mode is just dropped.

    Existing users who saved a config from the previous init UX will have a
    stray `mode` key in their TOML. The provider must not error or behave
    differently because of it; host is the only signal.
    """
    from amplifier_module_provider_ollama import mount

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    coordinator = _fake_coordinator()
    with patch("amplifier_module_provider_ollama.OllamaProvider._check_connection"):
        await mount(
            coordinator=coordinator,
            config={"mode": "cloud", "host": "http://localhost:11434"},
        )

    provider = coordinator.mount.await_args.args[1]
    assert provider.host == "http://localhost:11434"
    assert provider.is_cloud is False
    assert provider.default_model == "llama3.2:3b"  # follows host, not legacy mode


# ---------- default_model derivation: SSOT regression tests ----------
#
# Reviewer flagged this divergence in the previous design:
#   - mode="cloud" + host="http://localhost:11434"
#     -> default_model=gpt-oss:120b but is_cloud=False, no Bearer, pull enabled
#   - mode="local" + host="https://ollama.com"
#     -> default_model=llama3.2:3b but is_cloud=True, Bearer applies, no pull
# These cases are no longer expressible - the tests below LOCK that in.


def test_default_model_for_cloud_host():
    provider = OllamaProvider(host="https://ollama.com", api_key="k")
    assert provider.default_model == "gpt-oss:120b"


def test_default_model_for_local_host():
    provider = OllamaProvider(host="http://localhost:11434")
    assert provider.default_model == "llama3.2:3b"


def test_default_model_for_custom_remote_host():
    """Non-Ollama-Cloud remote hosts default to the local model name."""
    provider = OllamaProvider(host="https://my-internal-ollama.example.com")
    assert provider.default_model == "llama3.2:3b"


def test_default_model_explicit_override_wins():
    cloud = OllamaProvider(
        host="https://ollama.com",
        api_key="k",
        config={"default_model": "qwen3-coder-next"},
    )
    local = OllamaProvider(
        host="http://localhost:11434",
        config={"default_model": "phi4-mini"},
    )
    assert cloud.default_model == "qwen3-coder-next"
    assert local.default_model == "phi4-mini"


def test_ssot_invariant_default_model_follows_host_not_legacy_mode():
    """Regression test for the reviewer's divergence cases.

    Even if a caller passes a legacy `mode` key, default_model must follow
    the host URL (is_cloud) - never `mode`. If this test ever fails, the
    SSOT principle has been broken.
    """
    case_1 = OllamaProvider(
        host="http://localhost:11434",
        config={"mode": "cloud"},  # silently ignored
    )
    assert case_1.is_cloud is False
    assert case_1.default_model == "llama3.2:3b"

    case_2 = OllamaProvider(
        host="https://ollama.com",
        api_key="k",
        config={"mode": "local"},  # silently ignored
    )
    assert case_2.is_cloud is True
    assert case_2.default_model == "gpt-oss:120b"


# ---------- get_info() ConfigField shape ----------


def test_get_info_declares_credential_env_var():
    info = OllamaProvider(host="http://localhost:11434").get_info()
    assert "OLLAMA_API_KEY" in info.credential_env_vars


def test_get_info_has_no_mode_field():
    """`mode` must be GONE - host is the single source of truth now."""
    info = OllamaProvider(host="http://localhost:11434").get_info()
    mode_fields = [f for f in info.config_fields if f.id == "mode"]
    assert mode_fields == [], "mode field should have been removed"


def test_get_info_has_single_host_field_no_duplicates():
    """Only ONE `host` ConfigField (the duplicate-id pattern is gone)."""
    info = OllamaProvider(host="http://localhost:11434").get_info()
    host_fields = [f for f in info.config_fields if f.id == "host"]
    assert len(host_fields) == 1
    field = host_fields[0]
    assert field.field_type == "text"
    assert field.env_var == "OLLAMA_HOST"
    assert field.default == "http://localhost:11434"
    assert field.show_when is None  # SSOT, always relevant


def test_get_info_host_prompt_documents_cloud_option():
    info = OllamaProvider(host="http://localhost:11434").get_info()
    host_field = next(f for f in info.config_fields if f.id == "host")
    assert "ollama.com" in host_field.prompt.lower()


def test_get_info_api_key_field_gated_by_host_pattern():
    """api_key must be a secret with show_when keyed off host (not mode)."""
    info = OllamaProvider(host="http://localhost:11434").get_info()
    api_key_fields = [f for f in info.config_fields if f.id == "api_key"]
    assert len(api_key_fields) == 1
    field = api_key_fields[0]
    assert field.field_type == "secret"
    assert field.env_var == "OLLAMA_API_KEY"
    assert field.required is False
    assert field.show_when == {"host": "contains:ollama.com"}


def test_get_info_auto_pull_field_gated_by_host_pattern():
    info = OllamaProvider(host="http://localhost:11434").get_info()
    auto_pull_fields = [f for f in info.config_fields if f.id == "auto_pull"]
    assert len(auto_pull_fields) == 1
    assert auto_pull_fields[0].show_when == {"host": "not_contains:ollama.com"}


# ---------- Capability tagging tracks is_cloud (URL-derived) ----------


def test_get_info_capabilities_local_when_local_host():
    info = OllamaProvider(host="http://localhost:11434").get_info()
    assert "local" in info.capabilities
    assert "cloud" not in info.capabilities


def test_get_info_capabilities_cloud_when_cloud_host():
    info = OllamaProvider(host="https://ollama.com", api_key="k").get_info()
    assert "cloud" in info.capabilities
    assert "local" not in info.capabilities


def test_detect_model_capabilities_cloud_when_cloud_host():
    provider = OllamaProvider(host="https://ollama.com", api_key="k")
    caps = provider._detect_model_capabilities("gpt-oss:120b")
    assert "cloud" in caps
    assert "local" not in caps


def test_detect_model_capabilities_local_when_local_host():
    provider = OllamaProvider(host="http://localhost:11434")
    caps = provider._detect_model_capabilities("llama3.2:3b")
    assert "local" in caps
    assert "cloud" not in caps


# ---------- get_info().defaults["model"] tracks is_cloud-derived default_model ----------


def test_get_info_defaults_model_reflects_cloud_default():
    provider = OllamaProvider(host="https://ollama.com", api_key="k")
    info = provider.get_info()
    assert info.defaults["model"] == "gpt-oss:120b"


def test_get_info_defaults_model_reflects_local_default():
    provider = OllamaProvider(host="http://localhost:11434")
    info = provider.get_info()
    assert info.defaults["model"] == "llama3.2:3b"


def test_get_info_defaults_model_respects_explicit_override():
    provider = OllamaProvider(
        host="https://ollama.com",
        api_key="k",
        config={"default_model": "qwen3-coder-next"},
    )
    info = provider.get_info()
    assert info.defaults["model"] == "qwen3-coder-next"


# ---------- Cleanup ----------


@pytest.mark.asyncio(loop_scope="function")
async def test_close_resets_client_for_cloud_provider():
    provider = OllamaProvider(host="https://ollama.com", api_key="key")
    with patch("amplifier_module_provider_ollama.AsyncClient"):
        _ = provider.client
    assert provider._client is not None
    await provider.close()
    assert provider._client is None
