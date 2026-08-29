"""Config hygiene tests: bool/numeric coercion, unknown-key sweep, ghost-key
messages, extra_request_params, and stop-parameter placement.

These guard the "family hygiene wave" fixes:
  - auto_pull (and other boolean-ish keys) previously used bare
    ``self.config.get(key, default)`` -- a config wizard writes booleans as
    the STRING "false", and ``bool("false")`` is ``True`` in Python, so
    ``auto_pull: "false"`` was silently treated as ``auto_pull=True``.
  - numeric keys previously used bare ``int()``/``float()`` which raises
    (crashing mount) on an invalid string instead of warning and defaulting.
  - unknown config keys were silently ignored with no feedback.
  - `stop` was sent as a top-level Ollama API param instead of nested inside
    `options` (per docs/api.md).
"""

from __future__ import annotations

import logging

import pytest

import amplifier_module_provider_ollama as _provider_module

# These are intentionally module-private helpers with no public contract --
# accessed directly (not via `from X import _private_name`, which some
# static analyzers fail to resolve for underscore-prefixed module attrs).
_coerce_bool = _provider_module._coerce_bool  # type: ignore[attr-defined]
_coerce_float = _provider_module._coerce_float  # type: ignore[attr-defined]
_coerce_int = _provider_module._coerce_int  # type: ignore[attr-defined]
_warn_unknown_config_keys = _provider_module._warn_unknown_config_keys  # type: ignore[attr-defined]


class TestCoerceBool:
    def test_bool_passthrough(self):
        assert _coerce_bool(True, key="x", default=False) is True
        assert _coerce_bool(False, key="x", default=True) is False

    def test_string_false_is_false(self):
        # THE live bug: bool("false") is True in Python.
        assert _coerce_bool("false", key="x", default=True) is False

    def test_string_true_is_true(self):
        assert _coerce_bool("true", key="x", default=False) is True

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("True", True),
            ("FALSE", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
        ],
    )
    def test_case_insensitive_and_alt_forms(self, raw, expected):
        assert _coerce_bool(raw, key="x", default=not expected) is expected

    def test_none_uses_default(self):
        assert _coerce_bool(None, key="x", default=True) is True
        assert _coerce_bool(None, key="x", default=False) is False

    def test_unrecognized_string_warns_and_defaults(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = _coerce_bool("maybe", key="auto_pull", default=True)
        assert result is True
        assert "auto_pull" in caplog.text
        assert "maybe" in caplog.text


class TestCoerceNumeric:
    def test_int_passthrough(self):
        assert _coerce_int(5, key="x", default=0) == 5

    def test_int_from_string(self):
        assert _coerce_int("600", key="timeout", default=0) == 600

    def test_int_invalid_warns_and_defaults(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = _coerce_int("not-a-number", key="num_ctx", default=42)
        assert result == 42
        assert "num_ctx" in caplog.text

    def test_float_from_string(self):
        assert _coerce_float("1.5", key="temperature", default=0.0) == 1.5

    def test_float_invalid_warns_and_defaults(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = _coerce_float("garbage", key="temperature", default=0.7)
        assert result == 0.7
        assert "temperature" in caplog.text

    def test_none_uses_default(self):
        assert _coerce_int(None, key="x", default=3) == 3
        assert _coerce_float(None, key="x", default=1.0) == 1.0


class TestUnknownConfigKeySweep:
    def test_known_keys_are_silent(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"host": "http://x", "priority": 1})
        assert caplog.text == ""

    def test_extra_request_params_is_allowlisted(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"extra_request_params": {"mirostat": 1}})
        assert caplog.text == ""

    def test_unknown_key_warns_with_suggestion(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"tempurature": 0.5})
        assert "tempurature" in caplog.text
        assert "temperature" in caplog.text

    def test_ghost_key_debug_gets_targeted_message(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"debug": True})
        assert "no longer read" in caplog.text

    def test_ghost_key_raw_debug_points_at_raw(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"raw_debug": True})
        assert "raw" in caplog.text

    def test_ghost_key_mode_gets_targeted_message(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"mode": "local"})
        assert "instance_id" in caplog.text


class TestProviderConfigCoercionIntegration:
    """Verify OllamaProvider.__init__ actually applies coercion, not just the
    helper functions in isolation."""

    def test_auto_pull_string_false_is_false(self, make_provider):
        # THE live bug, reproduced end-to-end via the wizard's actual string form.
        provider = make_provider(auto_pull="false")
        assert provider.auto_pull is False

    def test_auto_pull_string_true_is_true(self, make_provider):
        provider = make_provider(auto_pull="true")
        assert provider.auto_pull is True

    def test_auto_pull_default_is_false(self, make_provider):
        provider = make_provider()
        assert provider.auto_pull is False

    def test_use_streaming_string_false_disables_streaming(self, make_provider):
        provider = make_provider(use_streaming="false")
        assert (
            _coerce_bool(
                provider.config.get("use_streaming"),
                key="use_streaming",
                default=True,
            )
            is False
        )

    def test_raw_string_false_is_false(self, make_provider):
        provider = make_provider(raw="false")
        assert provider.raw is False

    def test_enable_thinking_string_false_is_false(self, make_provider):
        provider = make_provider(enable_thinking="false")
        assert provider.enable_thinking is False

    def test_retry_jitter_string_false_is_false(self, make_provider):
        # RetryConfig.jitter is a numeric-compat wrapper: 0.2 if enabled,
        # 0.0 if disabled (see amplifier_core.utils.retry.RetryConfig).
        provider = make_provider(retry_jitter="false")
        assert provider._retry_config.jitter == 0.0

    def test_retry_jitter_string_true_is_enabled(self, make_provider):
        provider = make_provider(retry_jitter="true")
        assert provider._retry_config.jitter > 0.0

    def test_numeric_keys_from_strings(self, make_provider):
        provider = make_provider(
            max_tokens="123",
            temperature="0.3",
            timeout="30",
            num_ctx="2048",
            max_retries="5",
            min_retry_delay="0.5",
            max_retry_delay="10",
        )
        assert provider.max_tokens == 123
        assert provider.temperature == 0.3
        assert provider.timeout == 30.0
        assert provider.num_ctx == 2048
        assert provider._retry_config.max_retries == 5
        assert provider._retry_config.initial_delay == 0.5
        assert provider._retry_config.max_delay == 10.0

    def test_invalid_numeric_string_defaults_instead_of_crashing(self, make_provider):
        provider = make_provider(timeout="not-a-number")
        assert provider.timeout == 600.0

    def test_extra_request_params_stored(self, make_provider):
        provider = make_provider(extra_request_params={"mirostat": 2})
        assert provider.extra_request_params == {"mirostat": 2}

    def test_extra_request_params_non_dict_ignored(self, make_provider, caplog):
        with caplog.at_level(logging.WARNING):
            provider = make_provider(extra_request_params="nope")
        assert provider.extra_request_params == {}
        assert "extra_request_params" in caplog.text


class TestBuildOptions:
    """_build_options() is the shared helper that replaced two duplicated
    inline blocks -- verify stop placement and extra_request_params merge."""

    @pytest.mark.asyncio
    async def test_stop_is_nested_inside_options(self, make_provider, simple_request):
        provider = make_provider(stop=["\n", "user:"])
        request = simple_request()
        options = await provider._build_options("llama3.2:3b", request, {})
        assert options["stop"] == ["\n", "user:"]

    @pytest.mark.asyncio
    async def test_extra_request_params_merged_last(self, make_provider, simple_request):
        provider = make_provider(
            temperature=0.7, extra_request_params={"temperature": 0.1, "mirostat": 2}
        )
        request = simple_request()
        options = await provider._build_options("llama3.2:3b", request, {})
        # extra_request_params overrides the computed default when both set the
        # same key, and adds keys with no dedicated field.
        assert options["temperature"] == 0.1
        assert options["mirostat"] == 2

    @pytest.mark.asyncio
    async def test_no_extra_request_params_is_noop(self, make_provider, simple_request):
        provider = make_provider(temperature=0.5)
        request = simple_request()
        options = await provider._build_options("llama3.2:3b", request, {})
        assert options["temperature"] == 0.5
        assert "mirostat" not in options
