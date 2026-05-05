"""
Ollama provider module for Amplifier.
Integrates with local Ollama server for LLM completions.
"""

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import asyncio
import logging
import os
import time
from urllib.parse import urlparse
from collections import defaultdict
from ._constants import CLOUD_DEFAULT_MODEL, LOCAL_DEFAULT_MODEL
from typing import Any
from uuid import uuid4

from amplifier_core import ConfigField
from amplifier_core.utils.retry import RetryConfig, retry_with_backoff
from amplifier_core.llm_errors import (
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
from amplifier_core import ModelInfo
from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderInfo
from amplifier_core import TextContent
from amplifier_core import ThinkingContent
from amplifier_core import ToolCallContent
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Message
from amplifier_core.message_models import ThinkingBlock
from amplifier_core.message_models import ToolCall
from ollama import AsyncClient  # pyright: ignore[reportAttributeAccessIssue]
from ollama import ResponseError  # pyright: ignore[reportAttributeAccessIssue]

logger = logging.getLogger(__name__)

# Unified default context length used when model metadata is unavailable
DEFAULT_CONTEXT_LENGTH = 8192


def _is_cloud_host(host: str | None) -> bool:
    """True when host points at Ollama Cloud (ollama.com or any subdomain).

    URL-parse based to defend against lookalike hosts such as
    ``http://evil.ollama.com.attacker.io`` or ``https://notollama.com``.
    This is THE single source of truth for cloud-vs-local runtime decisions
    (default model selection, capability tagging, skip-pull behavior).
    """
    if not host:
        return False
    try:
        netloc = urlparse(host).netloc.lower()
    except ValueError:
        return False
    netloc = netloc.split(":")[0]  # strip port (e.g., "ollama.com:443")
    return netloc == "ollama.com" or netloc.endswith(".ollama.com")


class OllamaChatResponse(ChatResponse):
    """Extended ChatResponse with Ollama-specific metadata."""

    raw_response: dict[str, Any] | None = None
    model_name: str | None = None
    thinking_content: str | None = None
    # content_blocks for streaming UI compatibility (triggers content_block:start/end events)
    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


def _translate_ollama_error(e: Exception) -> LLMError:  # pyright: ignore[reportReturnType]
    """Translate native Ollama/connection errors to kernel LLM error types.

    Called inside _do_complete() / _do_stream_connect() so that
    retry_with_backoff sees LLMError subclasses and can check .retryable
    to decide whether to retry.
    5xx errors become ProviderUnavailableError(retryable=True), while
    4xx errors become non-retryable errors that raise immediately.

    The returned exception should be raised with ``raise ... from e`` to
    preserve the original ``__cause__``.
    """
    if isinstance(e, ResponseError):
        status = getattr(e, "status_code", None)
        if status in (401, 403):
            return AuthenticationError(str(e), provider="ollama", status_code=status)  # pyright: ignore[reportReturnType]
        if status == 429:
            # Note: ollama SDK's ResponseError doesn't expose HTTP headers,
            # so retry_after cannot be extracted from the response.
            return RateLimitError(
                str(e), provider="ollama", status_code=429, retryable=True
            )  # pyright: ignore[reportReturnType]
        if status == 400:
            msg = str(e).lower()
            if (
                "context length" in msg
                or "too many tokens" in msg
                or "token limit" in msg
            ):
                return ContextLengthError(str(e), provider="ollama", status_code=400)  # pyright: ignore[reportReturnType]
            if "content filter" in msg or "safety" in msg or "blocked" in msg:
                return ContentFilterError(str(e), provider="ollama", status_code=400)  # pyright: ignore[reportReturnType]
            return InvalidRequestError(str(e), provider="ollama", status_code=status)  # pyright: ignore[reportReturnType]
        if status == 404:
            return NotFoundError(str(e), provider="ollama", status_code=404)  # pyright: ignore[reportReturnType]
        if status is not None and 500 <= status < 600:
            return ProviderUnavailableError(
                str(e), provider="ollama", status_code=status
            )  # pyright: ignore[reportReturnType]
        return LLMError(str(e), provider="ollama", retryable=True)
    # TimeoutError is a subclass of OSError in Python 3.11+, so check it
    # *before* the broader (ConnectionError, OSError) catch.
    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
        return LLMTimeoutError(str(e), provider="ollama", retryable=True)  # pyright: ignore[reportReturnType]
    if isinstance(e, (ConnectionError, OSError)):
        return ProviderUnavailableError(str(e), provider="ollama", retryable=True)  # pyright: ignore[reportReturnType]
    return LLMError(str(e), provider="ollama", retryable=True)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the Ollama provider.

    Args:
        coordinator: Module coordinator
        config: Provider configuration including:
            - host: Ollama server URL (default: from OLLAMA_HOST or http://localhost:11434).
                    Use https://ollama.com for Ollama Cloud. The host URL is the
                    single source of truth — local-vs-cloud is derived from it.
            - default_model: Model to use. Defaults are host-derived: gpt-oss:120b
                    for Ollama Cloud (host on ollama.com), llama3.2:3b otherwise.
            - max_tokens: Maximum tokens (default: 4096)
            - temperature: Generation temperature (default: 0.7)
            - timeout: Request timeout in seconds (default: 600)
            - auto_pull: Whether to auto-pull missing models (default: False;
                    silently ignored for cloud hosts since pull isn't supported).
            - api_key: Ollama Cloud API key (default: from OLLAMA_API_KEY env var).
                    Used to attach Authorization: Bearer when set; harmless if unset.

        To run a mix of local + cloud simultaneously, configure two provider
        instances with different ``instance_id`` values — see the README
        "Mixed local + cloud (multi-instance)" section for an example.

    Returns:
        Optional cleanup function
    """
    config = config or {}

    # Single source of truth: the `host` URL drives all downstream decisions
    # (cloud-vs-local detection, default_model, capabilities, skip-pull).
    # Legacy configs containing a `mode` key are silently ignored — `mode`
    # was removed in favor of host-based derivation. To run a mix of local
    # + cloud, configure two provider instances with different `instance_id`
    # values (see README).
    host = config.get("host") or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    api_key = config.get("api_key") or os.environ.get("OLLAMA_API_KEY")
    provider = OllamaProvider(host, config, coordinator, api_key=api_key)
    await coordinator.mount("providers", provider, name="ollama")

    # Test connection but don't fail mount
    if not await provider._check_connection():
        logger.warning(
            f"Ollama server at {host} is not reachable. Provider mounted but will fail on use."
        )
    else:
        logger.info(f"Mounted OllamaProvider at {host}")

    # Return cleanup function (ollama client doesn't have explicit close)
    async def cleanup():
        # Ollama AsyncClient uses httpx internally which handles cleanup
        pass

    return cleanup


class OllamaProvider:
    """Ollama LLM integration (local and cloud)."""

    name = "ollama"
    api_label = "Ollama"

    def __init__(
        self,
        host: str | None = None,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize Ollama provider.

        The SDK client is created lazily on first use, allowing get_info()
        to work without a running Ollama server.

        Args:
            host: Ollama server URL (can be None for get_info() calls)
            config: Additional configuration
            coordinator: Module coordinator for event emission
        """
        self.host = host
        self._api_key: str | None = api_key
        # Authorization: Bearer is attached whenever an api_key is supplied,
        # regardless of host. This keeps custom auth-proxy deployments working
        # (some users put a Bearer-auth proxy in front of a local Ollama).
        # The is_cloud check below governs SEMANTIC behavior (default_model,
        # capability tags, skip-pull), not raw header attachment.
        self._headers: dict[str, str] | None = (
            {"Authorization": f"Bearer {api_key}"} if api_key else None
        )
        self._client: AsyncClient | None = None  # Lazy init
        self.config = config or {}
        self.coordinator = coordinator

        # Single source of truth: host URL determines local-vs-cloud. Cached
        # here so we don't re-parse the URL on every property access (used in
        # capabilities, default_model selection, skip-pull behavior, etc.).
        self._is_cloud_cached: bool = _is_cloud_host(host)

        # Default model is host-derived — gpt-oss:120b for Ollama Cloud,
        # llama3.2:3b otherwise. Override via `default_model` in config.
        # Constants come from _constants.py (single source of truth for names).
        _host_default = (
            CLOUD_DEFAULT_MODEL if self._is_cloud_cached else LOCAL_DEFAULT_MODEL
        )
        self.default_model = self.config.get("default_model", _host_default)
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.7)
        self.timeout = float(
            self.config.get("timeout", 600.0)
        )  # API timeout in seconds (default 10 min - local models need longer for prefill)
        self.auto_pull = self.config.get("auto_pull", False)
        self.raw = self.config.get("raw", False)
        # Context window size (num_ctx in ollama) - 0 means auto-detect from model
        self.num_ctx = int(self.config.get("num_ctx", 0))
        # Cache for model context lengths (avoid repeated API calls)
        self._model_ctx_cache: dict[str, int] = {}
        # Enable thinking/reasoning for models that support it (default: True)
        # Models that don't support thinking will simply ignore this option
        self.enable_thinking = self.config.get("enable_thinking", True)
        # Thinking effort level: None (boolean True), or "high"/"medium"/"low"
        self.thinking_effort: str | None = self.config.get("thinking_effort")

        # Sampling parameters (3-tier precedence: request -> kwargs -> instance default)
        # Only forwarded to Ollama when explicitly set (never send None)
        self.top_p: float | None = self.config.get("top_p")
        self.top_k: int | None = self.config.get("top_k")
        self.min_p: float | None = self.config.get("min_p")
        self.repeat_penalty: float | None = self.config.get("repeat_penalty")
        self.seed: int | None = self.config.get("seed")
        self.stop: list[str] | None = self.config.get("stop")

        # Keep model loaded in memory (e.g. "5m", "1h", "-1s" for indefinite)
        # Normalize bare numeric values like "-1" or "0" to include "s" unit suffix
        # since Ollama's duration parser requires a unit
        _keep_alive_raw = self.config.get("keep_alive")
        if _keep_alive_raw is not None:
            _ka_str = str(_keep_alive_raw).strip()
            # If it's a bare number (int/float, possibly negative), append "s"
            try:
                float(_ka_str)
                _ka_str = f"{_ka_str}s"
            except ValueError:
                pass  # Already has a unit suffix like "5m", "1h"
            self.keep_alive: str | None = _ka_str
        else:
            self.keep_alive: str | None = None

        # Logprobs support (requires Ollama >= 0.12.11)
        self.logprobs: bool | None = self.config.get("logprobs")
        self.top_logprobs: int | None = self.config.get("top_logprobs")

        # Track tool call IDs that have been repaired with synthetic results.
        # This prevents infinite loops when the same missing tool results are
        # detected repeatedly across LLM iterations (since synthetic results
        # are injected into request.messages but not persisted to message store).
        self._repaired_tool_ids: set[str] = set()

        # Retry configuration using amplifier-core's RetryConfig
        self._retry_config = RetryConfig(
            max_retries=int(self.config.get("max_retries", 3)),
            initial_delay=float(self.config.get("min_retry_delay", 1.0)),
            max_delay=float(self.config.get("max_retry_delay", 60.0)),
            jitter=bool(self.config.get("retry_jitter", True)),
        )

    @property
    def client(self) -> AsyncClient:
        """Lazily initialize the Ollama client on first access."""
        if self._client is None:
            if self.host is None:
                raise ValueError("host must be provided for API calls")
            self._client = AsyncClient(host=self.host, headers=self._headers)
        return self._client

    @property
    def is_cloud(self) -> bool:
        """True when configured against Ollama Cloud.

        Returns the value cached in ``__init__`` — see :func:`_is_cloud_host`
        for the URL-parse logic that defends against lookalike hosts.
        """
        return self._is_cloud_cached

    def get_info(self) -> ProviderInfo:
        """Get provider metadata."""
        return ProviderInfo(
            id="ollama",
            display_name="Ollama",
            credential_env_vars=["OLLAMA_API_KEY"],
            capabilities=["streaming", "tools", "cloud" if self.is_cloud else "local"],
            defaults={
                "model": self.default_model,
                "max_tokens": 4096,
                "temperature": 0.7,
                "timeout": 600.0,
                "context_window": 128000,
                "max_output_tokens": 64000,
            },
            config_fields=[
                # Single host field — THE single source of truth for local-vs-cloud.
                # Use http://localhost:11434 for local, https://ollama.com for
                # Ollama Cloud, or any custom URL for a self-hosted/proxied
                # deployment. To run BOTH local AND cloud simultaneously,
                # configure two provider instances with different ``instance_id``
                # values (see README "Mixed local + cloud").
                ConfigField(
                    id="host",
                    display_name="Ollama Host",
                    field_type="text",
                    prompt="Ollama server URL (use https://ollama.com for Ollama Cloud)",
                    env_var="OLLAMA_HOST",
                    default="http://localhost:11434",
                    required=False,
                ),
                # API key — only PROMPTED when host contains "ollama.com".
                # NOTE: ``contains:ollama.com`` is intentionally more permissive
                # than the runtime is_cloud check (which urlparses to defend
                # against lookalike hosts like ``evil.ollama.com.attacker.io``).
                # At init time this only governs whether to PROMPT for a key;
                # the runtime decision about whether to attach Authorization:
                # Bearer is governed solely by the api_key being set, so users
                # with custom auth proxies (Bearer-auth in front of a local
                # Ollama) still work without any host-name acrobatics.
                ConfigField(
                    id="api_key",
                    display_name="API Key",
                    field_type="secret",
                    prompt="Enter your Ollama Cloud API key",
                    env_var="OLLAMA_API_KEY",
                    required=False,
                    show_when={"host": "contains:ollama.com"},
                ),
                # auto_pull — Ollama Cloud doesn't support `ollama pull`, so
                # only prompt for non-cloud hosts. Runtime code skips pull()
                # for cloud regardless (defense in depth).
                ConfigField(
                    id="auto_pull",
                    display_name="Auto-Pull Models",
                    field_type="boolean",
                    prompt="Automatically pull missing models?",
                    default="false",
                    required=False,
                    show_when={"host": "not_contains:ollama.com"},
                ),
                ConfigField(
                    id="enable_thinking",
                    display_name="Enable Thinking",
                    field_type="boolean",
                    prompt="Enable thinking/reasoning for supported models?",
                    required=False,
                    default="true",
                ),
                ConfigField(
                    id="keep_alive",
                    display_name="Keep Alive",
                    field_type="text",
                    prompt="Model keep-alive duration (e.g., '5m', '-1' for indefinite)",
                    required=False,
                ),
                ConfigField(
                    id="num_ctx",
                    display_name="Context Window Override",
                    field_type="text",
                    prompt="Context window size override (0 = auto-detect from model)",
                    required=False,
                    default="0",
                ),
                ConfigField(
                    id="timeout",
                    display_name="Request Timeout",
                    field_type="text",
                    prompt="API request timeout in seconds (large models need longer for prefill)",
                    required=False,
                    default="600",
                ),
            ],
        )

    def _detect_model_capabilities(self, model_name: str) -> list[str]:
        """Detect capabilities based on model name/family.

        Uses string matching on model names to determine what features
        a model supports, following the same pattern as the Anthropic
        provider's family-based capability detection.

        Args:
            model_name: The model identifier (e.g., "deepseek-r1:14b", "qwen3-coder-next")

        Returns:
            List of capability strings
        """
        name_lower = model_name.lower()
        caps = ["streaming", "cloud" if self.is_cloud else "local"]

        # Most models support tools now
        caps.append("tools")

        # Thinking/reasoning models
        thinking_families = [
            "deepseek-r1",
            "qwen3:",
            "qwq",
            "magistral",
            "cogito",
        ]
        # Non-thinking models that should NOT get thinking even if they match above
        # e.g., qwen3-coder-next is explicitly non-thinking
        non_thinking = ["qwen3-coder"]
        is_non_thinking = any(f in name_lower for f in non_thinking)
        if not is_non_thinking and any(f in name_lower for f in thinking_families):
            caps.append("thinking")

        # Vision/multimodal models
        vision_families = [
            "llava",
            "llama3.2-vision",
            "gemma3",
            "qwen3-vl",
            "qwen2.5-vl",
            "deepseek-ocr",
            "glm-ocr",
            "minicpm-v",
        ]
        if any(f in name_lower for f in vision_families):
            caps.append("vision")

        # Fast/small models (useful for routing decisions)
        fast_indicators = [":1b", ":3b", ":7b", "gemma3n", "phi3:mini", "phi4-mini"]
        if any(f in name_lower for f in fast_indicators):
            caps.append("fast")

        # JSON/structured output (most modern models support this)
        caps.append("json_mode")

        return caps

    async def list_models(self) -> list[ModelInfo]:
        """
        List available models from local Ollama server.

        Queries the Ollama API to get list of installed models.
        Returns empty list if server is unreachable (allows wizard to fall back to manual input).
        """
        try:
            response = await self.client.list()
        except (ConnectionError, OSError, TimeoutError) as e:
            logger.warning("Could not connect to Ollama server: %s", e)
            return []
        models = []
        # response.models is a list of Model objects with .model attribute (not .name)
        for model in response.models:
            model_name = model.model  # Model objects use .model, not .name
            if model_name:
                # Extract details - model.details is a ModelDetails object
                details = model.details
                context_length = (
                    getattr(details, "context_length", None) or DEFAULT_CONTEXT_LENGTH
                )
                models.append(
                    ModelInfo(
                        id=model_name,
                        display_name=model_name,
                        context_window=context_length,
                        max_output_tokens=context_length,
                        capabilities=self._detect_model_capabilities(model_name),
                        defaults={
                            "temperature": 0.7,
                            "max_tokens": DEFAULT_CONTEXT_LENGTH,
                        },
                    )
                )
        return models

    async def _check_connection(self) -> bool:
        """Verify Ollama server is reachable."""
        try:
            await self.client.list()
            return True
        except Exception:
            return False

    async def _ensure_model_available(self, model: str) -> bool:
        """Check if model is available, attempt to pull if not and auto_pull is enabled.

        When running against Ollama Cloud (is_cloud=True), pulling is not supported.
        show() is still attempted for an availability check, but a failure there only
        logs a warning and returns True so that the actual chat call can proceed — the
        cloud service controls model availability and may surface a clearer error.
        """
        try:
            # Try to get model info
            await self.client.show(model)
            return True
        except ResponseError as e:
            if e.status_code == 404:
                if self.is_cloud:
                    # Cloud does not support pulling — log a warning and let the
                    # subsequent chat call fail with a proper error if needed.
                    logger.warning(
                        f"Model {model} not found via Ollama Cloud show(). "
                        "Skipping pull (not supported on cloud); the chat call may fail."
                    )
                    return True
                if self.auto_pull:
                    logger.info(f"Model {model} not found, pulling...")
                    try:
                        await self.client.pull(model)
                        return True
                    except Exception as pull_error:
                        logger.error(f"Failed to pull model {model}: {pull_error}")
                        return False
                else:
                    logger.warning(
                        f"Model {model} not found. Set auto_pull=True or run 'ollama pull {model}'"
                    )
                    return False
            return False
        except Exception as e:
            if self.is_cloud:
                # Cloud availability check failures should not block requests.
                logger.debug(
                    f"Ollama Cloud show() for {model} raised {type(e).__name__}: {e}"
                )
                return True
            raise

    async def _get_model_context_length(self, model: str) -> int:
        """Get context length for a model, with caching.

        Queries the ollama API to get the model's context_length from model_info.
        Falls back to 8192 if unable to determine.

        Args:
            model: Model name to query

        Returns:
            Context length in tokens
        """
        # Check cache first
        if model in self._model_ctx_cache:
            return self._model_ctx_cache[model]

        try:
            # Query model info from ollama
            info = await self.client.show(model)
            # modelinfo (no underscore) contains context_length (e.g., "gptoss.context_length": 131072)
            model_info = (
                getattr(info, "modelinfo", None)
                or getattr(info, "model_info", None)
                or {}
            )

            # Look for context_length in various formats
            ctx_length = None
            for key, value in model_info.items():
                if "context_length" in key.lower():
                    ctx_length = value
                    break

            if ctx_length and isinstance(ctx_length, int) and ctx_length > 0:
                self._model_ctx_cache[model] = ctx_length
                logger.debug(f"Model {model} context_length: {ctx_length}")
                return ctx_length
        except Exception as e:
            logger.debug(f"Could not get context_length for {model}: {e}")

        # Default fallback
        self._model_ctx_cache[model] = DEFAULT_CONTEXT_LENGTH
        return DEFAULT_CONTEXT_LENGTH

    async def complete(self, request: ChatRequest, **kwargs) -> OllamaChatResponse:
        """
        Generate completion from ChatRequest.

        Args:
            request: Typed chat request with messages, tools, config
            **kwargs: Provider-specific options (override request fields)

        Returns:
            OllamaChatResponse with content blocks, tool calls, usage, and optional thinking
        """
        # Check if streaming is requested
        if hasattr(request, "stream") and request.stream:
            return await self._complete_streaming(request, **kwargs)
        return await self._complete_chat_request(request, **kwargs)

    async def _complete_chat_request(
        self, request: ChatRequest, **kwargs
    ) -> OllamaChatResponse:
        """Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            OllamaChatResponse with content blocks
        """
        logger.info(
            f"[PROVIDER] Received ChatRequest with {len(request.messages)} messages"
        )

        # Validate tool call sequences and repair if needed
        missing = self._find_missing_tool_results(request.messages)

        if missing:
            logger.warning(
                f"[PROVIDER] Ollama: Detected {len(missing)} missing tool result(s). "
                f"Injecting synthetic errors. This indicates a bug in context management. "
                f"Tool IDs: {[call_id for _, call_id, _, _ in missing]}"
            )

            # Insert synthetic results at the correct positions in request.messages,
            # then track repaired IDs to prevent infinite loops.
            self._apply_jit_repair(request, missing)

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name}
                            for _, call_id, tool_name, _ in missing
                        ],
                    },
                )

        # Separate messages by role
        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [
            m for m in request.messages if m.role in ("user", "assistant", "tool")
        ]

        # Build ollama messages list
        ollama_messages = []

        # Add system messages with native role (Ollama supports role: system)
        for sys_msg in system_msgs:
            content = sys_msg.content if isinstance(sys_msg.content, str) else ""
            ollama_messages.append({"role": "system", "content": content})

        # Convert developer messages to XML-wrapped user messages
        for dev_msg in developer_msgs:
            content = dev_msg.content if isinstance(dev_msg.content, str) else ""
            wrapped = f"<context_file>\n{content}\n</context_file>"
            ollama_messages.append({"role": "user", "content": wrapped})

        # Convert conversation messages (synthetics are already in request.messages
        # at the correct positions from _apply_jit_repair above)
        conversation_msgs = self._convert_messages(
            [m.model_dump() for m in conversation]
        )
        ollama_messages.extend(conversation_msgs)

        # Prepare request parameters
        model = kwargs.get("model", self.default_model)

        # Ensure model is available (auto-pull if configured)
        if self.auto_pull:
            await self._ensure_model_available(model)

        # Build options dict with 3-tier precedence: request -> kwargs -> instance default
        options: dict[str, Any] = {
            "temperature": request.temperature
            or kwargs.get("temperature", self.temperature),
            "num_predict": request.max_output_tokens
            or kwargs.get("max_tokens", self.max_tokens),
        }

        # Sampling parameters - only include when explicitly set
        if (top_p := kwargs.get("top_p", self.top_p)) is not None:
            options["top_p"] = top_p
        if (top_k := kwargs.get("top_k", self.top_k)) is not None:
            options["top_k"] = top_k
        if (min_p := kwargs.get("min_p", self.min_p)) is not None:
            options["min_p"] = min_p
        if (
            repeat_penalty := kwargs.get("repeat_penalty", self.repeat_penalty)
        ) is not None:
            options["repeat_penalty"] = repeat_penalty
        if (seed := kwargs.get("seed", self.seed)) is not None:
            options["seed"] = seed

        # Set context window size (num_ctx controls how much context ollama uses)
        # If num_ctx is configured, use it; otherwise auto-detect from model
        if self.num_ctx > 0:
            options["num_ctx"] = self.num_ctx
        else:
            ctx_length = await self._get_model_context_length(model)
            options["num_ctx"] = ctx_length

        params: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "options": options,
        }

        # Stop sequences - top-level param for Ollama SDK
        if (stop := kwargs.get("stop", self.stop)) is not None:
            params["stop"] = stop

        # Keep model loaded in memory
        if self.keep_alive is not None:
            params["keep_alive"] = self.keep_alive

        # Logprobs support (Ollama >= 0.12.11) - top-level params
        if logprobs := kwargs.get("logprobs", self.logprobs):
            params["logprobs"] = logprobs
        if top_logprobs := kwargs.get("top_logprobs", self.top_logprobs):
            params["top_logprobs"] = top_logprobs

        # Add tools if provided
        if request.tools:
            params["tools"] = self._format_tools_from_request(request.tools)

        # Add structured output format if specified
        if hasattr(request, "response_format") and request.response_format:
            if isinstance(request.response_format, dict):
                # JSON schema for structured output
                params["format"] = request.response_format
            elif request.response_format == "json":
                # Simple JSON mode
                params["format"] = "json"

        # Enable thinking/reasoning only for models that support it
        # think is a top-level parameter (not inside options) since Ollama v0.9.0
        # Supports boolean True or effort levels: "high", "medium", "low"
        #
        # Precedence: kwargs/request.enable_thinking → request.reasoning_effort
        #             → provider config (self.enable_thinking) → default off
        include_thinking = False
        model_caps = self._detect_model_capabilities(model)
        if "thinking" in model_caps:
            if hasattr(request, "enable_thinking") and request.enable_thinking:  # pyright: ignore[reportAttributeAccessIssue]
                params["think"] = self.thinking_effort or True
                include_thinking = True
            elif request.reasoning_effort is not None:
                # Ollama v0.9.0+ supports effort levels ("high", "medium", "low")
                # via the `think` parameter — pass through directly.
                params["think"] = request.reasoning_effort
                include_thinking = True
            elif self.enable_thinking:
                params["think"] = self.thinking_effort or True
                include_thinking = True

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            request_payload: dict[str, Any] = {
                "provider": "ollama",
                "model": model,
                "message_count": len(ollama_messages),
            }
            if self.raw:
                request_payload["raw"] = params
            await self.coordinator.hooks.emit("llm:request", request_payload)

        start_time = time.time()

        # Inner function: wraps the Ollama API call with error translation
        # so that retry_with_backoff sees LLMError subclasses and can check
        # .retryable to decide whether to retry (e.g. 5xx → retried, 400 → not).
        async def _do_complete():
            try:
                return await asyncio.wait_for(
                    self.client.chat(**params), timeout=self.timeout
                )
            except ResponseError as e:
                raise _translate_ollama_error(e) from e
            except (asyncio.TimeoutError, TimeoutError) as e:
                raise LLMTimeoutError(
                    str(e) or f"Request timed out after {self.timeout}s",
                    provider="ollama",
                    retryable=True,
                ) from e
            except (ConnectionError, OSError) as e:
                raise ProviderUnavailableError(
                    str(e), provider="ollama", retryable=True
                ) from e
            except LLMError:
                raise
            except Exception as e:
                raise _translate_ollama_error(e) from e

        # Callback for retry events — signature matches amplifier-core's
        # retry_with_backoff on_retry contract: (attempt, delay, error)
        async def _on_retry(attempt: int, delay: float, error: LLMError) -> None:
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:retry",
                    {
                        "provider": self.name,
                        "attempt": attempt,
                        "max_retries": self._retry_config.max_retries,
                        "delay": delay,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )

        # Call Ollama API with retry_with_backoff for transient errors
        try:
            raw_response = await retry_with_backoff(
                _do_complete, self._retry_config, on_retry=_on_retry
            )
            # Convert Pydantic model to dict for consistent access
            response = (
                raw_response.model_dump()
                if hasattr(raw_response, "model_dump")
                else dict(raw_response)
            )
            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info("[PROVIDER] Received response from Ollama API")

            # Convert to OllamaChatResponse FIRST (before emitting llm:response)
            # so event usage fields come from the canonical ChatResponse
            chat_response = self._convert_to_chat_response(
                response, include_thinking=include_thinking
            )

            # Emit llm:response event using canonical usage fields from chat_response
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                event_usage: dict[str, Any] = {}
                if chat_response.usage:
                    event_usage["input_tokens"] = chat_response.usage.input_tokens
                    event_usage["output_tokens"] = chat_response.usage.output_tokens
                    if chat_response.usage.cache_read_tokens is not None:
                        event_usage["cache_read_tokens"] = (
                            chat_response.usage.cache_read_tokens
                        )

                response_payload: dict[str, Any] = {
                    "provider": "ollama",
                    "model": model,
                    "usage": event_usage,
                    "status": "ok",
                    "duration_ms": elapsed_ms,
                }
                if self.raw:
                    response_payload["raw"] = response
                await self.coordinator.hooks.emit("llm:response", response_payload)

            return chat_response

        except LLMError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[PROVIDER] Ollama API error: {e}")

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "ollama",
                        "model": model,
                        "status": "timeout"
                        if isinstance(e, LLMTimeoutError)
                        else "error",
                        "duration_ms": elapsed_ms,
                        "error": str(e),
                    },
                )
            raise

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[PROVIDER] Ollama API error: {e}")

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "ollama",
                        "model": model,
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": str(e),
                    },
                )
            raise _translate_ollama_error(e) from e

    async def _complete_streaming(
        self, request: ChatRequest, **kwargs
    ) -> OllamaChatResponse:
        """Handle streaming completion with event emission.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            OllamaChatResponse with accumulated content
        """
        logger.info(
            f"[PROVIDER] Streaming request with {len(request.messages)} messages"
        )

        # Validate tool call sequences (same as non-streaming)
        missing = self._find_missing_tool_results(request.messages)

        if missing:
            logger.warning(
                f"[PROVIDER] Ollama: Detected {len(missing)} missing tool result(s). "
                f"Injecting synthetic errors. This indicates a bug in context management. "
                f"Tool IDs: {[call_id for _, call_id, _, _ in missing]}"
            )

            # Insert synthetic results at the correct positions in request.messages,
            # then track repaired IDs to prevent infinite loops.
            self._apply_jit_repair(request, missing)

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name}
                            for _, call_id, tool_name, _ in missing
                        ],
                    },
                )

        # Separate messages by role
        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [
            m for m in request.messages if m.role in ("user", "assistant", "tool")
        ]

        # Build ollama messages list
        ollama_messages = []

        for sys_msg in system_msgs:
            content = sys_msg.content if isinstance(sys_msg.content, str) else ""
            ollama_messages.append({"role": "system", "content": content})

        for dev_msg in developer_msgs:
            content = dev_msg.content if isinstance(dev_msg.content, str) else ""
            wrapped = f"<context_file>\n{content}\n</context_file>"
            ollama_messages.append({"role": "user", "content": wrapped})

        # Convert conversation messages (synthetics are already in request.messages
        # at the correct positions from _apply_jit_repair above)
        conversation_msgs = self._convert_messages(
            [m.model_dump() for m in conversation]
        )
        ollama_messages.extend(conversation_msgs)

        # Prepare request parameters
        model = kwargs.get("model", self.default_model)

        # Ensure model is available (auto-pull if configured)
        if self.auto_pull:
            await self._ensure_model_available(model)

        # Build options dict with 3-tier precedence: request -> kwargs -> instance default
        options: dict[str, Any] = {
            "temperature": request.temperature
            or kwargs.get("temperature", self.temperature),
            "num_predict": request.max_output_tokens
            or kwargs.get("max_tokens", self.max_tokens),
        }

        # Sampling parameters - only include when explicitly set
        if (top_p := kwargs.get("top_p", self.top_p)) is not None:
            options["top_p"] = top_p
        if (top_k := kwargs.get("top_k", self.top_k)) is not None:
            options["top_k"] = top_k
        if (min_p := kwargs.get("min_p", self.min_p)) is not None:
            options["min_p"] = min_p
        if (
            repeat_penalty := kwargs.get("repeat_penalty", self.repeat_penalty)
        ) is not None:
            options["repeat_penalty"] = repeat_penalty
        if (seed := kwargs.get("seed", self.seed)) is not None:
            options["seed"] = seed

        # Set context window size (num_ctx controls how much context ollama uses)
        if self.num_ctx > 0:
            options["num_ctx"] = self.num_ctx
        else:
            ctx_length = await self._get_model_context_length(model)
            options["num_ctx"] = ctx_length

        params: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "options": options,
            "stream": True,
        }

        # Stop sequences - top-level param for Ollama SDK
        if (stop := kwargs.get("stop", self.stop)) is not None:
            params["stop"] = stop

        # Keep model loaded in memory
        if self.keep_alive is not None:
            params["keep_alive"] = self.keep_alive

        # Logprobs support (Ollama >= 0.12.11) - top-level params
        if logprobs := kwargs.get("logprobs", self.logprobs):
            params["logprobs"] = logprobs
        if top_logprobs := kwargs.get("top_logprobs", self.top_logprobs):
            params["top_logprobs"] = top_logprobs

        # Add tools if provided
        if request.tools:
            params["tools"] = self._format_tools_from_request(request.tools)

        # Add structured output format if specified
        if hasattr(request, "response_format") and request.response_format:
            if isinstance(request.response_format, dict):
                params["format"] = request.response_format
            elif request.response_format == "json":
                params["format"] = "json"

        # Enable thinking/reasoning only for models that support it
        # think is a top-level parameter (not inside options) since Ollama v0.9.0
        #
        # Precedence: kwargs/request.enable_thinking → request.reasoning_effort
        #             → provider config (self.enable_thinking) → default off
        include_thinking = False
        model_caps = self._detect_model_capabilities(model)
        if "thinking" in model_caps:
            if hasattr(request, "enable_thinking") and request.enable_thinking:  # pyright: ignore[reportAttributeAccessIssue]
                params["think"] = self.thinking_effort or True
                include_thinking = True
            elif request.reasoning_effort is not None:  # pyright: ignore[reportAttributeAccessIssue]
                # Ollama v0.9.0+ supports effort levels ("high", "medium", "low")
                # via the `think` parameter — pass through directly.
                params["think"] = request.reasoning_effort  # pyright: ignore[reportAttributeAccessIssue]
                include_thinking = True
            elif self.enable_thinking:
                params["think"] = self.thinking_effort or True
                include_thinking = True

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            stream_request_payload: dict[str, Any] = {
                "provider": "ollama",
                "model": model,
                "message_count": len(ollama_messages),
                "stream": True,
            }
            if self.raw:
                stream_request_payload["raw"] = params
            await self.coordinator.hooks.emit("llm:request", stream_request_payload)

        start_time = time.time()
        accumulated_content = ""
        accumulated_thinking = ""
        accumulated_tool_calls: list[dict[str, Any]] = []
        final_chunk: dict[str, Any] | None = None

        # Inner function: wraps the initial stream connection with error
        # translation so that retry_with_backoff sees LLMError subclasses and
        # can check .retryable to decide whether to retry.
        async def _do_stream_connect():
            try:
                return await asyncio.wait_for(
                    self.client.chat(**params), timeout=self.timeout
                )
            except ResponseError as e:
                raise _translate_ollama_error(e) from e
            except (asyncio.TimeoutError, TimeoutError) as e:
                raise LLMTimeoutError(
                    str(e) or f"Request timed out after {self.timeout}s",
                    provider="ollama",
                    retryable=True,
                ) from e
            except (ConnectionError, OSError) as e:
                raise ProviderUnavailableError(
                    str(e), provider="ollama", retryable=True
                ) from e
            except LLMError:
                raise
            except Exception as e:
                raise _translate_ollama_error(e) from e

        # Callback for retry events — signature matches amplifier-core's
        # retry_with_backoff on_retry contract: (attempt, delay, error)
        async def _on_retry(attempt: int, delay: float, error: LLMError) -> None:
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:retry",
                    {
                        "provider": self.name,
                        "attempt": attempt,
                        "max_retries": self._retry_config.max_retries,
                        "delay": delay,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )

        try:
            # Retry covers the initial stream connection only.
            stream = await retry_with_backoff(
                _do_stream_connect, self._retry_config, on_retry=_on_retry
            )

            # Mid-stream errors are NOT retried — they fall through to
            # the outer except blocks below.
            async for chunk in stream:
                message = chunk.get("message", {})

                # Handle content chunks
                if message.get("content"):
                    accumulated_content += message["content"]
                    if self.coordinator and hasattr(self.coordinator, "hooks"):
                        await self.coordinator.hooks.emit(
                            "llm:stream:chunk",
                            {"content": message["content"], "provider": "ollama"},
                        )

                # Handle thinking chunks
                if message.get("thinking"):
                    accumulated_thinking += message["thinking"]
                    if self.coordinator and hasattr(self.coordinator, "hooks"):
                        await self.coordinator.hooks.emit(
                            "llm:stream:thinking",
                            {"thinking": message["thinking"], "provider": "ollama"},
                        )

                # Accumulate tool calls from streaming chunks (supported since Ollama v0.8.0)
                if message.get("tool_calls"):
                    for tc in message["tool_calls"]:
                        accumulated_tool_calls.append(tc)

                if chunk.get("done"):
                    final_chunk = chunk

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info("[PROVIDER] Streaming complete")

            # Build final response FIRST (before emitting llm:response)
            # so event usage fields come from the canonical ChatResponse
            chat_response = self._build_streaming_response(
                accumulated_content,
                accumulated_thinking,
                accumulated_tool_calls,
                final_chunk,
                include_thinking,
            )

            # Emit llm:response event using canonical usage fields from chat_response
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                event_usage: dict[str, Any] = {}
                if chat_response.usage:
                    event_usage["input_tokens"] = chat_response.usage.input_tokens
                    event_usage["output_tokens"] = chat_response.usage.output_tokens
                    if chat_response.usage.cache_read_tokens is not None:
                        event_usage["cache_read_tokens"] = (
                            chat_response.usage.cache_read_tokens
                        )

                stream_response_payload: dict[str, Any] = {
                    "provider": "ollama",
                    "model": model,
                    "usage": event_usage,
                    "status": "ok",
                    "duration_ms": elapsed_ms,
                    "stream": True,
                }
                if self.raw:
                    stream_response_payload["raw"] = {
                        "content": accumulated_content,
                        "thinking": accumulated_thinking
                        if accumulated_thinking
                        else None,
                        "final_chunk": final_chunk,
                    }
                await self.coordinator.hooks.emit(
                    "llm:response", stream_response_payload
                )

            return chat_response

        except LLMError as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[PROVIDER] Streaming error: {e}")

            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "ollama",
                        "model": model,
                        "status": "timeout"
                        if isinstance(e, LLMTimeoutError)
                        else "error",
                        "duration_ms": elapsed_ms,
                        "error": str(e),
                        "stream": True,
                    },
                )
            raise

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[PROVIDER] Streaming error: {e}")

            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "ollama",
                        "model": model,
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": str(e),
                        "stream": True,
                    },
                )
            raise _translate_ollama_error(e) from e

    def _build_streaming_response(
        self,
        content: str,
        thinking: str,
        accumulated_tool_calls: list[dict[str, Any]],
        final_chunk: dict[str, Any] | None,
        include_thinking: bool,
    ) -> OllamaChatResponse:
        """Build final response from streamed chunks.

        Args:
            content: Accumulated content text
            thinking: Accumulated thinking text
            accumulated_tool_calls: Tool calls accumulated during streaming
            final_chunk: Final chunk with usage info
            include_thinking: Whether thinking was requested

        Returns:
            OllamaChatResponse with accumulated content and tool calls
        """
        from amplifier_core.message_models import TextBlock
        from amplifier_core.message_models import ToolCallBlock
        from amplifier_core.message_models import Usage

        content_blocks = []  # For context storage (message_models: ThinkingBlock, TextBlock, etc.)
        event_blocks: list[
            TextContent | ThinkingContent | ToolCallContent
        ] = []  # For streaming UI events
        tool_calls = []
        thinking_content = None

        # Add thinking block if present (always include when model returns it)
        if thinking:
            thinking_content = thinking
            content_blocks.append(
                ThinkingBlock(
                    thinking=thinking,
                    signature=None,
                )
            )
            # Also add to event_blocks for streaming UI hooks
            event_blocks.append(ThinkingContent(text=thinking))

        # Add text content
        if content:
            content_blocks.append(TextBlock(text=content))
            # Also add to event_blocks for streaming UI hooks
            event_blocks.append(TextContent(text=content))

        # Process accumulated tool calls (same parsing as non-streaming path)
        for tc in accumulated_tool_calls:
            function = (
                tc.get("function", {})
                if isinstance(tc, dict)
                else getattr(tc, "function", {})
            )
            if isinstance(function, dict):
                tool_name = function.get("name", "")
                tool_args = function.get("arguments", {})
            else:
                tool_name = getattr(function, "name", "")
                tool_args = getattr(function, "arguments", {})
            tool_id = (
                tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
            ) or f"call_{uuid4().hex[:8]}"

            content_blocks.append(
                ToolCallBlock(id=tool_id, name=tool_name, input=tool_args)
            )
            tool_calls.append(ToolCall(id=tool_id, name=tool_name, arguments=tool_args))
            event_blocks.append(
                ToolCallContent(id=tool_id, name=tool_name, arguments=tool_args)
            )

        # Extract usage from final chunk
        # NOTE: reasoning_tokens is always None for Ollama because eval_count
        # includes both reasoning and visible output tokens — Ollama does not
        # report them separately.
        usage = Usage(
            input_tokens=final_chunk.get("prompt_eval_count", 0) if final_chunk else 0,
            output_tokens=final_chunk.get("eval_count", 0) if final_chunk else 0,
            total_tokens=(
                (
                    final_chunk.get("prompt_eval_count", 0)
                    + final_chunk.get("eval_count", 0)
                )
                if final_chunk
                else 0
            ),
        )

        return OllamaChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=final_chunk.get("done_reason") if final_chunk else None,
            raw_response=final_chunk if self.raw else None,
            model_name=final_chunk.get("model") if final_chunk else None,
            thinking_content=thinking_content,
            content_blocks=event_blocks
            if event_blocks
            else None,  # For streaming UI events
            text=content or None,
        )

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """
        Parse tool calls from provider response.

        Args:
            response: Provider response

        Returns:
            List of tool calls
        """
        return response.tool_calls or []

    def _find_missing_tool_results(
        self, messages: list[Message]
    ) -> list[tuple[int, str, str, dict]]:
        """Find tool calls without corresponding results.

        Scans message history to detect tool calls that were never answered
        with a tool result message.

        Filters out tool call IDs that have already been repaired with synthetic
        results to prevent infinite detection loops across LLM iterations.

        Args:
            messages: List of conversation messages

        Returns:
            List of (msg_index, call_id, tool_name, tool_arguments) tuples for
            unpaired calls, where msg_index is the index of the assistant message
            that made the tool call.
        """
        tool_calls: dict[
            str, tuple[int, str, dict]
        ] = {}  # {call_id: (msg_idx, name, args)}
        tool_results: set[str] = set()  # {call_id}

        for msg_idx, msg in enumerate(messages):
            if msg.role == "assistant":
                # Check for tool calls in content blocks
                if hasattr(msg, "content") and isinstance(msg.content, list):
                    for block in msg.content:
                        if hasattr(block, "type") and block.type == "tool_use":
                            block_id = getattr(block, "id", "")
                            block_name = getattr(block, "name", "unknown")
                            block_input = getattr(block, "input", {})
                            if block_id:
                                tool_calls[block_id] = (
                                    msg_idx,
                                    block_name,
                                    block_input,
                                )
                        elif hasattr(block, "id") and hasattr(block, "name"):
                            # ToolCallBlock style
                            block_id = getattr(block, "id", "")
                            block_name = getattr(block, "name", "unknown")
                            block_input = getattr(block, "input", {})
                            if block_id:
                                tool_calls[block_id] = (
                                    msg_idx,
                                    block_name,
                                    block_input,
                                )
                # Also check tool_calls field
                if hasattr(msg, "tool_calls") and msg.tool_calls:  # pyright: ignore[reportAttributeAccessIssue]
                    for tc in msg.tool_calls:  # pyright: ignore[reportAttributeAccessIssue]
                        tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
                        tc_name = (
                            tc.name
                            if hasattr(tc, "name")
                            else tc.get("name", "unknown")
                        )
                        tc_args = (
                            tc.arguments
                            if hasattr(tc, "arguments")
                            else tc.get("arguments", {})
                        )
                        if tc_id:
                            tool_calls[tc_id] = (msg_idx, tc_name, tc_args)
            elif msg.role == "tool":
                # Tool result - mark as received
                tool_call_id = msg.tool_call_id if hasattr(msg, "tool_call_id") else ""
                if tool_call_id:
                    tool_results.add(tool_call_id)

        # Bound memory: clear tracking set if it grows too large
        if len(self._repaired_tool_ids) > 1000:
            self._repaired_tool_ids.clear()

        # Exclude IDs that have already been repaired to prevent infinite loops
        return [
            (msg_idx, call_id, name, args)
            for call_id, (msg_idx, name, args) in tool_calls.items()
            if call_id not in tool_results and call_id not in self._repaired_tool_ids
        ]

    def _create_synthetic_result_message(self, call_id: str, tool_name: str) -> Message:
        """Create synthetic error result Message for insertion into request.messages.

        Returns a Message object suitable for direct insertion into the
        request.messages list before Ollama format conversion.

        Args:
            call_id: The ID of the tool call that needs a result
            tool_name: The name of the tool that was called

        Returns:
            Message with role="tool" containing the synthetic error content
        """
        return Message(
            role="tool",
            tool_call_id=call_id,
            content=(
                f"[SYSTEM ERROR: Tool result missing from conversation history]\n\n"
                f"Tool: {tool_name}\n"
                f"Call ID: {call_id}\n\n"
                f"This indicates the tool result was lost after execution.\n"
                f"Likely causes: context compaction bug, message parsing error, or state corruption.\n\n"
                f"The tool may have executed successfully, but the result was lost.\n"
                f"Please acknowledge this error and offer to retry the operation."
            ),
        )

    def _apply_jit_repair(
        self,
        request: "ChatRequest",
        missing: list[tuple[int, str, str, dict]],
    ) -> None:
        """Insert synthetic tool results at the correct position in request.messages.

        Mutates request.messages in-place. For each assistant message that has
        unpaired tool calls, synthetic tool result messages are inserted immediately
        after that assistant message (FM1/FM2). If the next message after the
        inserted synthetics is a real user message, a synthetic assistant response
        is also inserted (FM3) to close the tool turn properly.

        Processing is done in reverse index order so earlier insertions don't
        shift the indices of later ones.

        Args:
            request: ChatRequest whose messages list will be mutated in-place
            missing: Output of _find_missing_tool_results() —
                     list of (msg_index, call_id, tool_name, tool_args)
        """
        by_msg_idx: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for msg_idx, call_id, tool_name, _ in missing:
            by_msg_idx[msg_idx].append((call_id, tool_name))

        for msg_idx in sorted(by_msg_idx.keys(), reverse=True):
            synthetics: list[Message] = []
            for call_id, tool_name in by_msg_idx[msg_idx]:
                synthetics.append(
                    self._create_synthetic_result_message(call_id, tool_name)
                )
                self._repaired_tool_ids.add(call_id)

            insert_pos = msg_idx + 1
            for i, synthetic in enumerate(synthetics):
                request.messages.insert(insert_pos + i, synthetic)

            # FM3: if the message immediately after the injected synthetics is a
            # real user message, insert a synthetic assistant response to close
            # the tool turn properly before the user speaks again.
            next_pos = insert_pos + len(synthetics)
            if (
                next_pos < len(request.messages)
                and request.messages[next_pos].role == "user"
            ):
                fm3_msg = Message(
                    role="assistant",
                    content=(
                        "[SYSTEM NOTE: Previous tool results were missing from "
                        "conversation history. Synthetic error responses were "
                        "injected to maintain valid conversation structure.]"
                    ),
                )
                request.messages.insert(next_pos, fm3_msg)

    def _create_synthetic_result(self, call_id: str, tool_name: str) -> dict[str, Any]:
        """Create synthetic error result for missing tool response.

        This is a BACKUP for when tool results go missing AFTER execution.
        The orchestrator should handle tool execution errors at runtime,
        so this should only trigger on context/parsing bugs.

        Args:
            call_id: The ID of the tool call that needs a result
            tool_name: The name of the tool that was called

        Returns:
            Dict in tool message format with error content
        """
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": (
                f"[SYSTEM ERROR: Tool result missing from conversation history]\n\n"
                f"Tool: {tool_name}\n"
                f"Call ID: {call_id}\n\n"
                f"This indicates the tool result was lost after execution.\n"
                f"Likely causes: context compaction bug, message parsing error, or state corruption.\n\n"
                f"The tool may have executed successfully, but the result was lost.\n"
                f"Please acknowledge this error and offer to retry the operation."
            ),
        }

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert Amplifier message format to Ollama/OpenAI format.

        Handles the conversion of:
        - Tool calls in assistant messages (Amplifier format -> OpenAI format)
        - Tool result messages
        - Developer messages (converted to XML-wrapped user messages)
        - Regular user/assistant/system messages
        - Structured content blocks (list of text/image blocks) -> plain string
        """
        ollama_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            # Handle structured content (list of content blocks from Amplifier)
            # Convert to plain string for Ollama which expects string content
            # Extract base64 images for multimodal models (vision)
            images: list[str] = []
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        # TextContent block: {"type": "text", "text": "..."}
                        if block.get("type") == "text" and "text" in block:
                            text_parts.append(block["text"])
                        # Image content block: {"type": "image", "source": {"type": "base64", "data": "..."}}
                        elif block.get("type") == "image":
                            source = block.get("source", {})
                            if source.get("type") == "base64":
                                images.append(source.get("data", ""))
                            elif source.get("type") == "url":
                                text_parts.append(
                                    f"[Image URL: {source.get('url', '')}]"
                                )
                        # ToolCallContent, ThinkingContent, etc. - handled by role-specific logic
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts) if text_parts else ""

            if role == "developer":
                # Developer messages -> XML-wrapped user messages (context files)
                wrapped = f"<context_file>\n{content}\n</context_file>"
                ollama_messages.append({"role": "user", "content": wrapped})

            elif role == "assistant":
                # Check for tool_calls in Amplifier format
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Convert Amplifier tool_calls to OpenAI format
                    ollama_tool_calls = []
                    for tc in msg["tool_calls"]:
                        ollama_tool_calls.append(
                            {
                                "id": tc.get("id", ""),
                                "type": "function",  # OpenAI requires this
                                "function": {
                                    "name": tc.get("tool", ""),
                                    "arguments": tc.get("arguments", {}),
                                },
                            }
                        )

                    ollama_messages.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": ollama_tool_calls,
                        }
                    )
                else:
                    # Regular assistant message
                    ollama_messages.append({"role": "assistant", "content": content})

            elif role == "tool":
                # Tool result message
                ollama_messages.append(
                    {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": msg.get("tool_call_id", ""),
                    }
                )

            else:
                # User, system, etc. - build message with optional images
                out_msg: dict[str, Any] = {"role": role, "content": content}
                if images:
                    out_msg["images"] = images
                ollama_messages.append(out_msg)

        return ollama_messages

    def _format_tools_for_ollama(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to Ollama format."""
        ollama_tools = []

        for tool in tools:
            # Get schema from tool if available
            input_schema = getattr(
                tool,
                "input_schema",
                {"type": "object", "properties": {}, "required": []},
            )

            ollama_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": input_schema,
                    },
                }
            )

        return ollama_tools

    def _format_tools_from_request(self, tools: list) -> list[dict[str, Any]]:
        """Convert ToolSpec objects from ChatRequest to Ollama format.

        Args:
            tools: List of ToolSpec objects

        Returns:
            List of Ollama-formatted tool definitions
        """
        ollama_tools = []
        for tool in tools:
            ollama_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.parameters,
                    },
                }
            )
        return ollama_tools

    def _convert_to_chat_response(
        self, response: Any, include_thinking: bool = False
    ) -> OllamaChatResponse:
        """Convert Ollama response to OllamaChatResponse format.

        Args:
            response: Ollama API response
            include_thinking: Whether to include thinking content in response

        Returns:
            OllamaChatResponse with content blocks and optional thinking
        """
        from amplifier_core.message_models import TextBlock
        from amplifier_core.message_models import ToolCall
        from amplifier_core.message_models import ToolCallBlock
        from amplifier_core.message_models import Usage

        content_blocks = []  # For context storage (message_models: ThinkingBlock, TextBlock, etc.)
        event_blocks: list[
            TextContent | ThinkingContent | ToolCallContent
        ] = []  # For streaming UI events
        tool_calls = []
        thinking_content = None
        text_accumulator: list[str] = []

        message = response.get("message", {})
        content = message.get("content", "")
        thinking = message.get("thinking", "")

        # Add thinking block if present (always include when model returns it)
        if thinking:
            thinking_content = thinking
            content_blocks.append(
                ThinkingBlock(
                    thinking=thinking,
                    signature=None,  # Ollama doesn't provide signatures
                )
            )
            # Also add to event_blocks for streaming UI hooks
            event_blocks.append(ThinkingContent(text=thinking))

        # Add text content if present
        if content:
            content_blocks.append(TextBlock(text=content))
            text_accumulator.append(content)
            # Also add to event_blocks for streaming UI hooks
            event_blocks.append(TextContent(text=content))

        # Parse tool calls if present (check both key exists and value is not None)
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                function = tc.get("function", {})
                tool_id = tc.get("id", "") or f"call_{uuid4().hex[:8]}"
                tool_name = function.get("name", "")
                tool_args = function.get("arguments", {})

                content_blocks.append(
                    ToolCallBlock(id=tool_id, name=tool_name, input=tool_args)
                )
                tool_calls.append(
                    ToolCall(id=tool_id, name=tool_name, arguments=tool_args)
                )
                # Also add to event_blocks for streaming UI hooks
                event_blocks.append(
                    ToolCallContent(id=tool_id, name=tool_name, arguments=tool_args)
                )

        # Build usage info
        # NOTE: reasoning_tokens is always None for Ollama because eval_count
        # includes both reasoning and visible output tokens — Ollama does not
        # report them separately.
        usage = Usage(
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            total_tokens=response.get("prompt_eval_count", 0)
            + response.get("eval_count", 0),
        )

        combined_text = "\n\n".join(text_accumulator).strip()

        return OllamaChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=response.get("done_reason") or None,
            raw_response=response if self.raw else None,
            model_name=response.get("model"),
            thinking_content=thinking_content,
            content_blocks=event_blocks
            if event_blocks
            else None,  # For streaming UI events
            text=combined_text or None,
        )

    async def close(self) -> None:
        """Release the Ollama client reference.

        Note: ollama.AsyncClient does not expose a close() method.
        Releasing the reference allows GC to clean up the underlying
        httpx transport while the event loop is still alive.
        """
        self._client = None
