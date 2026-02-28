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


class OllamaChatResponse(ChatResponse):
    """Extended ChatResponse with Ollama-specific metadata."""

    raw_response: dict[str, Any] | None = None
    model_name: str | None = None
    thinking_content: str | None = None
    # content_blocks for streaming UI compatibility (triggers content_block:start/end events)
    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


def _truncate_values(
    obj: Any,
    max_length: int = 200,
    max_depth: int = 10,
    _depth: int = 0,
) -> Any:
    """Truncate long strings in nested structures for logging.

    Args:
        obj: Object to truncate (dict, list, str, or other)
        max_length: Maximum length for strings before truncation
        max_depth: Maximum recursion depth
        _depth: Current recursion depth (internal)

    Returns:
        Truncated copy of the object
    """
    if _depth > max_depth:
        return "..."

    if isinstance(obj, str):
        if len(obj) > max_length:
            return obj[:max_length] + f"... ({len(obj)} chars)"
        return obj
    if isinstance(obj, dict):
        return {
            k: _truncate_values(v, max_length, max_depth, _depth + 1)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        if len(obj) > 10:
            truncated = [
                _truncate_values(item, max_length, max_depth, _depth + 1)
                for item in obj[:10]
            ]
            return truncated + [f"... ({len(obj)} items total)"]
        return [
            _truncate_values(item, max_length, max_depth, _depth + 1) for item in obj
        ]
    return obj


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
            - host: Ollama server URL (default: from OLLAMA_HOST or http://localhost:11434)
            - default_model: Model to use (default: "llama3.2:3b")
            - max_tokens: Maximum tokens (default: 4096)
            - temperature: Generation temperature (default: 0.7)
            - timeout: Request timeout in seconds (default: 120)
            - auto_pull: Whether to auto-pull missing models (default: False)

    Returns:
        Optional cleanup function
    """
    config = config or {}

    # Get configuration with defaults
    host = config.get("host", os.environ.get("OLLAMA_HOST", "http://localhost:11434"))

    provider = OllamaProvider(host, config, coordinator)
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
    """Ollama local LLM integration."""

    name = "ollama"
    api_label = "Ollama"

    def __init__(
        self,
        host: str | None = None,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
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
        self._client: AsyncClient | None = None  # Lazy init
        self.config = config or {}
        self.coordinator = coordinator

        # Configuration with sensible defaults
        self.default_model = self.config.get("default_model", "llama3.2:3b")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.7)
        self.timeout = float(
            self.config.get("timeout", 600.0)
        )  # API timeout in seconds (default 10 min - local models need longer for prefill)
        self.auto_pull = self.config.get("auto_pull", False)
        self.debug = self.config.get("debug", False)
        self.raw_debug = self.config.get(
            "raw_debug", False
        )  # Enable ultra-verbose raw API I/O logging
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
        # Backward compat: retry_jitter=True → 0.2, False → 0.0, float → as-is
        raw_jitter = self.config.get("retry_jitter", 0.2)
        if isinstance(raw_jitter, bool):
            jitter = 0.2 if raw_jitter else 0.0
        else:
            jitter = float(raw_jitter)
        self._retry_config = RetryConfig(
            max_retries=int(self.config.get("max_retries", 3)),
            min_delay=float(self.config.get("min_retry_delay", 1.0)),
            max_delay=float(self.config.get("max_retry_delay", 60.0)),
            jitter=jitter,
        )

    @property
    def client(self) -> AsyncClient:
        """Lazily initialize the Ollama client on first access."""
        if self._client is None:
            if self.host is None:
                raise ValueError("host must be provided for API calls")
            self._client = AsyncClient(host=self.host)
        return self._client

    def get_info(self) -> ProviderInfo:
        """Get provider metadata."""
        return ProviderInfo(
            id="ollama",
            display_name="Ollama",
            credential_env_vars=[],  # No API key needed for local Ollama
            capabilities=["streaming", "tools", "local"],
            defaults={
                "model": "llama3.2:3b",
                "max_tokens": 4096,
                "temperature": 0.7,
                "timeout": 600.0,
                "context_window": 128000,
                "max_output_tokens": 64000,
            },
            config_fields=[
                ConfigField(
                    id="host",
                    display_name="Ollama Host",
                    field_type="text",
                    prompt="Ollama server URL",
                    env_var="OLLAMA_HOST",
                    default="http://localhost:11434",
                    required=False,
                ),
                ConfigField(
                    id="auto_pull",
                    display_name="Auto-Pull Models",
                    field_type="boolean",
                    prompt="Automatically pull missing models?",
                    default="false",
                    required=False,
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
        caps = ["streaming", "local"]

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
                        cost_per_input_token=0.0,
                        cost_per_output_token=0.0,
                        metadata={"cost_tier": "free"},
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
        """Check if model is available, attempt to pull if not and auto_pull is enabled."""
        try:
            # Try to get model info
            await self.client.show(model)
            return True
        except ResponseError as e:
            if e.status_code == 404:
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
        extra_tool_messages: list[dict[str, Any]] = []

        if missing:
            logger.warning(
                f"[PROVIDER] Ollama: Detected {len(missing)} missing tool result(s). "
                f"Injecting synthetic errors. This indicates a bug in context management. "
                f"Tool IDs: {[call_id for call_id, _, _ in missing]}"
            )

            # Inject synthetic results and track repaired IDs to prevent infinite loops
            for call_id, tool_name, _ in missing:
                extra_tool_messages.append(
                    self._create_synthetic_result(call_id, tool_name)
                )
                # Track this ID so we don't detect it as missing again in future iterations
                self._repaired_tool_ids.add(call_id)

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name}
                            for call_id, tool_name, _ in missing
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

        # Convert conversation messages
        conversation_msgs = self._convert_messages(
            [m.model_dump() for m in conversation]
        )
        ollama_messages.extend(conversation_msgs)

        # Append synthetic tool results for any missing tool calls
        ollama_messages.extend(extra_tool_messages)

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
            # INFO level: Summary only
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "provider": "ollama",
                    "model": model,
                    "message_count": len(ollama_messages),
                },
            )

            # DEBUG level: Truncated request payload (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "provider": "ollama",
                        "request": _truncate_values(
                            {
                                "model": model,
                                "messages": ollama_messages,
                                "options": params.get("options", {}),
                            }
                        ),
                    },
                )

            # RAW level: Full request payload (if raw_debug enabled)
            if self.raw_debug:
                await self.coordinator.hooks.emit(
                    "llm:request:raw",
                    {
                        "lvl": "DEBUG",
                        "provider": "ollama",
                        "request": params,
                    },
                )

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

            # Emit llm:response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # Build usage info
                usage_info = {}
                if "prompt_eval_count" in response:
                    usage_info["input"] = response.get("prompt_eval_count", 0)
                if "eval_count" in response:
                    usage_info["output"] = response.get("eval_count", 0)

                # INFO level: Summary only
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "ollama",
                        "model": model,
                        "usage": usage_info,
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                    },
                )

                # DEBUG level: Truncated response (if debug enabled)
                if self.debug:
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "provider": "ollama",
                            "response": _truncate_values(response),
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

                # RAW level: Full response (if raw_debug enabled)
                if self.raw_debug:
                    await self.coordinator.hooks.emit(
                        "llm:response:raw",
                        {
                            "lvl": "DEBUG",
                            "provider": "ollama",
                            "response": response,
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

            # Convert to OllamaChatResponse
            return self._convert_to_chat_response(
                response, include_thinking=include_thinking
            )

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
        extra_tool_messages: list[dict[str, Any]] = []

        if missing:
            logger.warning(
                f"[PROVIDER] Ollama: Detected {len(missing)} missing tool result(s). "
                f"Injecting synthetic errors. This indicates a bug in context management. "
                f"Tool IDs: {[call_id for call_id, _, _ in missing]}"
            )

            # Inject synthetic results and track repaired IDs to prevent infinite loops
            for call_id, tool_name, _ in missing:
                extra_tool_messages.append(
                    self._create_synthetic_result(call_id, tool_name)
                )
                # Track this ID so we don't detect it as missing again in future iterations
                self._repaired_tool_ids.add(call_id)

            # Emit observability event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name}
                            for call_id, tool_name, _ in missing
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

        conversation_msgs = self._convert_messages(
            [m.model_dump() for m in conversation]
        )
        ollama_messages.extend(conversation_msgs)
        ollama_messages.extend(extra_tool_messages)

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
            # INFO level: Summary only
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "provider": "ollama",
                    "model": model,
                    "message_count": len(ollama_messages),
                    "stream": True,
                },
            )

            # DEBUG level: Truncated request payload (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "provider": "ollama",
                        "stream": True,
                        "request": _truncate_values(
                            {
                                "model": model,
                                "messages": ollama_messages,
                                "options": params.get("options", {}),
                            }
                        ),
                    },
                )

            # RAW level: Full request payload (if raw_debug enabled)
            if self.raw_debug:
                await self.coordinator.hooks.emit(
                    "llm:request:raw",
                    {
                        "lvl": "DEBUG",
                        "provider": "ollama",
                        "stream": True,
                        "request": params,
                    },
                )

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

            # Emit completion event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                usage_info = {}
                if final_chunk:
                    if "prompt_eval_count" in final_chunk:
                        usage_info["input"] = final_chunk.get("prompt_eval_count", 0)
                    if "eval_count" in final_chunk:
                        usage_info["output"] = final_chunk.get("eval_count", 0)

                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "ollama",
                        "model": model,
                        "usage": usage_info,
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                        "stream": True,
                    },
                )

                # DEBUG level: Truncated response (if debug enabled)
                if self.debug:
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "provider": "ollama",
                            "stream": True,
                            "response": _truncate_values(
                                {
                                    "content": accumulated_content,
                                    "thinking": accumulated_thinking
                                    if accumulated_thinking
                                    else None,
                                    "final_chunk": final_chunk,
                                }
                            ),
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

                # RAW level: Full response (if raw_debug enabled)
                if self.raw_debug:
                    await self.coordinator.hooks.emit(
                        "llm:response:raw",
                        {
                            "lvl": "DEBUG",
                            "provider": "ollama",
                            "stream": True,
                            "response": {
                                "content": accumulated_content,
                                "thinking": accumulated_thinking
                                if accumulated_thinking
                                else None,
                                "final_chunk": final_chunk,
                            },
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

            # Build final response
            return self._build_streaming_response(
                accumulated_content,
                accumulated_thinking,
                accumulated_tool_calls,
                final_chunk,
                include_thinking,
            )

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
            raw_response=final_chunk if self.raw_debug else None,
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
    ) -> list[tuple[str, str, dict]]:
        """Find tool calls without corresponding results.

        Scans message history to detect tool calls that were never answered
        with a tool result message.

        Filters out tool call IDs that have already been repaired with synthetic
        results to prevent infinite detection loops across LLM iterations.

        Args:
            messages: List of conversation messages

        Returns:
            List of (call_id, tool_name, tool_arguments) tuples for unpaired calls
        """
        tool_calls: dict[str, tuple[str, dict]] = {}  # {call_id: (name, args)}
        tool_results: set[str] = set()  # {call_id}

        for msg in messages:
            if msg.role == "assistant":
                # Check for tool calls in content blocks
                if hasattr(msg, "content") and isinstance(msg.content, list):
                    for block in msg.content:
                        if hasattr(block, "type") and block.type == "tool_use":
                            block_id = getattr(block, "id", "")
                            block_name = getattr(block, "name", "unknown")
                            block_input = getattr(block, "input", {})
                            if block_id:
                                tool_calls[block_id] = (block_name, block_input)
                        elif hasattr(block, "id") and hasattr(block, "name"):
                            # ToolCallBlock style
                            block_id = getattr(block, "id", "")
                            block_name = getattr(block, "name", "unknown")
                            block_input = getattr(block, "input", {})
                            if block_id:
                                tool_calls[block_id] = (block_name, block_input)
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
                            tool_calls[tc_id] = (tc_name, tc_args)
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
            (call_id, name, args)
            for call_id, (name, args) in tool_calls.items()
            if call_id not in tool_results and call_id not in self._repaired_tool_ids
        ]

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
            raw_response=response if self.raw_debug else None,
            model_name=response.get("model"),
            thinking_content=thinking_content,
            content_blocks=event_blocks
            if event_blocks
            else None,  # For streaming UI events
            text=combined_text or None,
        )
