"""
Ollama provider module for Amplifier.
Integrates with local Ollama server for LLM completions.
"""

import asyncio
import logging
import os
import time
from typing import Any

from amplifier_core import ConfigField
from amplifier_core import ModelInfo
from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderInfo
from amplifier_core.content_models import TextContent
from amplifier_core.content_models import ThinkingContent
from amplifier_core.content_models import ToolCallContent
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Message
from amplifier_core.message_models import ThinkingBlock
from amplifier_core.message_models import ToolCall

from ollama import AsyncClient  # pyright: ignore[reportAttributeAccessIssue]
from ollama import ResponseError  # pyright: ignore[reportAttributeAccessIssue]

logger = logging.getLogger(__name__)


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
        return {k: _truncate_values(v, max_length, max_depth, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 10:
            truncated = [_truncate_values(item, max_length, max_depth, _depth + 1) for item in obj[:10]]
            return truncated + [f"... ({len(obj)} items total)"]
        return [_truncate_values(item, max_length, max_depth, _depth + 1) for item in obj]
    return obj


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
        logger.warning(f"Ollama server at {host} is not reachable. Provider mounted but will fail on use.")
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

    def __init__(self, host: str, config: dict[str, Any] | None = None, coordinator: ModuleCoordinator | None = None):
        """
        Initialize Ollama provider.

        Args:
            host: Ollama server URL
            config: Additional configuration
            coordinator: Module coordinator for event emission
        """
        self.host = host
        self.client = AsyncClient(host=host)
        self.config = config or {}
        self.coordinator = coordinator

        # Configuration with sensible defaults
        self.default_model = self.config.get("default_model", "llama3.2:3b")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.7)
        self.timeout = self.config.get("timeout", 300.0)  # API timeout in seconds (default 5 minutes)
        self.auto_pull = self.config.get("auto_pull", False)
        self.debug = self.config.get("debug", False)
        self.raw_debug = self.config.get("raw_debug", False)  # Enable ultra-verbose raw API I/O logging
        # Context window size (num_ctx in ollama) - 0 means auto-detect from model
        self.num_ctx = self.config.get("num_ctx", 0)
        # Cache for model context lengths (avoid repeated API calls)
        self._model_ctx_cache: dict[str, int] = {}
        # Enable thinking/reasoning for models that support it (default: True)
        # Models that don't support thinking will simply ignore this option
        self.enable_thinking = self.config.get("enable_thinking", True)

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
                "timeout": 300.0,
            },
            config_fields=[
                ConfigField(
                    id="host",
                    display_name="Ollama Host",
                    field_type="text",
                    prompt="Enter Ollama server URL",
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
            ],
        )

    async def list_models(self) -> list[ModelInfo]:
        """
        List available models from local Ollama server.

        Queries the Ollama API to get list of installed models.
        """
        try:
            response = await self.client.list()
            models = []
            # response.models is a list of Model objects with .model attribute (not .name)
            for model in response.models:
                model_name = model.model  # Model objects use .model, not .name
                if model_name:
                    # Extract details - model.details is a ModelDetails object
                    details = model.details
                    context_length = getattr(details, "context_length", None) or 4096
                    models.append(
                        ModelInfo(
                            id=model_name,
                            display_name=model_name,
                            context_window=context_length,
                            max_output_tokens=context_length,
                            capabilities=["tools", "streaming", "local"],
                            defaults={"temperature": 0.7, "max_tokens": 4096},
                        )
                    )
            return models
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            return []

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
                    logger.warning(f"Model {model} not found. Set auto_pull=True or run 'ollama pull {model}'")
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
            model_info = getattr(info, "modelinfo", None) or getattr(info, "model_info", None) or {}

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
        default_ctx = 8192
        self._model_ctx_cache[model] = default_ctx
        return default_ctx

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

    async def _complete_chat_request(self, request: ChatRequest, **kwargs) -> OllamaChatResponse:
        """Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            OllamaChatResponse with content blocks
        """
        logger.info(f"[PROVIDER] Received ChatRequest with {len(request.messages)} messages")

        # Validate tool call sequences and repair if needed
        missing_tool_ids = self._find_missing_tool_results(request.messages)
        extra_tool_messages: list[dict[str, Any]] = []
        for tool_id in missing_tool_ids:
            logger.warning(f"Adding synthetic tool result for missing tool call: {tool_id}")
            extra_tool_messages.append(self._create_synthetic_tool_result(tool_id))

        # Separate messages by role
        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [m for m in request.messages if m.role in ("user", "assistant", "tool")]

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
        conversation_msgs = self._convert_messages([m.model_dump() for m in conversation])
        ollama_messages.extend(conversation_msgs)

        # Append synthetic tool results for any missing tool calls
        ollama_messages.extend(extra_tool_messages)

        # Prepare request parameters
        model = kwargs.get("model", self.default_model)
        params = {
            "model": model,
            "messages": ollama_messages,
            "options": {
                "temperature": request.temperature or kwargs.get("temperature", self.temperature),
                "num_predict": request.max_output_tokens or kwargs.get("max_tokens", self.max_tokens),
            },
        }

        # Set context window size (num_ctx controls how much context ollama uses)
        # If num_ctx is configured, use it; otherwise auto-detect from model
        if self.num_ctx > 0:
            params["options"]["num_ctx"] = self.num_ctx
        else:
            # Auto-detect context length from model info
            ctx_length = await self._get_model_context_length(model)
            params["options"]["num_ctx"] = ctx_length

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

        # Enable thinking/reasoning if requested or if provider config enables it
        # Models that don't support thinking will simply ignore the think option
        include_thinking = False
        if hasattr(request, "enable_thinking") and request.enable_thinking:
            params["options"]["think"] = True
            include_thinking = True
        elif self.enable_thinking:
            # Provider config enables thinking by default
            params["options"]["think"] = True
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

        # Call Ollama API with timeout
        try:
            raw_response = await asyncio.wait_for(
                self.client.chat(**params),
                timeout=self.timeout,
            )
            # Convert Pydantic model to dict for consistent access
            response = raw_response.model_dump() if hasattr(raw_response, "model_dump") else dict(raw_response)
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
            return self._convert_to_chat_response(response, include_thinking=include_thinking)

        except TimeoutError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[PROVIDER] Ollama API call timed out after {self.timeout}s")

            # Emit timeout event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "ollama",
                        "model": model,
                        "status": "timeout",
                        "duration_ms": elapsed_ms,
                        "error": f"Request timed out after {self.timeout}s",
                    },
                )
            raise TimeoutError(f"Ollama API call timed out after {self.timeout}s")

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
            raise

    async def _complete_streaming(self, request: ChatRequest, **kwargs) -> OllamaChatResponse:
        """Handle streaming completion with event emission.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            OllamaChatResponse with accumulated content
        """
        from amplifier_core.message_models import TextBlock
        from amplifier_core.message_models import Usage

        logger.info(f"[PROVIDER] Streaming request with {len(request.messages)} messages")

        # Validate tool call sequences (same as non-streaming)
        missing_tool_ids = self._find_missing_tool_results(request.messages)
        extra_tool_messages: list[dict[str, Any]] = []
        for tool_id in missing_tool_ids:
            logger.warning(f"Adding synthetic tool result for missing tool call: {tool_id}")
            extra_tool_messages.append(self._create_synthetic_tool_result(tool_id))

        # Separate messages by role
        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [m for m in request.messages if m.role in ("user", "assistant", "tool")]

        # Build ollama messages list
        ollama_messages = []

        for sys_msg in system_msgs:
            content = sys_msg.content if isinstance(sys_msg.content, str) else ""
            ollama_messages.append({"role": "system", "content": content})

        for dev_msg in developer_msgs:
            content = dev_msg.content if isinstance(dev_msg.content, str) else ""
            wrapped = f"<context_file>\n{content}\n</context_file>"
            ollama_messages.append({"role": "user", "content": wrapped})

        conversation_msgs = self._convert_messages([m.model_dump() for m in conversation])
        ollama_messages.extend(conversation_msgs)
        ollama_messages.extend(extra_tool_messages)

        # Prepare request parameters
        model = kwargs.get("model", self.default_model)
        params: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "options": {
                "temperature": request.temperature or kwargs.get("temperature", self.temperature),
                "num_predict": request.max_output_tokens or kwargs.get("max_tokens", self.max_tokens),
            },
            "stream": True,
        }

        # Set context window size (num_ctx controls how much context ollama uses)
        # If num_ctx is configured, use it; otherwise auto-detect from model
        if self.num_ctx > 0:
            params["options"]["num_ctx"] = self.num_ctx
        else:
            # Auto-detect context length from model info
            ctx_length = await self._get_model_context_length(model)
            params["options"]["num_ctx"] = ctx_length

        # Add tools if provided (note: tool calls may not work well with streaming)
        if request.tools:
            params["tools"] = self._format_tools_from_request(request.tools)

        # Add structured output format if specified
        if hasattr(request, "response_format") and request.response_format:
            if isinstance(request.response_format, dict):
                params["format"] = request.response_format
            elif request.response_format == "json":
                params["format"] = "json"

        # Enable thinking/reasoning if requested or if provider config enables it
        # Models that don't support thinking will simply ignore the think option
        include_thinking = False
        if hasattr(request, "enable_thinking") and request.enable_thinking:
            params["options"]["think"] = True
            include_thinking = True
        elif self.enable_thinking:
            # Provider config enables thinking by default
            params["options"]["think"] = True
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
        final_chunk: dict[str, Any] | None = None

        try:
            async for chunk in await asyncio.wait_for(
                self.client.chat(**params),
                timeout=self.timeout,
            ):
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
                                    "thinking": accumulated_thinking if accumulated_thinking else None,
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
                                "thinking": accumulated_thinking if accumulated_thinking else None,
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
                final_chunk,
                include_thinking,
            )

        except TimeoutError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[PROVIDER] Streaming timed out after {self.timeout}s")

            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "ollama",
                        "model": model,
                        "status": "timeout",
                        "duration_ms": elapsed_ms,
                        "stream": True,
                    },
                )
            raise TimeoutError(f"Ollama streaming timed out after {self.timeout}s")

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
            raise

    def _build_streaming_response(
        self,
        content: str,
        thinking: str,
        final_chunk: dict[str, Any] | None,
        include_thinking: bool,
    ) -> OllamaChatResponse:
        """Build final response from streamed chunks.

        Args:
            content: Accumulated content text
            thinking: Accumulated thinking text
            final_chunk: Final chunk with usage info
            include_thinking: Whether thinking was requested

        Returns:
            OllamaChatResponse with accumulated content
        """
        from amplifier_core.message_models import TextBlock
        from amplifier_core.message_models import Usage

        content_blocks = []  # For context storage (message_models: ThinkingBlock, TextBlock, etc.)
        event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []  # For streaming UI events
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

        # Extract usage from final chunk
        usage = Usage(
            input_tokens=final_chunk.get("prompt_eval_count", 0) if final_chunk else 0,
            output_tokens=final_chunk.get("eval_count", 0) if final_chunk else 0,
            total_tokens=(
                (final_chunk.get("prompt_eval_count", 0) + final_chunk.get("eval_count", 0)) if final_chunk else 0
            ),
        )

        return OllamaChatResponse(
            content=content_blocks,
            tool_calls=None,  # Tool calls not fully supported in streaming
            usage=usage,
            finish_reason=None,
            raw_response=final_chunk if self.raw_debug else None,
            model_name=final_chunk.get("model") if final_chunk else None,
            thinking_content=thinking_content,
            content_blocks=event_blocks if event_blocks else None,  # For streaming UI events
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

    def _find_missing_tool_results(self, messages: list[Message]) -> list[str]:
        """Find tool calls without corresponding results.

        Scans message history to detect tool calls that were never answered
        with a tool result message.

        Args:
            messages: List of conversation messages

        Returns:
            List of tool call IDs that are missing results
        """
        pending_tool_ids: set[str] = set()

        for msg in messages:
            if msg.role == "assistant":
                # Check for tool calls in content blocks
                if hasattr(msg, "content") and isinstance(msg.content, list):
                    for block in msg.content:
                        if hasattr(block, "type") and block.type == "tool_use":
                            pending_tool_ids.add(block.id)
                        elif hasattr(block, "id") and hasattr(block, "name"):
                            # ToolCallBlock style
                            pending_tool_ids.add(block.id)
                # Also check tool_calls field
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
                        if tc_id:
                            pending_tool_ids.add(tc_id)
            elif msg.role == "tool":
                # Tool result - remove from pending
                tool_call_id = msg.tool_call_id if hasattr(msg, "tool_call_id") else ""
                pending_tool_ids.discard(tool_call_id)

        return list(pending_tool_ids)

    def _create_synthetic_tool_result(self, tool_use_id: str) -> dict[str, Any]:
        """Create placeholder result for missing tool call.

        Args:
            tool_use_id: The ID of the tool call that needs a result

        Returns:
            Dict in tool message format with error content
        """
        return {
            "role": "tool",
            "tool_call_id": tool_use_id,
            "content": "Tool execution was interrupted or result was lost",
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
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        # TextContent block: {"type": "text", "text": "..."}
                        if block.get("type") == "text" and "text" in block:
                            text_parts.append(block["text"])
                        # ToolCallContent or other blocks - skip for now
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
                                "function": {"name": tc.get("tool", ""), "arguments": tc.get("arguments", {})},
                            }
                        )

                    ollama_messages.append({"role": "assistant", "content": content, "tool_calls": ollama_tool_calls})
                else:
                    # Regular assistant message
                    ollama_messages.append({"role": "assistant", "content": content})

            elif role == "tool":
                # Tool result message
                ollama_messages.append(
                    {"role": "tool", "content": content, "tool_call_id": msg.get("tool_call_id", "")}
                )

            else:
                # User, system, etc. - pass through
                ollama_messages.append(msg)

        return ollama_messages

    def _format_tools_for_ollama(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to Ollama format."""
        ollama_tools = []

        for tool in tools:
            # Get schema from tool if available
            input_schema = getattr(tool, "input_schema", {"type": "object", "properties": {}, "required": []})

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

    def _convert_to_chat_response(self, response: Any, include_thinking: bool = False) -> OllamaChatResponse:
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
        event_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []  # For streaming UI events
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
                tool_id = tc.get("id", "")
                tool_name = function.get("name", "")
                tool_args = function.get("arguments", {})

                content_blocks.append(ToolCallBlock(id=tool_id, name=tool_name, input=tool_args))
                tool_calls.append(ToolCall(id=tool_id, name=tool_name, arguments=tool_args))
                # Also add to event_blocks for streaming UI hooks
                event_blocks.append(ToolCallContent(id=tool_id, name=tool_name, arguments=tool_args))

        # Build usage info
        usage = Usage(
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
        )

        combined_text = "\n\n".join(text_accumulator).strip()

        return OllamaChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=None,  # Ollama doesn't provide finish_reason
            raw_response=response if self.raw_debug else None,
            model_name=response.get("model"),
            thinking_content=thinking_content,
            content_blocks=event_blocks if event_blocks else None,  # For streaming UI events
            text=combined_text or None,
        )
