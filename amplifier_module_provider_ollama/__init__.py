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
from amplifier_core.content_models import ToolCallContent
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Message
from amplifier_core.message_models import ToolCall

from ollama import AsyncClient
from ollama import ResponseError

logger = logging.getLogger(__name__)


class OllamaChatResponse(ChatResponse):
    """Extended ChatResponse with Ollama-specific metadata."""

    raw_response: dict[str, Any] | None = None
    model_name: str | None = None
    thinking_content: str | None = None


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

    async def complete(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """
        Generate completion from ChatRequest.

        Args:
            request: Typed chat request with messages, tools, config
            **kwargs: Provider-specific options (override request fields)

        Returns:
            ChatResponse with content blocks, tool calls, usage
        """
        return await self._complete_chat_request(request, **kwargs)

    async def _complete_chat_request(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content blocks
        """
        logger.info(f"[PROVIDER] Received ChatRequest with {len(request.messages)} messages")

        # Separate messages by role
        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [m for m in request.messages if m.role in ("user", "assistant")]

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

        # Add tools if provided
        if request.tools:
            params["tools"] = self._format_tools_from_request(request.tools)

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

            # DEBUG level: Full request payload (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "provider": "ollama",
                        "request": {
                            "model": model,
                            "messages": ollama_messages,
                            "options": params["options"],
                        },
                    },
                )

        start_time = time.time()

        # Call Ollama API with timeout
        try:
            response = await asyncio.wait_for(
                self.client.chat(**params),
                timeout=self.timeout,
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

                # DEBUG level: Full response (if debug enabled)
                if self.debug:
                    message = response.get("message", {})
                    content = message.get("content", "")
                    content_preview = content[:500] if content else ""
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "provider": "ollama",
                            "response": {
                                "content_preview": content_preview,
                            },
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

            # Convert to ChatResponse
            return self._convert_to_chat_response(response)

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

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """
        Parse tool calls from provider response.

        Args:
            response: Provider response

        Returns:
            List of tool calls
        """
        return response.tool_calls or []

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert Amplifier message format to Ollama/OpenAI format.

        Handles the conversion of:
        - Tool calls in assistant messages (Amplifier format -> OpenAI format)
        - Tool result messages
        - Developer messages (converted to XML-wrapped user messages)
        - Regular user/assistant/system messages
        """
        ollama_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

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

    def _convert_to_chat_response(self, response: Any) -> ChatResponse:
        """Convert Ollama response to ChatResponse format.

        Args:
            response: Ollama API response

        Returns:
            ChatResponse with content blocks
        """
        from amplifier_core.message_models import TextBlock
        from amplifier_core.message_models import ToolCall
        from amplifier_core.message_models import ToolCallBlock
        from amplifier_core.message_models import Usage

        content_blocks = []
        tool_calls = []

        message = response.get("message", {})
        content = message.get("content", "")

        # Add text content if present
        if content:
            content_blocks.append(TextBlock(text=content))

        # Parse tool calls if present
        if "tool_calls" in message:
            for tc in message["tool_calls"]:
                function = tc.get("function", {})
                tool_id = tc.get("id", "")
                tool_name = function.get("name", "")
                tool_args = function.get("arguments", {})

                content_blocks.append(ToolCallBlock(id=tool_id, name=tool_name, input=tool_args))
                tool_calls.append(ToolCall(id=tool_id, name=tool_name, arguments=tool_args))

        # Build usage info
        usage = Usage(
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
        )

        return ChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=None,  # Ollama doesn't provide finish_reason
        )
