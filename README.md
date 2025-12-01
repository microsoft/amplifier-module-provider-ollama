# Ollama Provider Module

Local LLM provider integration for Amplifier using Ollama.

## Features

- Connect to local Ollama server
- Support for all Ollama-compatible models
- Tool calling with automatic validation and repair
- Streaming responses with real-time events
- Thinking/reasoning support for compatible models
- Structured output with JSON schema validation
- Automatic model pulling (optional)

## Configuration

```python
{
    "host": "http://localhost:11434",  # Ollama server URL (or set OLLAMA_HOST env var)
    "default_model": "llama3.2:3b",    # Default model to use
    "max_tokens": 4096,                # Maximum tokens to generate
    "temperature": 0.7,                # Generation temperature
    "timeout": 300,                    # Request timeout in seconds (default: 5 minutes)
    "debug": false,                    # Enable standard debug events
    "raw_debug": false,                # Enable ultra-verbose raw API I/O logging
    "auto_pull": false                 # Automatically pull missing models
}
```

### Debug Configuration

**Standard Debug** (`debug: true`):
- Emits `llm:request:debug` and `llm:response:debug` events
- Contains request/response summaries with message counts, model info, usage stats
- Long values automatically truncated for readability
- Moderate log volume, suitable for development

**Raw Debug** (`debug: true, raw_debug: true`):
- Emits `llm:request:raw` and `llm:response:raw` events
- Contains complete, unmodified request params and response objects
- Extreme log volume, use only for deep provider integration debugging
- Captures the exact data sent to/from Ollama API before any processing

**Example**:
```yaml
providers:
  - module: provider-ollama
    config:
      debug: true      # Enable debug events
      raw_debug: true  # Enable raw API I/O capture
      default_model: llama3.2:3b
```

## Usage

### Prerequisites

#### Installation

1. **Install Ollama**: Download from https://ollama.ai or use:
   ```bash
   # Linux
   curl -fsSL https://ollama.com/install.sh | sh

   # macOS (with Homebrew)
   brew install ollama
   ```

2. **Pull a model**:
   ```bash
   ollama pull llama3.2:3b
   ```

3. **Start Ollama server** (usually starts automatically after installation)

### Configuration File

```toml
[provider]
name = "ollama"
model = "llama3.2:3b"
host = "http://localhost:11434"
auto_pull = true
```

### Environment Variables

- `OLLAMA_HOST`: Override default Ollama server URL

## Supported Models

Any model available in Ollama:

- llama3.2:3b (small, fast)
- llama3.2:1b (tiny, fastest)
- mistral (7B)
- mixtral (8x7B)
- codellama (code generation)
- deepseek-r1 (reasoning/thinking)
- qwen3 (reasoning + tools)
- And many more...

See: https://ollama.ai/library

## Thinking/Reasoning Support

The provider supports thinking/reasoning for compatible models like DeepSeek R1 and Qwen 3. When enabled, the model's internal reasoning is captured separately from the final response.

**Enable thinking in your request**:
```python
request = ChatRequest(
    model="deepseek-r1",
    messages=[...],
    enable_thinking=True
)
```

**Response structure**:
The response includes both the thinking process and the final answer as separate content blocks:
- `ThinkingBlock`: Contains the model's reasoning process
- `TextBlock`: Contains the final response

**Compatible models**:
- `deepseek-r1` - DeepSeek's reasoning model
- `qwen3` - Alibaba's Qwen 3 (with `think` parameter)
- `qwq` - Alibaba's QwQ reasoning model
- `phi4-reasoning` - Microsoft's Phi-4 reasoning variant

## Streaming

The provider supports streaming responses for real-time token delivery. When streaming is enabled, events are emitted as tokens arrive.

**Enable streaming**:
```python
request = ChatRequest(
    model="llama3.2:3b",
    messages=[...],
    stream=True
)
```

**Stream events**:
- `llm:stream:chunk` - Emitted for each content token
- `llm:stream:thinking` - Emitted for thinking tokens (when thinking enabled)

The final response contains the complete accumulated content.

## Structured Output

The provider supports structured output using JSON schemas. This ensures the model's response conforms to a specific format.

**Request JSON output**:
```python
request = ChatRequest(
    model="llama3.2:3b",
    messages=[...],
    response_format="json"  # Simple JSON mode
)
```

**Request schema-validated output**:
```python
request = ChatRequest(
    model="llama3.2:3b",
    messages=[...],
    response_format={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
)
```

## Tool Calling

Supports tool calling with compatible models. Tools are automatically formatted in Ollama's expected format (OpenAI-compatible).

**Automatic validation**: The provider validates tool call sequences and repairs broken chains. If a tool call is missing its result, a synthetic error result is inserted to maintain conversation integrity.

**Compatible models**:
- Llama 3.1+ (8B, 70B, 405B)
- Llama 3.2 (1B, 3B)
- Qwen 3
- Mistral Nemo
- And others with tool support

## Error Handling

The provider handles common scenarios gracefully:

- **Server offline**: Mounts successfully, fails on use with clear error
- **Model not found**: Pulls automatically (if auto_pull=true) or provides helpful error
- **Connection issues**: Clear error messages with troubleshooting hints
- **Timeout**: Configurable timeout with clear error when exceeded

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
