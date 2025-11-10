# Ollama Provider Module

Local LLM provider integration for Amplifier using Ollama.

## Features

- Connect to local Ollama server
- Support for all Ollama-compatible models
- Tool calling support
- Streaming responses
- Automatic model pulling (optional)

## Configuration

```python
{
    "host": "http://localhost:11434",  # Ollama server URL (or set OLLAMA_HOST env var)
    "default_model": "llama3.2:3b",    # Default model to use
    "max_tokens": 4096,                # Maximum tokens to generate
    "temperature": 0.7,                # Generation temperature
    "debug": false,                    # Enable standard debug events
    "raw_debug": false                 # Enable ultra-verbose raw API I/O logging
    "timeout": 120,                    # Request timeout in seconds
    "auto_pull": false                 # Automatically pull missing models
}
```

### Debug Configuration

**Standard Debug** (`debug: true`):
- Emits `llm:request:debug` and `llm:response:debug` events
- Contains request/response summaries with message counts, model info, usage stats
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

#### Option 1: Automated Installation (Recommended)

Use the provided installer script:

```bash
# Install Ollama and pull default model
./scripts/install-ollama.sh --pull-model llama3.2:3b

# Or install Ollama only
./scripts/install-ollama.sh
```

#### Option 2: Manual Installation

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2:3b`
3. Start Ollama server (usually automatic)

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
- And many more...

See: https://ollama.ai/library

## Error Handling

The provider handles common scenarios gracefully:

- **Server offline**: Mounts successfully, fails on use with clear error
- **Model not found**: Pulls automatically (if auto_pull=true) or provides helpful error
- **Connection issues**: Clear error messages with troubleshooting hints

## Tool Calling

Supports tool calling with compatible models. Tools are automatically formatted in Ollama's expected format (OpenAI-compatible).

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
