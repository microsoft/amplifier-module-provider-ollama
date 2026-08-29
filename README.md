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

`host` is the **single source of truth** — local-vs-cloud is derived from
the URL. Set it to `http://localhost:11434` for a local Ollama, or
`https://ollama.com` for Ollama Cloud. The provider then automatically
picks sensible defaults (model, capability tags, pull behavior) for that
deployment.

```python
{
    "host": "http://localhost:11434",  # Ollama server URL (or set OLLAMA_HOST)
                                       # Use https://ollama.com for Ollama Cloud.
    "api_key": None,                   # Required for Ollama Cloud. Read from
                                       # OLLAMA_API_KEY env var if not set.
    "default_model": None,             # Defaults to "gpt-oss:120b" for cloud,
                                       # "llama3.2:3b" otherwise.
    "max_tokens": 4096,                # Maximum tokens to generate
    "temperature": 0.7,                # Generation temperature
    "timeout": 600,                    # Request timeout (seconds; 10 min default)
    "auto_pull": False,                # Auto-pull missing models (local only;
                                       # silently ignored for Ollama Cloud).
}
```

### All config keys

Only `host` and `api_key` are prompted by the setup wizard (they're the
connection essentials). Every key below is a fully supported config key --
set it directly in `settings.yaml` / the provider's `config:` block.

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `host` | string | `http://localhost:11434` | Single source of truth for local-vs-cloud (see above) |
| `api_key` | string | `None` | Ollama Cloud / auth-proxy Bearer token |
| `default_model` | string | host-derived | `gpt-oss:120b` (cloud) / `llama3.2:3b` (local) |
| `max_tokens` | int | `4096` | Maps to `num_predict` |
| `temperature` | float | `0.7` | |
| `timeout` | float (seconds) | `600` | Request timeout |
| `auto_pull` | bool | `False` | Auto-pull missing models (local only) |
| `enable_thinking` | bool | `True` | Enable thinking/reasoning for supported models |
| `thinking_effort` | string | `None` | `"low"`/`"medium"`/`"high"` |
| `keep_alive` | string | `None` | e.g. `"5m"`, `"-1"` for indefinite |
| `num_ctx` | int | `0` (auto-detect) | Context window override |
| `top_p`, `top_k`, `min_p`, `repeat_penalty`, `seed`, `stop` | various | unset | Forwarded to Ollama's `options` only when explicitly set |
| `logprobs`, `top_logprobs` | bool / int | unset | Requires Ollama >= 0.12.11 |
| `max_retries`, `min_retry_delay`, `max_retry_delay`, `retry_jitter` | various | `3` / `1.0` / `60.0` / `True` | Client-side retry policy |
| `use_streaming` | bool | `True` | Set `False` to force non-streaming completions |
| `raw` | bool | `False` | Attach exact request params to the `llm:request` event |
| `extra_request_params` | dict | `{}` | Merged last into Ollama's `options` -- an escape hatch for any Ollama-native option not listed above (e.g. `mirostat`, `repeat_last_n`) |
| `priority` | int | n/a | Read by the orchestrator's provider-selection logic, not by this module directly |

Boolean and numeric keys accept native types (`true`/`false`, `123`) or the
string forms a config wizard writes (`"true"`/`"false"`); invalid numeric
strings warn and fall back to the default rather than crashing at mount.
Unrecognized config keys produce a mount-time warning (with a did-you-mean
suggestion) rather than a silent no-op or a crash.

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
[[providers]]
module = "amplifier-module-provider-ollama"

[providers.config]
host = "http://localhost:11434"
default_model = "llama3.2:3b"
auto_pull = true
```

### Environment Variables

- `OLLAMA_HOST`: Override default Ollama server URL
- `OLLAMA_API_KEY`: API key for Ollama Cloud (only used when `host`
  points at a remote endpoint that requires Bearer auth)

## Authentication

The provider uses a single, simple convention:

- **Local Ollama** (`host = "http://localhost:11434"`): no auth.
- **Ollama Cloud** (`host = "https://ollama.com"`): set `api_key` (or
  `OLLAMA_API_KEY` env var). The provider attaches an
  `Authorization: Bearer <key>` header to every request.
- **Custom auth proxy** (any other URL with `api_key` set): the same
  `Authorization: Bearer <key>` header is attached. Useful when you
  front a local Ollama with a Bearer-auth reverse proxy.

The decision about whether to *attach* the header is governed solely by
whether `api_key` is present. The decision about *cloud-only behaviors*
(skipping `ollama pull`, defaulting to `gpt-oss:120b`, advertising the
`cloud` capability tag) is governed by the host URL — specifically a
URL-parsed match against `ollama.com` or any subdomain. Lookalike hosts
like `evil.ollama.com.attacker.io` are correctly rejected.

## Mixed local + cloud (multi-instance)

To use **both** local Ollama and Ollama Cloud in the same session — for
example, routing heavy reasoning to `gpt-oss:120b` on Ollama Cloud while
keeping `llama3.2:3b` local for fast utility tasks — configure two
provider instances. Amplifier's kernel supports multiple named instances
of the same module via the `instance_id` key:

```toml
# Default local instance — keeps the natural mount name "ollama"
[[providers]]
module = "amplifier-module-provider-ollama"
[providers.config]
host = "http://localhost:11434"
auto_pull = true

# Second instance — explicit `instance_id` makes it addressable as "ollama-cloud"
[[providers]]
module = "amplifier-module-provider-ollama"
instance_id = "ollama-cloud"
[providers.config]
host = "https://ollama.com"
api_key = "${OLLAMA_API_KEY}"
```

A routing matrix can then target each independently:

```yaml
roles:
  reasoning:
    candidates:
      - provider: ollama-cloud
        model: gpt-oss:120b
      - provider: ollama
        model: "deepseek-r1:*"
  fast:
    candidates:
      - provider: ollama
        model: "llama3.2:*"
```

> **Backward compat note.** Earlier releases of this provider exposed a
> `mode` config field (with values `local`/`cloud`) plus a duplicate-id
> `host` ConfigField gated by `mode`. Both have been removed in favor of
> the host-as-SSOT design above. Existing TOML configs containing a stray
> `mode` key are silently ignored — no re-init required. The `OLLAMA_HOST`
> and `OLLAMA_API_KEY` env vars continue to work unchanged.

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
