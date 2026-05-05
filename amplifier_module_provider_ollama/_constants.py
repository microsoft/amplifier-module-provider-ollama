"""Single source of truth for Ollama provider defaults.

Mirrors the convention used by amplifier-module-provider-openai
(`_constants.py:DEFAULT_MODEL`).
"""

# Cloud-mode default model. Direct API access at https://ollama.com uses
# the bare model id (no -cloud suffix); the -cloud suffix is reserved for
# the local-Ollama-with-cloud-offload path. Confirmed against the official
# Ollama Cloud docs (https://docs.ollama.com/cloud "Cloud API access").
CLOUD_DEFAULT_MODEL = "gpt-oss:120b"

# Local-mode default model. Small enough to run on most developer machines.
LOCAL_DEFAULT_MODEL = "llama3.2:3b"
