#!/usr/bin/env bash
# Usage: bash scripts/serve/vllm_gpt-oss-120b.sh [port] [extra vLLM arguments...]
# Install in the active environment: python -m pip install '.[vllm]'
# MODEL_PATH accepts a Hugging Face model ID or a local model directory.
# Set CUDA_VISIBLE_DEVICES to select GPUs; Ctrl+C stops the server.
# Requires cached model files and the Harmony o200k_base vocabulary.

set -euo pipefail

# configuration: edit and uncomment to override external values.
# Use a public model ID or replace it with your local model directory.
# export MODEL_PATH="openai/gpt-oss-120b"
# export VLLM_PORT="8000"  # A positional port argument takes precedence.
# export VLLM_HOST="127.0.0.1"
# export MAX_MODEL_LEN="131072"  # Otherwise derived from the model configuration.
# export GPU_MEMORY_UTILIZATION="0.975"
# export CUDA_VISIBLE_DEVICES="0"  # Example GPU selection; otherwise inherited.
# export TIKTOKEN_RS_CACHE_DIR="${HOME}/.cache/tiktoken"  # Prepopulate for offline use.

PORT="${1:-${VLLM_PORT:-}}"
if [[ $# -gt 0 ]]; then
    shift
fi

EXTRA_ARGS=()
if [[ -n "${PORT}" ]]; then
    EXTRA_ARGS+=(--port "${PORT}")
fi
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
    EXTRA_ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi

exec vllm serve "${MODEL_PATH:-openai/gpt-oss-120b}" \
    --served-model-name gpt-oss-120b \
    --host "${VLLM_HOST:-127.0.0.1}" \
    --tool-call-parser openai \
    --reasoning-parser openai_gptoss \
    --enable-auto-tool-choice \
    --generation-config vllm \
    --override-generation-config '{"max_tokens":8192}' \
    --enforce-eager \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.975}" \
    --kv-cache-dtype fp8 \
    "${EXTRA_ARGS[@]}" \
    "$@"