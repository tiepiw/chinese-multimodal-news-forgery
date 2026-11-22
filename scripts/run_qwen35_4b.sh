#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-auto}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
TEMPERATURE="${TEMPERATURE:-0}"
NUM_CTX="${NUM_CTX:-16384}"
NUM_PREDICT="${NUM_PREDICT:-8192}"
TIMEOUT="${TIMEOUT:-600}"
cd "$ROOT"
for TASK in text_binary mm_binary forgery5; do
  "$PYTHON" -m Qwen.qwen_direct --root "$ROOT" --task "$TASK" \
    --ollama-url "$OLLAMA_URL" --temperature "$TEMPERATURE" \
    --num-ctx "$NUM_CTX" --num-predict "$NUM_PREDICT" --timeout "$TIMEOUT" "$@"
done
