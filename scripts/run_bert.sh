#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-auto}"

# Editable benchmark hyperparameters.
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-128}"
PATIENCE="${PATIENCE:-5}"
SEED="${SEED:-2026}"
LR="${LR:-5e-05}"
BERT_LR="${BERT_LR:-5e-05}"
VISION_LR="${VISION_LR:-5e-05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
SCHEDULE="${SCHEDULE:-linear}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"

exec "$PYTHON" "$ROOT/run_all.py" --models bert --tasks text_binary mm_binary forgery5 --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --patience "$PATIENCE" \
  --seed "$SEED" \
  --lr "$LR" \
  --bert-lr "$BERT_LR" \
  --vision-lr "$VISION_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --optimizer "$OPTIMIZER" \
  --schedule "$SCHEDULE" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  "$@"
