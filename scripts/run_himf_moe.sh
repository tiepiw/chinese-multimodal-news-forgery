#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-auto}"

# Editable joint-model parameters.
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_IMAGES="${MAX_IMAGES:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
NUM_WORKERS="${NUM_WORKERS:-0}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
ATTENTION_HEADS="${ATTENTION_HEADS:-4}"
DROPOUT="${DROPOUT:-0.2}"
BERT_LR="${BERT_LR:-2e-05}"
IMAGE_LR="${IMAGE_LR:-0.0001}"
HEAD_LR="${HEAD_LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TEXT_LOSS_WEIGHT="${TEXT_LOSS_WEIGHT:-1.0}"
BINARY_LOSS_WEIGHT="${BINARY_LOSS_WEIGHT:-1.0}"
FORGERY_LOSS_WEIGHT="${FORGERY_LOSS_WEIGHT:-1.5}"
IMAGE_LOSS_WEIGHT="${IMAGE_LOSS_WEIGHT:-0.5}"
ROUTER_LOSS_WEIGHT="${ROUTER_LOSS_WEIGHT:-0.2}"
CONTRASTIVE_LOSS_WEIGHT="${CONTRASTIVE_LOSS_WEIGHT:-0.1}"
CONSISTENCY_LOSS_WEIGHT="${CONSISTENCY_LOSS_WEIGHT:-0.1}"
BALANCE_LOSS_WEIGHT="${BALANCE_LOSS_WEIGHT:-0.02}"
CONTRASTIVE_TEMPERATURE="${CONTRASTIVE_TEMPERATURE:-0.2}"
MODALITY_DROPOUT="${MODALITY_DROPOUT:-0.1}"
BASELINE_VAL_TEXT="${BASELINE_VAL_TEXT:-0.860543}"
BASELINE_VAL_BINARY="${BASELINE_VAL_BINARY:-0.686559}"
BASELINE_VAL_FORGERY="${BASELINE_VAL_FORGERY:-0.536687}"
PATIENCE="${PATIENCE:-5}"
SPAUC_MAX_FPR="${SPAUC_MAX_FPR:-0.1}"
SEED="${SEED:-2026}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/joint/himf_moe/seed_$SEED}"

exec "$PYTHON" "$ROOT/HIMF_MoE/train.py" --device "$DEVICE" --output-dir "$OUTPUT_DIR" --no-require-valid-image --image-pretrained --amp \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --max-images "$MAX_IMAGES" \
  --image-size "$IMAGE_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --hidden-dim "$HIDDEN_DIM" \
  --attention-heads "$ATTENTION_HEADS" \
  --dropout "$DROPOUT" \
  --bert-lr "$BERT_LR" \
  --image-lr "$IMAGE_LR" \
  --head-lr "$HEAD_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --warmup-ratio "$WARMUP_RATIO" \
  --label-smoothing "$LABEL_SMOOTHING" \
  --grad-clip "$GRAD_CLIP" \
  --text-loss-weight "$TEXT_LOSS_WEIGHT" \
  --binary-loss-weight "$BINARY_LOSS_WEIGHT" \
  --forgery-loss-weight "$FORGERY_LOSS_WEIGHT" \
  --image-loss-weight "$IMAGE_LOSS_WEIGHT" \
  --router-loss-weight "$ROUTER_LOSS_WEIGHT" \
  --contrastive-loss-weight "$CONTRASTIVE_LOSS_WEIGHT" \
  --consistency-loss-weight "$CONSISTENCY_LOSS_WEIGHT" \
  --balance-loss-weight "$BALANCE_LOSS_WEIGHT" \
  --contrastive-temperature "$CONTRASTIVE_TEMPERATURE" \
  --modality-dropout "$MODALITY_DROPOUT" \
  --baseline-val-text "$BASELINE_VAL_TEXT" \
  --baseline-val-binary "$BASELINE_VAL_BINARY" \
  --baseline-val-forgery "$BASELINE_VAL_FORGERY" \
  --patience "$PATIENCE" \
  --spauc-max-fpr "$SPAUC_MAX_FPR" \
  --seed "$SEED" \
  "$@"
