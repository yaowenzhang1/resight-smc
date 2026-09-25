#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
DATASET="${DATASET:-logicvista}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
MODEL_TAG="${MODEL//\//_}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${RUN_ID:-resight_${MODEL_TAG}_${DATASET}_s${SEED}_${STAMP}}"

case "$DATASET" in
  logicvista|mathvista|mmstar_r) PROMPT_MODE=cot ;;
  mmstar_p|realworldqa) PROMPT_MODE=direct ;;
  *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"

python run.py \
  --config configs/default.yaml \
  --method resight \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --prompt-mode "$PROMPT_MODE" \
  --seed "$SEED" \
  --run-id "$RUN_ID" \
  "$@"
