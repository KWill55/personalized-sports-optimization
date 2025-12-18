#!/bin/bash
# scripts/03_analysis_and_modeling/train_models/train_phase_models_cv.sh
set -euo pipefail

ATHLETE="$1"
SESSION="$2"

OUT_DIR="data/${ATHLETE}/${SESSION}/analysis/datasets"
X_PATH="${OUT_DIR}/X.csv"
Y_PATH="${OUT_DIR}/y.csv"

# sanity checks
[[ -f "$X_PATH" ]] || { echo "❌ Missing $X_PATH"; exit 1; }
[[ -f "$Y_PATH" ]] || { echo "❌ Missing $Y_PATH"; exit 1; }
mkdir -p "$OUT_DIR"

echo "🔧 Training with:"
echo "  X: $X_PATH"
echo "  y: $Y_PATH"
echo "  out_dir: $OUT_DIR"

python scripts/03_analysis_and_modeling/train_models/training.py \
  --X "$X_PATH" \
  --y "$Y_PATH" \
  --out_dir "$OUT_DIR"
