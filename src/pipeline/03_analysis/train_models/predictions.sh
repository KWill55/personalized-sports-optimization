#!/bin/bash
set -euo pipefail
ATHLETE=$(python3 -c "import yaml;print(yaml.safe_load(open('project_config.yaml'))['athlete'])")
SESSION=$(python3 -c "import yaml;print(yaml.safe_load(open('project_config.yaml'))['session'])")

BUNDLE="models/${ATHLETE}/${SESSION}/best_phase_model.joblib"
X_NEW="data/${ATHLETE}/${SESSION}/analysis/X.csv"         # or some new features file
OUT="models/${ATHLETE}/${SESSION}/predictions.csv"

[[ -f "$BUNDLE" ]] || { echo "❌ Missing $BUNDLE"; exit 1; }
[[ -f "$X_NEW"  ]] || { echo "❌ Missing $X_NEW";  exit 1; }

python scripts/03_analysis_and_modeling/train_models/predictions.py \
  --bundle "$BUNDLE" --X "$X_NEW" --out "$OUT"
