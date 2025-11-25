#!/usr/bin/env python3
import argparse, joblib, pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True, help="Path to best_phase_model.joblib")
ap.add_argument("--X", required=True, help="CSV with SAME feature columns")
ap.add_argument("--out", required=True, help="Where to write predictions CSV")
args = ap.parse_args()

bundle = joblib.load(args.bundle)
pipe = bundle["pipeline"]
feat = bundle["feature_names"]

X = pd.read_csv(args.X)
missing = [c for c in feat if c not in X.columns]
if missing:
    raise SystemExit(f"Missing required features: {missing}")
X = X[feat]

y_pred = pipe.predict(X)
out = pd.DataFrame({"pred": y_pred})
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.out, index=False)
print(f"✅ Wrote predictions to {args.out}")
